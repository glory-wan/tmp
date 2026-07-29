from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .coco import CocoDataset, write_yolo_data_yaml
from .config import ensure_dirs, load_config
from .detector import MockDetector, YoloV7Detector, run_yolov7_eval, run_yolov7_train
from .failures import classify_failures, load_failures, save_failures, stratified_limit
from .generate import HardExampleGenerator, merge_image_lists
from .logging_utils import setup_logging, stage_log_path


class ClosedLoop:
    def __init__(self, config: str | Path):
        self.config = Path(config)
        self.cfg = load_config(config)
        ensure_dirs(self.cfg)
        self.workspace = Path(self.cfg["paths"]["workspace"])
        self.logger = setup_logging(self.workspace)
        self.logger.info("config loaded: %s", self.config)
        self.logger.info("workspace: %s", self.workspace.resolve())
        dcfg = self.cfg["dataset"]
        root = Path(self.cfg["paths"]["dataset"])
        self.train = CocoDataset(root / dcfg["train_annotations"], root / dcfg["train_images"])
        self.val = CocoDataset(root / dcfg["val_annotations"], root / dcfg["val_images"])
        self.logger.info("dataset loaded: train_annotations=%s train_images=%s images=%d annotations=%d",
                         self.train.annotation_file, self.train.images_dir,
                         len(self.train.images), sum(len(v) for v in self.train.annotations.values()))
        self.logger.info("dataset loaded: val_annotations=%s val_images=%s images=%d annotations=%d",
                         self.val.annotation_file, self.val.images_dir,
                         len(self.val.images), sum(len(v) for v in self.val.annotations.values()))

    def prepare(self):
        self.logger.info("stage prepare start")
        dcfg = self.cfg["dataset"]
        out = self.workspace / "datasets" / "real"
        train_ids = [x[0] for x in self.train.iter_images(dcfg.get("max_train_images"))]
        val_ids = [x[0] for x in self.val.iter_images(dcfg.get("max_val_images"))]
        self.logger.info("prepare image scan: train=%d val=%d max_train=%s max_val=%s",
                         len(train_ids), len(val_ids), dcfg.get("max_train_images"), dcfg.get("max_val_images"))
        train_list = self.train.export_yolo(out / "train", train_ids)
        val_list = self.val.export_yolo(out / "val", val_ids)
        data_yaml = out / "coco_mini.yaml"
        write_yolo_data_yaml(data_yaml, train_list, val_list, self.train.class_names)
        self.logger.info("stage prepare complete: data_yaml=%s train_list=%s val_list=%s", data_yaml, train_list, val_list)
        return data_yaml

    def detector(self, weights: Path | None = None):
        dcfg = self.cfg["detector"]
        if dcfg["backend"] == "mock":
            self.logger.info("detector init: backend=mock seed=%s", self.cfg["seed"])
            return MockDetector(seed=self.cfg["seed"])
        self.logger.info("detector init: backend=yolov7 weights=%s device=%s imgsz=%s",
                         weights or self.cfg["paths"]["weights"], dcfg["device"], dcfg["image_size"])
        return YoloV7Detector(self.cfg["paths"]["yolov7"], weights or self.cfg["paths"]["weights"],
                              dcfg["device"], dcfg["image_size"])

    def mine(self, round_index: int, weights: Path | None = None):
        self.logger.info("stage mine start: round=%d weights=%s", round_index, weights or self.cfg["paths"]["weights"])
        detector, failures = self.detector(weights), []
        state_file = self.workspace / "state" / "forgotten.json"
        forgotten = json.loads(state_file.read_text()) if state_file.exists() else {}
        images = list(self.train.iter_images(self.cfg["mining"].get("max_images")))
        self.logger.info("failure mining data: images=%d max_images=%s conf=%s match_iou=%s",
                         len(images), self.cfg["mining"].get("max_images"),
                         self.cfg["mining"]["prediction_confidence"], self.cfg["mining"]["match_iou"])
        mined_images = 0
        for image_id, path, annotations in tqdm(images, desc="failure mining"):
            predictions = detector.predict(Image.open(path).convert("RGB"), conf=self.cfg["mining"]["prediction_confidence"])
            failures.extend(classify_failures(image_id, annotations, predictions, self.cfg["mining"], forgotten))
            mined_images += 1
            if mined_images == 1 or mined_images % 250 == 0 or mined_images == len(images):
                self.logger.info("failure mining progress: images=%d/%d failures_raw=%d",
                                 mined_images, len(images), len(failures))
        raw_count = len(failures)
        failures = stratified_limit(failures, self.cfg["mining"]["max_per_prompt"])
        output = self.workspace / "failures" / f"round_{round_index}.json"
        save_failures(output, failures, {"round": round_index, "source": "real_only", "weights": str(weights or self.cfg["paths"]["weights"])})
        state_file.write_text(json.dumps(forgotten, indent=2))
        by_type = {}
        for item in failures:
            by_type[item.failure_type] = by_type.get(item.failure_type, 0) + 1
        self.logger.info("stage mine complete: raw_failures=%d kept_failures=%d by_type=%s output=%s",
                         raw_count, len(failures), by_type, output)
        return output

    def optimize(self, round_index: int, weights: Path | None = None):
        from .prompt_optimizer import ObjectPromptOptimizer
        failure_file = self.workspace / "failures" / f"round_{round_index}.json"
        failures = load_failures(failure_file)
        self.logger.info("stage optimize start: round=%d failures=%d tokens_per_prompt=%s steps=%s init_mode=%s",
                         round_index, len(failures), self.cfg["prompt_optimization"]["tokens_per_prompt"],
                         self.cfg["prompt_optimization"]["steps"],
                         self.cfg["prompt_optimization"].get("init_mode", "class_name"))
        output = self.workspace / "prompts" / f"round_{round_index}.pt"
        result = ObjectPromptOptimizer(self.cfg, self.detector(weights), logger=self.logger).optimize(
            failure_file, self.train, output)
        self.logger.info("stage optimize complete: output=%s metadata=%s", result, result.with_suffix(".json"))
        return result

    def generate(self, round_index: int):
        self.logger.info("stage generate start: round=%d", round_index)
        failure_file = self.workspace / "failures" / f"round_{round_index}.json"
        output = self.workspace / "synthetic" / f"round_{round_index}"
        generator = HardExampleGenerator(self.cfg, self.workspace / "prompts" / f"round_{round_index}.pt")
        failures = load_failures(failure_file)
        self.logger.info("generation input: failures=%d max_images=%s max_objects_per_image=%s",
                         len(failures), self.cfg["generation"]["max_images"],
                         self.cfg["generation"]["max_objects_per_image"])
        manifest = generator.generate(self.train, failures, output, round_index, logger=self.logger)
        self.logger.info("stage generate complete: synthetic_images=%d output=%s", len(manifest), output)
        return manifest

    def finetune(self, round_index: int, weights: Path | None = None):
        self.logger.info("stage finetune start: round=%d base_weights=%s", round_index, weights or self.cfg["paths"]["weights"])
        data_yaml = self.workspace / "datasets" / f"round_{round_index}.yaml"
        real_yaml = self.workspace / "datasets" / "real" / "coco_mini.yaml"
        import yaml
        data = yaml.safe_load(real_yaml.read_text())
        merged = self.workspace / "datasets" / f"round_{round_index}_train.txt"
        merge_image_lists(Path(data["train"]), self.workspace / "synthetic" / f"round_{round_index}", merged)
        write_yolo_data_yaml(data_yaml, merged, Path(data["val"]), self.train.class_names)
        train_count = len([x for x in merged.read_text().splitlines() if x])
        val_count = len([x for x in Path(data["val"]).read_text().splitlines() if x])
        self.logger.info("finetune dataset merged: train_images=%d val_images=%d data_yaml=%s", train_count, val_count, data_yaml)
        # YOLOv7 caches label discovery next to image-list files. A new round
        # can reuse the same filename with different symlink targets, so force
        # cache rebuilding after every merge.
        for image_list in (merged, Path(data["val"])):
            image_list.with_suffix(".cache").unlink(missing_ok=True)
        result = run_yolov7_train(Path(self.cfg["paths"]["yolov7"]), data_yaml,
                                  Path(weights or self.cfg["paths"]["weights"]), self.workspace / "runs",
                                  {**self.cfg["finetune"], "run_name": f"finetune_round_{round_index}"},
                                  logger=self.logger,
                                  log_file=stage_log_path(self.workspace, f"yolov7_train_round_{round_index}"))
        self.logger.info("stage finetune complete: checkpoint=%s", result)
        return result

    def evaluate(self, round_index: int, weights: Path):
        self.logger.info("stage evaluate start: round=%d weights=%s", round_index, weights)
        metrics = run_yolov7_eval(Path(self.cfg["paths"]["yolov7"]), self.workspace / "datasets" / "real" / "coco_mini.yaml",
                                  weights, self.workspace / "runs", {**self.cfg["evaluation"],
                                  "annotation_file": str(self.val.annotation_file.resolve()),
                                  "run_name": f"eval_round_{round_index}"},
                                  logger=self.logger,
                                  log_file=stage_log_path(self.workspace, f"yolov7_eval_round_{round_index}"))
        self.logger.info("stage evaluate complete: metrics=%s", metrics)
        return metrics

    def all(self):
        self.logger.info("closed-loop pipeline start: rounds=%d", self.cfg["rounds"])
        self.prepare()
        weights = Path(self.cfg["paths"]["weights"])
        summary = []
        for round_index in range(self.cfg["rounds"]):
            self.logger.info("round start: %d", round_index)
            failure_file = self.mine(round_index, weights)
            self.optimize(round_index, weights)
            generated = self.generate(round_index)
            weights = self.finetune(round_index, weights)
            metrics = self.evaluate(round_index, weights)
            summary.append({"round": round_index, "failures": str(failure_file), "generated": len(generated), **metrics})
            self.logger.info("round complete: %d summary=%s", round_index, summary[-1])
        (self.workspace / "summary.json").write_text(json.dumps(summary, indent=2))
        self.logger.info("closed-loop pipeline complete: summary=%s", self.workspace / "summary.json")


def main():
    parser = argparse.ArgumentParser(description="YOLOv7 adversarial-prompt closed loop")
    parser.add_argument("command", choices=("prepare", "mine", "optimize", "generate", "finetune", "evaluate", "all"))
    parser.add_argument("--config", default="configs/detection_coco_mini.yaml")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--weights")
    args = parser.parse_args()
    loop = ClosedLoop(args.config)
    weights = Path(args.weights) if args.weights else None
    result = getattr(loop, args.command)(args.round, weights) if args.command in ("mine", "optimize", "finetune") else (
        loop.evaluate(args.round, weights) if args.command == "evaluate" else
        getattr(loop, args.command)(args.round) if args.command == "generate" else getattr(loop, args.command)())
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
