from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .common import CocoMini, parse_class_subset, run_logged_subprocess, setup_logger, subset_label, write_yolo_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-list", default=None, help="Pre-exported YOLO val image list. Skips COCO->YOLO export when set.")
    parser.add_argument("--coco-annotations", default=None, help="Pre-exported COCO annotation file for YOLOv7 json evaluation.")
    parser.add_argument("--yolov7", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--run-name", default="eval_depth_adapt")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "datasets").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    val = CocoMini(args.val_annotations, args.val_images, class_subset=class_subset)
    logger.info("class subset: %s", subset_label(class_subset))
    val_ids = [x[0] for x in val.iter_images(None)]
    if args.val_list and args.coco_annotations:
        val_list = Path(args.val_list).resolve()
        ann_file = Path(args.coco_annotations).resolve()
        logger.info("using shared YOLO val dataset: val_list=%s coco_annotations=%s", val_list, ann_file)
    else:
        val_list = val.export_yolo(output / "datasets" / "val", val_ids)
        ann_file = val.export_coco_annotations(output / "datasets" / "instances_val_subset.json", val_ids)
        logger.info("exported local YOLO val dataset: val_list=%s coco_annotations=%s", val_list, ann_file)
    data_yaml = output / "datasets" / "eval.yaml"
    write_yolo_yaml(data_yaml, val_list, val_list, val.class_names)
    val_list.with_suffix(".cache").unlink(missing_ok=True)
    logger.info("evaluation start: val_images=%d val_annotations=%d weights=%s data_yaml=%s",
                len(val_ids), sum(len(v) for v in val.annotations.values()), args.weights, data_yaml)
    command = [
        sys.executable, str(Path(args.yolov7) / "test.py"),
        "--data", str(data_yaml.resolve()),
        "--img", str(args.image_size),
        "--batch", str(args.batch_size),
        "--device", args.device,
        "--weights", str(Path(args.weights).resolve()),
        "--task", "val",
        "--project", str((output / "runs").resolve()),
        "--name", args.run_name,
        "--exist-ok",
        "--save-json",
    ]
    env = os.environ.copy()
    env["YOLO_IS_COCO"] = "1"
    env["COCO_ANNOTATIONS"] = str(ann_file.resolve())
    log_file = output / "yolov7_eval.log"
    logger.info("COCO annotations for eval: %s", ann_file)
    # Use subprocess directly here to pass the COCO annotation environment.
    from subprocess import Popen, STDOUT, CalledProcessError
    logger.info("command: %s", " ".join(command))
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        process = Popen(command, cwd=Path(args.yolov7), env=env, stdout=-1, stderr=STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            handle.write(line + "\n")
            handle.flush()
            logger.info("[yolov7-eval] %s", line)
        code = process.wait()
        if code:
            raise CalledProcessError(code, command)
    metrics_file = output / "runs" / args.run_name / "coco_metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("evaluation complete: metrics=%s metrics_file=%s", metrics, output / "metrics.json")


if __name__ == "__main__":
    main()
