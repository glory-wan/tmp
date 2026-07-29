from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from .common import CocoMini, parse_class_subset, run_logged_subprocess, setup_logger, subset_label


def run(command: list[str], logger, workspace: Path, stage: str) -> None:
    run_logged_subprocess(command, Path.cwd(), logger, workspace / "logs" / f"{stage}.log", prefix=stage)


def resolved(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def best_or_last(run_output: Path, run_name: str) -> Path:
    weights_dir = run_output / "runs" / run_name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    return best if best.exists() else last


def existing_checkpoint(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    return path


def load_failure_stats(path: Path) -> tuple[int, dict[str, int]]:
    if not path.exists():
        return 0, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = data.get("failures", [])
    return len(failures), dict(Counter(str(x.get("failure_type", "unknown")) for x in failures))


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def extract_map(metrics: dict):
    for key in ("map", "mAP", "mAP_0.5:0.95", "mAP50-95", "AP", "box_map"):
        if key in metrics:
            return metrics[key]
    stats = metrics.get("stats")
    if isinstance(stats, list) and stats:
        return stats[0]
    return None


def image_stem(path: Path) -> str:
    return path.stem


def select_synthetic_images(synthetic_dir: Path, limit: int, seed: int, round_idx: int, logger) -> list[Path]:
    image_dir = synthetic_dir / "images"
    candidates = sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.png")])
    if len(candidates) > limit:
        rng = random.Random(seed + round_idx)
        selected = sorted(rng.sample(candidates, limit))
        selected_names = [x.name for x in selected]
        (synthetic_dir / "selected_images.json").write_text(json.dumps(selected_names, indent=2), encoding="utf-8")
        logger.info("synthetic selection: generated=%d selected=%d file=%s",
                    len(candidates), len(selected), synthetic_dir / "selected_images.json")
        return selected
    logger.info("synthetic selection: generated=%d selected=%d limit=%d", len(candidates), len(candidates), limit)
    return candidates


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_synthetic_pool(round_dir: Path, selected_by_round: list[list[Path]], logger) -> Path:
    pool = round_dir / "synthetic_pool"
    images_out = pool / "images"
    labels_out = pool / "labels"
    if pool.exists():
        shutil.rmtree(pool)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing_labels = 0
    for source_round, images in enumerate(selected_by_round, start=1):
        for image_path in images:
            label_path = image_path.parent.parent / "labels" / f"{image_stem(image_path)}.txt"
            prefix = f"r{source_round}_"
            link_file(image_path, images_out / f"{prefix}{image_path.name}")
            if label_path.exists():
                link_file(label_path, labels_out / f"{prefix}{label_path.name}")
            else:
                missing_labels += 1
            linked += 1
    logger.info("synthetic pool built: pool=%s images=%d missing_labels=%d", pool, linked, missing_labels)
    return pool


def write_round_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def latest_prompt_checkpoint(prompts_dir: Path) -> Path | None:
    meta_path = prompts_dir / "object_prompts.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            latest = meta.get("latest")
        except json.JSONDecodeError:
            latest = None
        if latest and (prompts_dir / latest).exists():
            return prompts_dir / latest
    checkpoints = sorted(prompts_dir.glob("learned_embeds-*.bin"))
    return checkpoints[-1] if checkpoints else None


def checkpoint_for_round(workspace: Path, round_idx: int) -> Path | None:
    checkpoint_dir = workspace / f"round_{round_idx}" / "checkpoint"
    alias = checkpoint_dir / "best.pt"
    if alias.exists():
        return alias
    return_checkpoint = best_or_last(checkpoint_dir, f"finetune_round_{round_idx}")
    return return_checkpoint if return_checkpoint.exists() else None


def input_weights_for_round(workspace: Path, base_weights: Path, round_idx: int) -> Path:
    if round_idx > 1:
        previous = checkpoint_for_round(workspace, round_idx - 1)
        if previous is not None:
            return previous
    warmup = workspace / "warmup" / "best.pt"
    return warmup if warmup.exists() else base_weights


def prepare_shared_yolo_datasets(root: Path, workspace: Path, data: dict, class_subset: set[int] | None, logger) -> dict[str, Path]:
    """Export real COCO train/val once under workspace/datasets and reuse in all rounds.

    Synthetic data and per-stage data.yaml files still change round by round,
    but real_train/real_val images and labels are no longer regenerated under
    every warmup/checkpoint/evaluation directory.
    """
    shared = workspace / "datasets"
    manifest_path = shared / "manifest.json"
    expected = {
        "train_annotations": str(resolved(root, data["train_annotations"]).resolve()),
        "train_images": str(resolved(root, data["train_images"]).resolve()),
        "val_annotations": str(resolved(root, data["val_annotations"]).resolve()),
        "val_images": str(resolved(root, data["val_images"]).resolve()),
        "class_subset": subset_label(class_subset),
    }
    required = {
        "real_train": shared / "real_train" / "images.txt",
        "real_val": shared / "real_val" / "images.txt",
        "val_coco": shared / "instances_val_subset.json",
    }
    if manifest_path.exists() and all(path.exists() for path in required.values()):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("inputs") == expected:
            logger.info("shared YOLO datasets reused: root=%s class_subset=%s", shared, expected["class_subset"])
            return required

    if shared.exists():
        shutil.rmtree(shared)
    shared.mkdir(parents=True, exist_ok=True)
    logger.info("shared YOLO datasets export start: root=%s class_subset=%s", shared, expected["class_subset"])
    train = CocoMini(expected["train_annotations"], expected["train_images"], class_subset=class_subset)
    val = CocoMini(expected["val_annotations"], expected["val_images"], class_subset=class_subset)
    train_ids = [x[0] for x in train.iter_images(None)]
    val_ids = [x[0] for x in val.iter_images(None)]
    real_train = train.export_yolo(shared / "real_train", train_ids)
    real_val = val.export_yolo(shared / "real_val", val_ids)
    val_coco = val.export_coco_annotations(shared / "instances_val_subset.json", val_ids)
    manifest = {
        "inputs": expected,
        "real_train_images": len(train_ids),
        "real_train_annotations": sum(len(v) for v in train.annotations.values()),
        "real_val_images": len(val_ids),
        "real_val_annotations": sum(len(v) for v in val.annotations.values()),
        "real_train": str(real_train),
        "real_val": str(real_val),
        "val_coco": str(val_coco),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("shared YOLO datasets export complete: %s", manifest)
    return {"real_train": real_train.resolve(), "real_val": real_val.resolve(), "val_coco": val_coco.resolve()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-style YOLOv7 hard-example closed-loop runner.")
    parser.add_argument("command", choices=["warmup", "mine", "optimize", "generate", "finetune", "evaluate", "round", "all"])
    parser.add_argument("--config", default="detection_depth_adapt/configs/coco_mini.yaml")
    parser.add_argument("--round", type=int, default=1, help="Round index for single-stage debug commands.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of closed-loop rounds. Defaults to config or 3.")
    parser.add_argument("--classes-subset", type=int, default=None, help="Enable class subset mode with the first N COCO class ids.")
    parser.add_argument("--class-ids", default=None, help="Enable class subset mode with comma-separated COCO class ids, e.g. 0,2,16.")
    args = parser.parse_args()

    root = Path.cwd()
    cfg_path = resolved(root, args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    subset_cfg = cfg.get("class_subset", {})
    subset_enabled = bool(subset_cfg.get("enabled", False))
    subset_count = args.classes_subset if args.classes_subset is not None else subset_cfg.get("count")
    subset_ids = args.class_ids if args.class_ids is not None else subset_cfg.get("class_ids")
    if args.classes_subset is not None or args.class_ids is not None:
        subset_enabled = True
    class_subset = parse_class_subset(subset_enabled, subset_count, subset_ids)
    subset_args: list[str] = []
    if class_subset is not None:
        if args.class_ids is not None or subset_ids:
            subset_args = ["--class-ids", subset_label(class_subset)]
        else:
            subset_args = ["--classes-subset", str(len(class_subset))]

    label_cfg = cfg.get("label_scope", {})
    train_eval_label_scope = str(label_cfg.get("train_eval", cfg.get("train_eval_label_scope", "subset"))).lower()
    if train_eval_label_scope not in {"subset", "full"}:
        raise ValueError("label_scope.train_eval must be either 'subset' or 'full'")
    train_eval_class_subset = None if train_eval_label_scope == "full" else class_subset
    train_eval_subset_args: list[str] = []
    if train_eval_class_subset is not None:
        if args.class_ids is not None or subset_ids:
            train_eval_subset_args = ["--class-ids", subset_label(train_eval_class_subset)]
        else:
            train_eval_subset_args = ["--classes-subset", str(len(train_eval_class_subset))]

    workspace = resolved(root, cfg["workspace"])
    logger = setup_logger(workspace)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)

    loop_cfg = cfg.get("closed_loop", {})
    rounds = int(args.rounds if args.rounds is not None else loop_cfg.get("rounds", 3))
    warmup_epochs = int(loop_cfg.get("warmup_epochs", cfg.get("warmup", {}).get("epochs", 70)))
    finetune_epochs = int(loop_cfg.get("finetune_epochs", cfg.get("finetune", {}).get("epochs", 10)))
    synthetic_images_per_round = int(loop_cfg.get("synthetic_images_per_round", cfg.get("generation", {}).get("max_images", 2500)))
    skip_warmup = bool(loop_cfg.get("skip_warmup", False))
    warmup_weights = loop_cfg.get("warmup_weights")

    data = cfg["data"]
    yolo = cfg["yolov7"]
    sd = cfg["stable_diffusion"]
    base_weights = resolved(root, yolo["weights"])
    state = workspace / "state" / "forgotten.json"
    shared_datasets = prepare_shared_yolo_datasets(root, workspace, data, train_eval_class_subset, logger)

    logger.info(
        "runner start: command=%s config=%s workspace=%s mining_generation_class_subset=%s train_eval_label_scope=%s train_eval_class_subset=%s rounds=%d warmup_epochs=%d skip_warmup=%s warmup_weights=%s finetune_epochs=%d synthetic_images_per_round=%d",
        args.command, cfg_path.resolve(), workspace.resolve(), subset_label(class_subset),
        train_eval_label_scope, subset_label(train_eval_class_subset), rounds, warmup_epochs,
        skip_warmup, warmup_weights, finetune_epochs, synthetic_images_per_round,
    )
    if train_eval_label_scope == "full" and class_subset is not None:
        logger.info(
            "label policy: mine/optimize/generate are restricted to classes %s; newly run warmup/finetune/evaluate dataset exports use full 80-class labels",
            subset_label(class_subset),
        )
        if skip_warmup:
            logger.info("label policy note: skip_warmup=true, so the reused warmup checkpoint label scope depends on how that checkpoint was trained")

    def run_mine(round_idx: int, current_weights: Path) -> Path:
        round_dir = workspace / f"round_{round_idx}"
        failures = round_dir / "failures.json"
        logger.info("round_%d mine config: train_annotations=%s train_images=%s weights=%s max_images=%s",
                    round_idx, resolved(root, data["train_annotations"]), resolved(root, data["train_images"]),
                    current_weights, cfg["mining"]["max_images"])
        run([
            sys.executable, "-m", "detection_depth_adapt.mine_failures",
            "--annotations", str(resolved(root, data["train_annotations"])),
            "--images", str(resolved(root, data["train_images"])),
            "--yolov7", str(resolved(root, yolo["repo"])),
            "--weights", str(current_weights),
            "--output", str(failures),
            "--state", str(state),
            "--device", str(yolo["device"]),
            "--image-size", str(yolo["image_size"]),
            "--conf", str(cfg["mining"]["conf"]),
            "--match-iou", str(cfg["mining"]["match_iou"]),
            "--low-confidence", str(cfg["mining"]["low_confidence"]),
            "--ambiguous-gap", str(cfg["mining"]["ambiguous_gap"]),
            "--forgotten-threshold", str(cfg["mining"]["forgotten_threshold"]),
            "--max-images", str(cfg["mining"]["max_images"]),
        ] + subset_args, logger, workspace, f"round_{round_idx}_mine")
        count, dist = load_failure_stats(failures)
        logger.info("round_%d failure mining complete: failures=%d failure_types=%s", round_idx, count, dist)
        return failures

    def run_optimize(round_idx: int, current_weights: Path, failures: Path) -> Path:
        round_dir = workspace / f"round_{round_idx}"
        prompts = round_dir / "prompts"
        resume_token = None
        resume_mode = str(cfg["prompt_optimization"].get("resume_mode", "overwrite"))
        if cfg["prompt_optimization"].get("resume_from_previous", False) and round_idx > 1:
            resume_token = latest_prompt_checkpoint(workspace / f"round_{round_idx - 1}" / "prompts")
        logger.info("round_%d optimize config: failures=%s weights=%s output_dir=%s prompt_scope=%s steps=%s",
                    round_idx, failures, current_weights, prompts,
                    cfg["prompt_optimization"]["prompt_scope"], cfg["prompt_optimization"]["max_train_steps"])
        cmd = [
            sys.executable, "-m", "detection_depth_adapt.optimize_object_prompts",
            "--pretrained-model-name-or-path", sd["model"],
            "--annotations", str(resolved(root, data["train_annotations"])),
            "--images", str(resolved(root, data["train_images"])),
            "--failures", str(failures),
            "--yolov7", str(resolved(root, yolo["repo"])),
            "--weights", str(current_weights),
            "--output-dir", str(prompts),
            "--prompt-scope", cfg["prompt_optimization"]["prompt_scope"],
            "--num-new-tokens", str(cfg["prompt_optimization"]["num_new_tokens"]),
            "--initializer-token", cfg["prompt_optimization"]["initializer_token"],
            "--init-mode", cfg["prompt_optimization"]["init_mode"],
            "--resume-mode", resume_mode,
            "--resolution", str(cfg["prompt_optimization"]["resolution"]),
            "--max-train-steps", str(cfg["prompt_optimization"]["max_train_steps"]),
            "--learning-rate", str(cfg["prompt_optimization"]["learning_rate"]),
            "--strength", str(cfg["prompt_optimization"]["strength"]),
            "--num-inference-steps", str(cfg["prompt_optimization"]["num_inference_steps"]),
            "--semantic-weight", str(cfg["prompt_optimization"]["semantic_weight"]),
            "--detector-weight", str(cfg["prompt_optimization"]["detector_weight"]),
            "--save-steps", str(cfg["prompt_optimization"]["save_steps"]),
            "--log-every", str(cfg["prompt_optimization"]["log_every"]),
            "--seed", str(cfg["seed"]),
        ] + subset_args
        if cfg["prompt_optimization"].get("include_class_name", False):
            cmd.append("--include-class-name")
        if resume_token is not None:
            cmd += ["--resume-token", str(resume_token)]
            logger.info("round_%d prompt resume enabled: previous=%s mode=%s", round_idx, resume_token, resume_mode)
        elif cfg["prompt_optimization"].get("resume_from_previous", False) and round_idx > 1:
            logger.info("round_%d prompt resume requested but previous learned_embeds not found; falling back to init_mode=%s",
                        round_idx, cfg["prompt_optimization"]["init_mode"])
        if sd.get("local_files_only", True):
            cmd.append("--local-files-only")
        run(cmd, logger, workspace, f"round_{round_idx}_optimize")
        logger.info("round_%d prompt optimization complete: prompts=%s", round_idx, prompts)
        return prompts

    def run_generate(round_idx: int, prompts: Path) -> tuple[Path, list[Path]]:
        round_dir = workspace / f"round_{round_idx}"
        synthetic = round_dir / "synthetic"
        logger.info("round_%d generate config: tokens_dir=%s output_dir=%s target_images=%d steps=%s",
                    round_idx, prompts, synthetic, synthetic_images_per_round, cfg["generation"]["num_inference_steps"])
        cmd = [
            sys.executable, "-m", "detection_depth_adapt.generate_hard_examples",
            "--pretrained-model-name-or-path", sd["model"],
            "--annotations", str(resolved(root, data["train_annotations"])),
            "--images", str(resolved(root, data["train_images"])),
            "--tokens-dir", str(prompts),
            "--output-dir", str(synthetic),
            "--max-images", str(synthetic_images_per_round),
            "--max-objects-per-image", str(cfg["generation"]["max_objects_per_image"]),
            "--mask-padding", str(cfg["generation"]["mask_padding"]),
            "--strength", str(cfg["generation"]["strength"]),
            "--guidance-scale", str(cfg["generation"]["guidance_scale"]),
            "--num-inference-steps", str(cfg["generation"]["num_inference_steps"]),
            "--device", cfg["generation"]["device"],
            "--seed", str(cfg["seed"] + round_idx),
            "--clean-output",
        ] + subset_args
        if cfg["generation"].get("include_class_name") is True:
            cmd.append("--include-class-name")
        elif cfg["generation"].get("include_class_name") is False:
            cmd.append("--no-include-class-name")
        if sd.get("local_files_only", True):
            cmd.append("--local-files-only")
        run(cmd, logger, workspace, f"round_{round_idx}_generate")
        selected = select_synthetic_images(synthetic, synthetic_images_per_round, int(cfg["seed"]), round_idx, logger)
        logger.info("round_%d hard-example generation complete: synthetic_dir=%s selected=%d", round_idx, synthetic, len(selected))
        return synthetic, selected

    def run_finetune(run_output: Path, run_name: str, weights: Path, synthetic_dir: Path, epochs: int) -> Path:
        logger.info("finetune config: output_dir=%s run_name=%s weights=%s synthetic_dir=%s epochs=%d batch_size=%s",
                    run_output, run_name, weights, synthetic_dir, epochs, cfg["finetune"]["batch_size"])
        run([
            sys.executable, "-m", "detection_depth_adapt.finetune_yolov7",
            "--train-annotations", str(resolved(root, data["train_annotations"])),
            "--train-images", str(resolved(root, data["train_images"])),
            "--val-annotations", str(resolved(root, data["val_annotations"])),
            "--val-images", str(resolved(root, data["val_images"])),
            "--synthetic-dir", str(synthetic_dir),
            "--real-train-list", str(shared_datasets["real_train"]),
            "--real-val-list", str(shared_datasets["real_val"]),
            "--yolov7", str(resolved(root, yolo["repo"])),
            "--weights", str(weights),
            "--output-dir", str(run_output),
            "--device", str(yolo["device"]),
            "--batch-size", str(cfg["finetune"]["batch_size"]),
            "--image-size", str(yolo["image_size"]),
            "--epochs", str(epochs),
            "--workers", str(cfg["finetune"]["workers"]),
            "--run-name", run_name,
        ] + train_eval_subset_args, logger, workspace, run_name)
        ckpt = best_or_last(run_output, run_name)
        if not ckpt.exists():
            raise FileNotFoundError(f"YOLOv7 checkpoint not found after finetune: {ckpt}")
        logger.info("finetune complete: run_name=%s checkpoint=%s", run_name, ckpt)
        return ckpt

    def run_evaluate(round_idx: int, weights: Path) -> tuple[Path, dict]:
        eval_dir = workspace / f"round_{round_idx}" / "evaluation"
        run_name = f"eval_round_{round_idx}"
        logger.info("round_%d evaluate config: weights=%s batch_size=%s", round_idx, weights, cfg["evaluation"]["batch_size"])
        run([
            sys.executable, "-m", "detection_depth_adapt.evaluate_yolov7",
            "--val-annotations", str(resolved(root, data["val_annotations"])),
            "--val-images", str(resolved(root, data["val_images"])),
            "--val-list", str(shared_datasets["real_val"]),
            "--coco-annotations", str(shared_datasets["val_coco"]),
            "--yolov7", str(resolved(root, yolo["repo"])),
            "--weights", str(weights),
            "--output-dir", str(eval_dir),
            "--device", str(yolo["device"]),
            "--batch-size", str(cfg["evaluation"]["batch_size"]),
            "--image-size", str(yolo["image_size"]),
            "--run-name", run_name,
        ] + train_eval_subset_args, logger, workspace, f"round_{round_idx}_evaluate")
        metrics = load_metrics(eval_dir / "metrics.json")
        (workspace / f"round_{round_idx}" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("round_%d evaluation complete: metrics=%s mAP=%s", round_idx, metrics, extract_map(metrics))
        return eval_dir / "metrics.json", metrics

    def run_warmup() -> Path:
        warmup_dir = workspace / "warmup"
        synthetic_empty = warmup_dir / "empty_synthetic"
        (synthetic_empty / "images").mkdir(parents=True, exist_ok=True)
        (synthetic_empty / "labels").mkdir(parents=True, exist_ok=True)
        logger.info("warmup start: base_weights=%s epochs=%d output=%s", base_weights, warmup_epochs, warmup_dir)
        ckpt = run_finetune(warmup_dir, "warmup", base_weights, synthetic_empty, warmup_epochs)
        target = warmup_dir / "best.pt"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(ckpt.resolve())
        logger.info("warmup complete: checkpoint=%s alias=%s", ckpt, target)
        write_round_summary(warmup_dir / "summary.json", {
            "stage": "warmup",
            "input_weights": str(base_weights),
            "epochs": warmup_epochs,
            "best_checkpoint": str(ckpt),
            "alias": str(target),
        })
        return ckpt

    def run_round(round_idx: int, current_weights: Path, selected_by_round: list[list[Path]]) -> Path:
        round_dir = workspace / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        logger.info("closed-loop round start: round=%d current_weights=%s", round_idx, current_weights)

        failures = run_mine(round_idx, current_weights)
        failure_count, failure_dist = load_failure_stats(failures)
        prompts = run_optimize(round_idx, current_weights, failures)
        synthetic, selected = run_generate(round_idx, prompts)
        selected_by_round.append(selected)
        synthetic_pool = build_synthetic_pool(round_dir, selected_by_round, logger)
        checkpoint_dir = round_dir / "checkpoint"
        checkpoint = run_finetune(checkpoint_dir, f"finetune_round_{round_idx}", current_weights, synthetic_pool, finetune_epochs)
        alias = checkpoint_dir / "best.pt"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(checkpoint.resolve())
        metrics_file, metrics = run_evaluate(round_idx, checkpoint)

        summary = {
            "round": round_idx,
            "current_weights": str(current_weights),
            "failure_count": failure_count,
            "failure_type_distribution": failure_dist,
            "synthetic_dir": str(synthetic),
            "synthetic_selected_this_round": len(selected),
            "synthetic_pool_size": sum(len(x) for x in selected_by_round),
            "finetune_epochs": finetune_epochs,
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_alias": str(alias),
            "metrics_file": str(metrics_file),
            "evaluation_mAP": extract_map(metrics),
            "metrics": metrics,
        }
        write_round_summary(round_dir / "summary.json", summary)
        logger.info(
            "closed-loop round complete: round=%d failures=%d failure_types=%s synthetic_this_round=%d synthetic_pool=%d checkpoint=%s mAP=%s",
            round_idx, failure_count, failure_dist, len(selected), summary["synthetic_pool_size"], checkpoint, summary["evaluation_mAP"],
        )
        return checkpoint

    def run_closed_loop() -> None:
        overall_start = time.time()
        if skip_warmup:
            warmup_path = resolved(root, warmup_weights) if warmup_weights else workspace / "warmup" / "best.pt"
            current_weights = existing_checkpoint(warmup_path, "reused warmup")
            logger.info("warmup skipped: reusing checkpoint=%s", current_weights)
            warmup_dir = workspace / "warmup"
            warmup_dir.mkdir(parents=True, exist_ok=True)
            alias = warmup_dir / "best.pt"
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            alias.symlink_to(current_weights)
            write_round_summary(warmup_dir / "summary.json", {
                "stage": "warmup",
                "skipped": True,
                "reused_checkpoint": str(current_weights),
                "alias": str(alias),
            })
        else:
            current_weights = run_warmup()
        selected_by_round: list[list[Path]] = []
        summaries = []
        for round_idx in range(1, rounds + 1):
            start = time.time()
            current_weights = run_round(round_idx, current_weights, selected_by_round)
            summary_path = workspace / f"round_{round_idx}" / "summary.json"
            summaries.append(load_metrics(summary_path))
            logger.info("round elapsed: round=%d elapsed_sec=%.1f next_current_weights=%s",
                        round_idx, time.time() - start, current_weights)
        write_round_summary(workspace / "experiment_summary.json", {
            "rounds": rounds,
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "synthetic_images_per_round": synthetic_images_per_round,
            "final_weights": str(current_weights),
            "mining_generation_class_subset": subset_label(class_subset),
            "train_eval_label_scope": train_eval_label_scope,
            "train_eval_class_subset": subset_label(train_eval_class_subset),
            "elapsed_sec": time.time() - overall_start,
            "round_summaries": summaries,
        })
        logger.info("closed-loop experiment complete: final_weights=%s summary=%s elapsed_sec=%.1f",
                    current_weights, workspace / "experiment_summary.json", time.time() - overall_start)

    if args.command == "all":
        run_closed_loop()
    elif args.command == "warmup":
        run_warmup()
    else:
        round_idx = int(args.round)
        current = input_weights_for_round(workspace, base_weights, round_idx)
        eval_weights = checkpoint_for_round(workspace, round_idx) or current

        start = time.time()
        logger.info("stage start: command=%s round=%d current_weights=%s eval_weights=%s",
                    args.command, round_idx, current, eval_weights)
        if args.command == "mine":
            run_mine(round_idx, current)
        elif args.command == "optimize":
            run_optimize(round_idx, current, workspace / f"round_{round_idx}" / "failures.json")
        elif args.command == "generate":
            run_generate(round_idx, workspace / f"round_{round_idx}" / "prompts")
        elif args.command == "finetune":
            synthetic = workspace / f"round_{round_idx}" / "synthetic"
            run_finetune(workspace / f"round_{round_idx}" / "checkpoint", f"finetune_round_{round_idx}", current, synthetic, finetune_epochs)
        elif args.command == "evaluate":
            run_evaluate(round_idx, eval_weights)
        elif args.command == "round":
            selected_by_round = []
            for old_round in range(1, round_idx):
                old_dir = workspace / f"round_{old_round}" / "synthetic"
                selected_by_round.append(select_synthetic_images(old_dir, synthetic_images_per_round, int(cfg["seed"]), old_round, logger))
            run_round(round_idx, current, selected_by_round)
        logger.info("stage complete: command=%s round=%d elapsed_sec=%.1f", args.command, round_idx, time.time() - start)

    logger.info("runner complete: command=%s", args.command)


if __name__ == "__main__":
    main()
