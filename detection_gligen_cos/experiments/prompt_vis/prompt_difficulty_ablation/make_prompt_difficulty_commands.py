from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible commands for the prompt difficulty ablation.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--yolov7", default="external/yolov7")
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--max-objects-per-image", type=int, default=6)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--gligen-scheduled-sampling-beta", type=float, default=0.5)
    parser.add_argument("--global-prompt-source", choices=["category_count", "template", "empty"], default="empty")
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transfer-source-class-ids", default=None)
    parser.add_argument("--include-finetune", action="store_true")
    parser.add_argument("--train-annotations", default=None)
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--val-annotations", default=None)
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--real-train-list", default=None)
    parser.add_argument("--real-val-list", default=None)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--baseline-metrics", default=None)
    return parser.parse_args()


def command(parts: list[object]) -> str:
    return shlex.join([str(part) for part in parts])


def add_optional(parts: list[object], flag: str, value) -> None:
    if value is not None:
        parts.extend([flag, value])


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_conditions = ["class", "learned", "class_learned"]
    transfer_ids = [int(value) for value in args.transfer_source_class_ids.split(",")] if args.transfer_source_class_ids else []
    transfer_conditions = [f"learned_transfer_from_{class_id}" for class_id in transfer_ids]
    all_generated = base_conditions + transfer_conditions
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "# Run from the repository root.", ""]

    def generation_parts(condition: str, phrase_mode: str, transfer_id: int | None = None) -> list[object]:
        parts: list[object] = [
            "conda", "run", "-n", "py10", "python", "-m",
            "detection_gligen_sdedit.experiments.prompt_vis.prompt_difficulty_ablation.generate_prompt_difficulty_condition",
            "--phrase-mode", phrase_mode,
            "--pretrained-model-name-or-path", args.pretrained_model_name_or_path,
            "--annotations", args.annotations,
            "--images", args.images,
            "--tokens-dir", args.tokens_dir,
            "--output-dir", root / "generated" / condition,
            "--max-images", args.max_images,
            "--max-objects-per-image", args.max_objects_per_image,
            "--width", 512, "--height", 512,
            "--strength", args.strength,
            "--guidance-scale", args.guidance_scale,
            "--num-inference-steps", args.num_inference_steps,
            "--gligen-scheduled-sampling-beta", args.gligen_scheduled_sampling_beta,
            "--global-prompt-source", args.global_prompt_source,
            "--enable-generation-filter",
            "--min-bbox-area-ratio", args.min_bbox_area_ratio,
            "--seed", args.seed,
            "--device", args.device,
            "--local-files-only", "--clean-output", "--save-visualization",
        ]
        add_optional(parts, "--classes-subset", args.classes_subset)
        add_optional(parts, "--class-ids", args.class_ids)
        add_optional(parts, "--transfer-source-class-id", transfer_id)
        return parts

    lines.append("# 1. Paired generation: only the per-box phrase changes.")
    for condition in base_conditions:
        lines.append(command(generation_parts(condition, condition)))
    for transfer_id, condition in zip(transfer_ids, transfer_conditions):
        lines.append(command(generation_parts(condition, "learned", transfer_id)))
    lines.extend(["", "# 2. Original-image reference with exactly the same layout objects."])
    lines.append(command([
        "conda", "run", "-n", "py10", "python", "-m",
        "detection_gligen_sdedit.experiments.prompt_vis.prompt_difficulty_ablation.prepare_prompt_difficulty_source_reference",
        "--condition-dir", root / "generated" / "class",
        "--output-dir", root / "generated" / "source_original",
        "--clean-output",
    ]))

    lines.extend(["", "# 3. Initial detector confidence, errors and backbone ROI features."])
    for condition in ["source_original", *all_generated]:
        lines.append(command([
            "conda", "run", "-n", "py10", "python", "-m",
            "detection_gligen_sdedit.experiments.prompt_vis.prompt_difficulty_ablation.evaluate_prompt_difficulty_detector",
            "--synthetic-dir", root / "generated" / condition,
            "--yolov7", args.yolov7,
            "--weights", args.detector_weights,
            "--output-dir", root / "analysis" / condition,
            "--device", args.device,
            "--image-size", 640,
            "--feature-layer", 50,
        ]))

    if args.include_finetune:
        required = ["train_annotations", "train_images", "val_annotations", "val_images"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"--include-finetune requires: {', '.join('--' + x.replace('_', '-') for x in missing)}")
        lines.extend(["", "# 4. Independent finetuning from the same initial detector weights, then full validation."])
        for condition in all_generated:
            train_output = root / "finetune" / condition
            run_name = f"prompt_difficulty_{condition}"
            train_parts: list[object] = [
                "conda", "run", "-n", "py10", "python", "-m", "detection_gligen_sdedit.finetune_yolov7",
                "--train-annotations", args.train_annotations,
                "--train-images", args.train_images,
                "--val-annotations", args.val_annotations,
                "--val-images", args.val_images,
                "--synthetic-dir", root / "generated" / condition,
                "--yolov7", args.yolov7,
                "--weights", args.detector_weights,
                "--output-dir", train_output,
                "--device", args.device,
                "--batch-size", args.batch_size,
                "--image-size", 640,
                "--epochs", args.finetune_epochs,
                "--workers", args.workers,
                "--run-name", run_name,
            ]
            add_optional(train_parts, "--real-train-list", args.real_train_list)
            add_optional(train_parts, "--real-val-list", args.real_val_list)
            lines.append(command(train_parts))
            eval_parts: list[object] = [
                "conda", "run", "-n", "py10", "python", "-m", "detection_gligen_sdedit.evaluate_yolov7",
                "--val-annotations", args.val_annotations,
                "--val-images", args.val_images,
                "--yolov7", args.yolov7,
                "--weights", train_output / "runs" / run_name / "weights" / "best.pt",
                "--output-dir", root / "posttrain" / condition,
                "--device", args.device,
                "--batch-size", args.batch_size,
                "--image-size", 640,
                "--run-name", f"eval_{condition}",
            ]
            if args.real_val_list is not None:
                eval_parts.extend(["--val-list", args.real_val_list, "--coco-annotations", args.val_annotations])
            lines.append(command(eval_parts))

    report_conditions = ["source_original", *all_generated]
    lines.extend(["", "# 5. Detector metrics, t-SNE and paired visual report."])
    report_parts: list[object] = [
        "conda", "run", "-n", "py10", "python", "-m",
        "detection_gligen_sdedit.experiments.prompt_vis.prompt_difficulty_ablation.build_prompt_difficulty_report",
        "--experiment-dir", root,
        "--conditions", ",".join(report_conditions),
    ]
    add_optional(report_parts, "--baseline-metrics", args.baseline_metrics)
    lines.append(command(report_parts))
    script = root / "commands.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    config = vars(args) | {"conditions": report_conditions}
    (root / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(script)


if __name__ == "__main__":
    main()
