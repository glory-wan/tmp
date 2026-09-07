# 生成 steps / detector_weight 消融配置和命令
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def optimize_cmd(config: Path, round_idx: int = 1) -> str:
    return (
        "conda run -n py10 python -m detection_gligen_sdedit.runner optimize "
        f"--config {config} --round {round_idx}"
    )


def generate_cmd(config: Path, round_idx: int = 1) -> str:
    return (
        "conda run -n py10 python -m detection_gligen_sdedit.runner generate "
        f"--config {config} --round {round_idx}"
    )


def synthetic_vis_cmd(synthetic_dir: Path, output_dir: Path) -> str:
    return (
        "conda run -n py10 python -m detection_gligen_sdedit.experiments.build_synthetic_pool_html "
        f"--synthetic-pool {synthetic_dir} "
        f"--output-dir {output_dir}"
    )


def finetune_eval_cmds(config: Path, round_idx: int = 1) -> list[str]:
    return [
        f"conda run -n py10 python -m detection_gligen_sdedit.runner finetune --config {config} --round {round_idx}",
        f"conda run -n py10 python -m detection_gligen_sdedit.runner evaluate --config {config} --round {round_idx}",
    ]


def prompt_sample_cmd(tokens_dir: Path, checkpoint: str, output_dir: Path, model: str, steps: int, seed: int) -> str:
    return (
        "conda run -n py10 python -m detection_gligen_sdedit.experiments.sample_prompts "
        f"--pretrained-model-name-or-path {model} "
        "--annotations data/coco/annotations/instances_minitrain2017.json "
        "--images data/coco/images/train2017 "
        f"--tokens-dir {tokens_dir} "
        f"--checkpoint {checkpoint} "
        f"--output-dir {output_dir} "
        "--num-images 4 "
        f"--num-inference-steps {steps} "
        f"--seed {seed} "
        "--device cuda:0 "
        "--local-files-only"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create prompt optimization ablation configs and command lists.")
    parser.add_argument("--base-config", default="detection_gligen_sdedit/configs/coco_mini_exp_0723.yaml")
    parser.add_argument("--output-root", default="outputs/detection_gligen_sdedit/prompt_ablation_0723")
    parser.add_argument("--steps", default="100,300,500,1000,2000")
    parser.add_argument("--detector-weights", default="0.0,0.05,0.1,0.2")
    parser.add_argument("--class-semantic-weights", default="0.0,0.1,0.2,0.4")
    parser.add_argument("--fixed-detector-weight", type=float, default=0.1)
    parser.add_argument("--fixed-class-semantic-weight", type=float, default=None)
    parser.add_argument("--fixed-steps", type=int, default=500)
    parser.add_argument("--reuse-round-dir", default=None,
                        help="Existing round dir whose failures.json is reused for quick optimize/generate ablations.")
    parser.add_argument("--sample-model", default="/data/model/models_zwq/stable-diffusion-v1-5",
                        help="Plain StableDiffusionPipeline model for sample_prompts qualitative grids.")
    parser.add_argument("--sample-checkpoints", default=None,
                        help="Comma-separated checkpoints for prompt sampling. Defaults to each config's final step.")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    output_root = Path(args.output_root)
    config_dir = output_root / "configs"
    commands = {
        "quick_steps": [],
        "quick_detector_weight": [],
        "quick_class_semantic_weight": [],
        "formal": [],
        "prompt_sampling_existing": [],
    }
    steps_values = [int(x) for x in args.steps.split(",") if x.strip()]
    weight_values = [float(x) for x in args.detector_weights.split(",") if x.strip()]
    class_semantic_values = [float(x) for x in args.class_semantic_weights.split(",") if x.strip()]
    fixed_class_semantic = (
        args.fixed_class_semantic_weight
        if args.fixed_class_semantic_weight is not None
        else float(base.get("prompt_optimization", {}).get("class_semantic_weight", 0.0))
    )
    sample_checkpoints = [x.strip() for x in args.sample_checkpoints.split(",")] if args.sample_checkpoints else None
    reuse_round = Path(args.reuse_round_dir) if args.reuse_round_dir else Path(base["workspace"]) / "round_1"
    reuse_failures = reuse_round / "failures.json"
    if not reuse_failures.exists():
        raise FileNotFoundError(f"reuse failures not found: {reuse_failures}")

    def prepare_quick_config(name: str, *, steps: int, detector_weight: float, class_semantic_weight: float) -> Path:
        cfg = copy.deepcopy(base)
        workspace = output_root / name
        cfg["workspace"] = str(workspace)
        cfg["closed_loop"]["rounds"] = 1
        cfg["prompt_optimization"]["max_train_steps"] = steps
        cfg["prompt_optimization"]["save_steps"] = min(100, steps)
        cfg["prompt_optimization"]["detector_weight"] = detector_weight
        cfg["prompt_optimization"]["class_semantic_weight"] = class_semantic_weight
        cfg["generation"]["max_images"] = 64
        cfg["closed_loop"]["synthetic_images_per_round"] = 64
        cfg_path = config_dir / f"{name}.yaml"
        write_yaml(cfg_path, cfg)
        link_file(reuse_failures, workspace / "round_1" / "failures.json")
        return cfg_path

    def add_quick_commands(section: str, name: str, cfg_path: Path, steps: int) -> None:
        workspace = output_root / name
        tokens_dir = workspace / "round_1" / "prompts"
        synthetic_dir = workspace / "round_1" / "synthetic"
        commands[section].append(optimize_cmd(cfg_path))
        commands[section].append(generate_cmd(cfg_path))
        checkpoints = sample_checkpoints or [f"learned_embeds-{steps}.bin"]
        for ckpt in checkpoints:
            out = output_root / "qual" / name / ckpt.replace(".bin", "")
            commands[section].append(prompt_sample_cmd(tokens_dir, ckpt, out, args.sample_model, 30, args.seed))
        commands[section].append(synthetic_vis_cmd(synthetic_dir, output_root / "gen_vis" / name))

    # Quick ablation A: fixed detector_weight, different max_train_steps.
    for steps in steps_values:
        name = f"quick_steps_{steps}"
        cfg_path = prepare_quick_config(
            name,
            steps=steps,
            detector_weight=args.fixed_detector_weight,
            class_semantic_weight=fixed_class_semantic,
        )
        add_quick_commands("quick_steps", name, cfg_path, steps)

    # Quick ablation B: fixed steps, different detector_weight.
    for weight in weight_values:
        weight_tag = str(weight).replace(".", "p")
        name = f"quick_dw_{weight_tag}_steps_{args.fixed_steps}"
        cfg_path = prepare_quick_config(
            name,
            steps=args.fixed_steps,
            detector_weight=weight,
            class_semantic_weight=fixed_class_semantic,
        )
        add_quick_commands("quick_detector_weight", name, cfg_path, args.fixed_steps)

    # Quick ablation C: fixed steps/detector_weight, different class_semantic_weight.
    for class_semantic_weight in class_semantic_values:
        csw_tag = str(class_semantic_weight).replace(".", "p")
        name = f"quick_csw_{csw_tag}_steps_{args.fixed_steps}_dw_{str(args.fixed_detector_weight).replace('.', 'p')}"
        cfg_path = prepare_quick_config(
            name,
            steps=args.fixed_steps,
            detector_weight=args.fixed_detector_weight,
            class_semantic_weight=class_semantic_weight,
        )
        add_quick_commands("quick_class_semantic_weight", name, cfg_path, args.fixed_steps)

    commands["formal"].append("# intentionally empty for quick prompt/generation ablation; select winners and run finetune/evaluate separately")

    # Existing checkpoint standalone prompt sampling: useful for current exp_0723.
    model = args.sample_model
    for round_idx in (1, 2):
        tokens_dir = Path(base["workspace"]) / f"round_{round_idx}" / "prompts"
        if not tokens_dir.exists():
            continue
        for ckpt in ("learned_embeds-100.bin", "learned_embeds-300.bin", "learned_embeds-500.bin", "learned_embeds-1000.bin"):
            if not (tokens_dir / ckpt).exists():
                continue
            out = output_root / "qual_existing_exp_0723" / f"round_{round_idx}" / ckpt.replace(".bin", "")
            commands["prompt_sampling_existing"].append(prompt_sample_cmd(tokens_dir, ckpt, out, model, 30, args.seed + round_idx))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    command_lines = []
    for section, cmds in commands.items():
        command_lines.append(f"# {section}")
        command_lines.extend(cmds)
        command_lines.append("")
    (output_root / "commands.sh").write_text("\n".join(command_lines), encoding="utf-8")
    print(f"wrote configs: {config_dir}")
    print(f"wrote commands: {output_root / 'commands.sh'}")


if __name__ == "__main__":
    main()
