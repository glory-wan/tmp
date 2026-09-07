"""GLIGEN-SDEdit closed loop with Prompt gradient alignment and fresh retraining.

This runner is intentionally independent from :mod:`detection_gligen_sdedit.runner`.
It uses Ultralytics for failure feedback, aligns Prompt optimization gradients
with a validation Guide set, generates full-image GLIGEN-SDEdit examples, and
trains one fresh detector on original train2017 plus all synthetic examples.

The state JSON embeds the complete effective configuration. A stopped run can
therefore be resumed with only ``--state-json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERALL_YAML = (
    Path(__file__).resolve().parent / "configs" / "coco_bus_bird_umbrella_retrain.yaml"
)
DEFAULT_MODEL_YAML = Path(
    "/home/suhu/data/wgr/code/prompt/ultralytics-main/ultralytics/cfg/models/26/yolo26n.yaml"
)
STATE_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolved(value: str | Path, root: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = utc_now()
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"State JSON must contain an object: {path}")
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported state schema {payload.get('schema_version')!r}; expected {STATE_SCHEMA_VERSION}: {path}"
        )
    return payload


def write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def dataset_root_from_yaml(dataset_yaml: Path) -> tuple[Path, dict[str, Any]]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {dataset_yaml}")
    root_value = payload.get("path")
    root = (
        resolved(root_value, dataset_yaml.parent)
        if root_value
        else dataset_yaml.parent.resolve()
    )
    return root, payload


def normalize_effective_config(
    raw: dict[str, Any], args: argparse.Namespace, overall_yaml: Path
) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    baseline = (
        args.baseline or cfg.get("baseline") or cfg.get("model", {}).get("weights")
    )
    dataset = args.dataset or cfg.get("dataset") or cfg.get("model", {}).get("data")
    if not baseline:
        raise ValueError(
            "A baseline checkpoint is required via --baseline or overall_yaml baseline"
        )
    if not dataset:
        raise ValueError(
            "A dataset YAML is required via --dataset or overall_yaml dataset"
        )

    cfg["baseline"] = str(resolved(baseline))
    cfg["dataset"] = str(resolved(dataset))
    cfg["model_yaml"] = str(
        resolved(args.model_yaml or cfg.get("model_yaml") or DEFAULT_MODEL_YAML)
    )
    cfg["rounds"] = int(
        args.rounds
        if args.rounds is not None
        else cfg.get("rounds", cfg.get("round", 10))
    )
    cfg["syn_sample"] = int(
        args.syn_sample
        if args.syn_sample is not None
        else cfg.get(
            "syn_sample",
            cfg.get(
                "syn_smaple",
                cfg.get("closed_loop", {}).get("synthetic_images_per_round", 2000),
            ),
        )
    )
    cfg["device"] = str(
        args.device if args.device is not None else cfg.get("device", "6")
    )
    cfg["workspace"] = str(
        resolved(cfg.get("workspace", "outputs/gligen_sdedit_retrain_closed_loop"))
    )
    cfg["overall_yaml"] = str(overall_yaml.resolve())

    if cfg["rounds"] <= 0:
        raise ValueError("rounds must be positive")
    if cfg["syn_sample"] <= 0:
        raise ValueError("syn_sample must be positive")
    if cfg["device"].lower() != "cpu" and not cfg["device"].isdigit():
        raise ValueError(
            "device must be one physical GPU index (for example 6), or cpu"
        )

    closed_loop = cfg.setdefault("closed_loop", {})
    closed_loop["rounds"] = cfg["rounds"]
    closed_loop["synthetic_images_per_round"] = cfg["syn_sample"]
    closed_loop["warmup_weights"] = cfg["baseline"]

    model_cfg = cfg.setdefault("model", {})
    model_cfg.setdefault("backend", "ultralytics")
    model_cfg.setdefault("family", "yolo")
    model_cfg.setdefault("task", "detect")
    model_cfg["weights"] = cfg["baseline"]
    model_cfg["data"] = cfg["dataset"]
    model_cfg["ultralytics_root"] = str(
        resolved(model_cfg.get("ultralytics_root", "ultralytics"))
    )
    model_cfg["device"] = "cpu" if cfg["device"].lower() == "cpu" else "0"
    model_cfg.setdefault("image_size", 640)
    model_cfg.setdefault("inference", {"iou": 0.7, "max_det": 300})

    dataset_root, _ = dataset_root_from_yaml(Path(cfg["dataset"]))
    data_cfg = cfg.setdefault("data", {})
    data_cfg.update(
        {
            "train_annotations": str(
                dataset_root / "annotation/instances_train2017.json"
            ),
            "train_images": str(dataset_root / "images/train2017"),
            "val_annotations": str(dataset_root / "annotation/instances_val2017.json"),
            "val_images": str(dataset_root / "images/val2017"),
        }
    )

    generation = cfg.setdefault("generation", {})
    generation["device"] = "cpu" if cfg["device"].lower() == "cpu" else "cuda:0"
    generation["max_images"] = cfg["syn_sample"]
    generation.setdefault("local_files_only", cfg.get("stable_diffusion", {}).get("local_files_only", True))
    generation.setdefault("variant", None)
    generation.setdefault("include_class_name", None)
    generation.setdefault("max_objects_per_image", 20)
    generation.setdefault("width", 512)
    generation.setdefault("height", 512)
    generation.setdefault("strength", 0.65)
    generation.setdefault("noise_timestep", None)
    generation.setdefault("guidance_scale", 7.5)
    generation.setdefault("num_inference_steps", 30)
    generation.setdefault("gligen_scheduled_sampling_beta", 0.3)
    generation.setdefault("global_prompt_source", "template")
    generation.setdefault(
        "global_prompt_template",
        "{phrases}, photo, highly detailed, photorealistic",
    )
    generation.setdefault(
        "negative_prompt",
        "wrong class, duplicate object, deformed, cropped, text, watermark",
    )
    generation.setdefault("enable_generation_filter", False)
    generation.setdefault("min_bbox_area_ratio", 0.0)
    generation.setdefault("save_visualization", False)

    optimization = cfg.setdefault("prompt_optimization", {})
    gradient = optimization.setdefault("gradient_alignment", {})
    gradient.setdefault("enabled", True)
    gradient.setdefault("guide_root", None)
    gradient.setdefault("guide_split", "val")
    gradient.setdefault("task", "detect")
    gradient.setdefault("parameter_scope", "detection_head")
    gradient.setdefault("image_size", int(model_cfg["image_size"]))
    gradient.setdefault("scaleup", False)
    gradient.setdefault("guide_batch_size", 8)
    gradient.setdefault("workers", 4)
    gradient.setdefault("max_guide_images", None)
    gradient.setdefault("target_scope", "full_layout")
    gradient.setdefault("loss", "hinge")
    gradient.setdefault("weight", 1.0)
    gradient.setdefault("margin", 0.0)
    gradient.setdefault(
        "warmup_steps", max(1, int(optimization.get("max_train_steps", 200)) // 10)
    )
    gradient.setdefault("epsilon", 1.0e-12)
    gradient.setdefault("implementation", "exact")
    gradient.setdefault("cache", True)
    gradient.setdefault("max_objects_per_image", generation["max_objects_per_image"])
    gradient.setdefault("enable_layout_filter", generation["enable_generation_filter"])
    gradient.setdefault("min_bbox_area_ratio", generation["min_bbox_area_ratio"])
    # V1 stored post-generation splitting options at the top level. They have no
    # meaning in schema 2 and must not leak into the effective configuration.
    cfg.pop("gradient_alignment", None)

    retrain = cfg.setdefault("retrain", {})
    retrain.setdefault("epochs", 500)
    retrain.setdefault("image_size", 640)
    retrain.setdefault("batch_size", 156)
    retrain.setdefault("project", "yolo26n_COCO_subset")
    retrain.setdefault("patience", 10)
    retrain.setdefault("workers", 24)
    retrain.setdefault("exist_ok", True)
    retrain.setdefault("run_name_template", "round_{round}_cos_prompt")

    return cfg


def validate_effective_config(cfg: dict[str, Any]) -> None:
    baseline = Path(cfg["baseline"])
    dataset_yaml = Path(cfg["dataset"])
    ultralytics_root = Path(cfg["model"]["ultralytics_root"])
    require_file(baseline, "Baseline checkpoint")
    if baseline.suffix.lower() != ".pt":
        raise ValueError(f"Baseline must be a native .pt checkpoint: {baseline}")
    require_file(dataset_yaml, "Dataset YAML")
    require_file(
        ultralytics_root / "ultralytics/__init__.py", "Local Ultralytics package"
    )
    generation_model = cfg.get("generation", {}).get("model") or cfg.get("generation", {}).get("gligen_model")
    if not generation_model:
        raise ValueError("generation.model is required for GLIGEN-SDEdit generation")
    if cfg["generation"].get("local_files_only", True):
        require_directory(resolved(generation_model), "Local GLIGEN model")

    dataset_root, _ = dataset_root_from_yaml(dataset_yaml)
    for path, label in (
        (
            dataset_root / "annotation/instances_train2017.json",
            "train COCO annotations",
        ),
        (dataset_root / "annotation/instances_val2017.json", "val COCO annotations"),
        (dataset_root / "images/train2017", "train images"),
        (dataset_root / "images/val2017", "val images"),
        (dataset_root / "labels/train2017", "train labels"),
        (dataset_root / "labels/val2017", "val labels"),
    ):
        (require_file if path.suffix == ".json" else require_directory)(path, label)

    gradient = cfg["prompt_optimization"]["gradient_alignment"]
    if float(gradient["weight"]) < 0:
        raise ValueError("prompt_optimization.gradient_alignment.weight must be non-negative")
    if not -1.0 <= float(gradient["margin"]) <= 1.0:
        raise ValueError("prompt_optimization.gradient_alignment.margin must be in [-1,1]")
    if int(gradient["warmup_steps"]) < 0:
        raise ValueError("prompt_optimization.gradient_alignment.warmup_steps must be non-negative")
    if int(gradient["image_size"]) <= 0 or int(gradient["guide_batch_size"]) <= 0:
        raise ValueError("alignment image_size and guide_batch_size must be positive")
    if int(gradient["workers"]) < 0:
        raise ValueError("prompt_optimization.gradient_alignment.workers must be non-negative")
    if float(gradient["epsilon"]) <= 0:
        raise ValueError("prompt_optimization.gradient_alignment.epsilon must be positive")
    if gradient["task"] != "detect":
        raise ValueError("Prompt gradient alignment currently requires task=detect")
    if gradient["parameter_scope"] != "detection_head":
        raise ValueError("Prompt gradient alignment currently requires parameter_scope=detection_head")
    if gradient["target_scope"] != "full_layout":
        raise ValueError("Prompt gradient alignment currently requires target_scope=full_layout")
    if gradient["loss"] != "hinge":
        raise ValueError("Prompt gradient alignment currently requires loss=hinge")
    if gradient["implementation"] != "exact":
        raise ValueError("Prompt gradient alignment currently requires implementation=exact")
    if gradient.get("enabled", True):
        if cfg["model"]["family"] != "yolo" or cfg["model"]["task"] != "detect":
            raise ValueError("Exact Prompt gradient alignment currently supports YOLO detect only")
        guide_root = (
            resolved(gradient["guide_root"], dataset_yaml.parent)
            if gradient.get("guide_root")
            else dataset_root
        )
        if gradient["guide_split"] != "val":
            raise ValueError("Prompt gradient alignment currently requires guide_split=val")
        guide_images = guide_root / "images/val2017"
        guide_labels = guide_root / "labels/val2017"
        require_directory(guide_images, "Guide images")
        require_directory(guide_labels, "Guide labels")
        if not any(path.is_file() for path in guide_images.iterdir()):
            raise RuntimeError(f"Guide image directory is empty: {guide_images}")

    model_yaml = Path(cfg["model_yaml"])
    if not model_yaml.is_file():
        unified = Path(
            re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(model_yaml.with_suffix("")))
        ).with_suffix(model_yaml.suffix)
        if not unified.is_file():
            raise FileNotFoundError(
                f"Model YAML not found, and its Ultralytics unified-scale fallback is absent: {model_yaml}"
            )


def project_root(cfg: dict[str, Any]) -> Path:
    project = Path(str(cfg["retrain"]["project"])).expanduser()
    if project.is_absolute():
        return project.resolve()
    return (
        Path(cfg["model"]["ultralytics_root"]) / "runs" / "detect" / project
    ).resolve()


def retrain_run_name(cfg: dict[str, Any], round_idx: int) -> str:
    name = str(cfg["retrain"].get("run_name_template", "round_{round}_cos_prompt")).format(
        round=round_idx
    )
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid retrain run name: {name!r}")
    return name


def initial_state(cfg: dict[str, Any], state_path: Path) -> dict[str, Any]:
    workspace = Path(cfg["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    existing_rounds = sorted(workspace.glob("round_*"))
    if existing_rounds:
        raise FileExistsError(
            f"New run workspace already contains round directories; resume with its state JSON or choose another workspace: "
            f"{existing_rounds[:3]}"
        )
    run_root = project_root(cfg)
    collisions = [
        run_root / retrain_run_name(cfg, round_idx)
        for round_idx in range(1, int(cfg["rounds"]) + 1)
        if (run_root / retrain_run_name(cfg, round_idx)).exists()
    ]
    if collisions:
        raise FileExistsError(
            "Fresh retrain output names already exist. Use the matching state JSON to resume or change retrain.project: "
            + ", ".join(str(path) for path in collisions[:5])
        )
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "state_json": str(state_path.resolve()),
        "config": cfg,
        "current": {"round": 1, "stage": "mine", "status": "pending"},
        "rounds": {},
        "status": "running",
    }


def round_state(state: dict[str, Any], round_idx: int) -> dict[str, Any]:
    return state.setdefault("rounds", {}).setdefault(
        str(round_idx),
        {
            "status": "running",
            "input_model": None,
            "mine": {"status": "pending"},
            "optimize": {"status": "pending"},
            "generate": {"status": "pending"},
            "retrain": {"status": "pending", "epoch": 0},
        },
    )


def mark_stage_running(
    state: dict[str, Any], state_path: Path, round_idx: int, stage: str, **values: Any
) -> dict[str, Any]:
    record = round_state(state, round_idx)[stage]
    record.update(values)
    record["status"] = "running"
    record["started_at"] = record.get("started_at", utc_now())
    record["attempts"] = int(record.get("attempts", 0)) + 1
    state["current"] = {
        "round": round_idx,
        "stage": stage,
        "status": "running",
        **values,
    }
    write_json_atomic(state_path, state)
    return record


def mark_stage_complete(
    state: dict[str, Any], state_path: Path, round_idx: int, stage: str, **values: Any
) -> None:
    record = round_state(state, round_idx)[stage]
    record.update(values)
    record["status"] = "completed"
    record["completed_at"] = utc_now()
    state["current"] = {
        "round": round_idx,
        "stage": stage,
        "status": "completed",
        **values,
    }
    write_json_atomic(state_path, state)


def subprocess_environment(
    cfg: dict[str, Any], quiet_stage_logs: bool
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["YOLO_AUTOINSTALL"] = "false"
    if cfg.get("stable_diffusion", {}).get("local_files_only", True):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if str(cfg["device"]).lower() != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["device"])
    if quiet_stage_logs:
        env["DETECTION_GLIGEN_SDEDIT_FILE_LOG_ONLY"] = "1"
    else:
        env.pop("DETECTION_GLIGEN_SDEDIT_FILE_LOG_ONLY", None)
    return env


def run_command(
    command: list[str], cfg: dict[str, Any], quiet_stage_logs: bool = True
) -> None:
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=subprocess_environment(cfg, quiet_stage_logs),
        check=True,
    )


def class_subset_args(cfg: dict[str, Any]) -> list[str]:
    subset = cfg.get("class_subset", {})
    if not bool(subset.get("enabled", False)):
        return []
    class_ids = subset.get("class_ids")
    if class_ids:
        return ["--class-ids", ",".join(str(int(value)) for value in class_ids)]
    return ["--classes-subset", str(int(subset.get("count", 10)))]


def data_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    dataset_yaml = Path(cfg["dataset"])
    root, _ = dataset_root_from_yaml(dataset_yaml)
    return {
        "root": root,
        "dataset_yaml": dataset_yaml,
        "train_annotations": root / "annotation/instances_train2017.json",
        "val_annotations": root / "annotation/instances_val2017.json",
        "train_images": root / "images/train2017",
        "val_images": root / "images/val2017",
        "train_labels": root / "labels/train2017",
        "val_labels": root / "labels/val2017",
    }


def guide_data_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    paths = data_paths(cfg)
    gradient = cfg["prompt_optimization"]["gradient_alignment"]
    guide_root = (
        resolved(gradient["guide_root"], paths["dataset_yaml"].parent)
        if gradient.get("guide_root")
        else paths["root"]
    )
    if gradient.get("guide_split", "val") != "val":
        raise ValueError("Prompt gradient alignment currently requires guide_split=val")
    return guide_root / "images/val2017", guide_root / "labels/val2017"


def ensure_missing_labels(cfg: dict[str, Any]) -> None:
    """Create empty labels for missing train, val, and enabled Guide images."""
    paths = data_paths(cfg)
    directories = [
        (paths["train_images"], paths["train_labels"]),
        (paths["val_images"], paths["val_labels"]),
    ]
    if cfg["prompt_optimization"]["gradient_alignment"].get("enabled", True):
        directories.append(guide_data_paths(cfg))
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for images, labels in dict.fromkeys(directories):
        require_directory(images, "Images")
        require_directory(labels, "Labels")
        checked = created = 0
        for image in sorted(images.iterdir()):
            if not image.is_file() or image.suffix.lower() not in image_suffixes:
                continue
            checked += 1
            label = labels / f"{image.stem}.txt"
            try:
                # Exclusive creation also prevents overwrites by concurrent runs.
                with label.open("x", encoding="utf-8"):
                    pass
            except FileExistsError:
                require_file(label, "Existing label")
            else:
                created += 1
        print(
            f"label check: {images}, checked {checked} images, "
            f"created {created} empty labels in {labels}",
            flush=True,
        )


def model_for_round(state: dict[str, Any], cfg: dict[str, Any], round_idx: int) -> Path:
    if round_idx == 1:
        return Path(cfg["baseline"])
    previous = round_state(state, round_idx - 1)["retrain"].get("best")
    if not previous:
        raise RuntimeError(
            f"Round {round_idx - 1} has no completed cos-Prompt best.pt"
        )
    path = Path(previous)
    require_file(path, f"Round {round_idx - 1} cos-Prompt best checkpoint")
    return path


def run_mine(
    state: dict[str, Any],
    state_path: Path,
    cfg: dict[str, Any],
    round_idx: int,
    model: Path,
) -> None:
    record = round_state(state, round_idx)["mine"]
    if record["status"] == "completed":
        return
    paths = data_paths(cfg)
    output = Path(cfg["workspace"]) / f"round_{round_idx}" / "failures.json"
    forgotten = Path(cfg["workspace"]) / "state" / "forgotten.json"
    mark_stage_running(
        state, state_path, round_idx, "mine", output=str(output), model=str(model)
    )
    mining = cfg["mining"]
    inference = cfg["model"].get("inference", {})
    command = [
        sys.executable,
        "-m",
        "detection_gligen_cos.mine_failures_ultralytics",
        "--annotations",
        str(paths["train_annotations"]),
        "--images",
        str(paths["train_images"]),
        "--dataset-yaml",
        str(paths["dataset_yaml"]),
        "--model-family",
        str(cfg["model"]["family"]),
        "--ultralytics-root",
        str(cfg["model"]["ultralytics_root"]),
        "--weights",
        str(model),
        "--output",
        str(output),
        "--state",
        str(forgotten),
        "--device",
        str(cfg["model"]["device"]),
        "--image-size",
        str(cfg["model"]["image_size"]),
        "--iou",
        str(inference.get("iou", 0.7)),
        "--max-det",
        str(inference.get("max_det", 300)),
        "--conf",
        str(mining["conf"]),
        "--match-iou",
        str(mining["match_iou"]),
        "--low-confidence",
        str(mining["low_confidence"]),
        "--ambiguous-gap",
        str(mining["ambiguous_gap"]),
        "--forgotten-threshold",
        str(mining["forgotten_threshold"]),
    ] + class_subset_args(cfg)
    if mining.get("max_images") is not None:
        command += ["--max-images", str(mining["max_images"])]
    if mining.get("min_failures_per_class") is not None:
        command += ["--min-failures-per-class", str(mining["min_failures_per_class"])]
    print(f"round {round_idx}, begin to mine", flush=True)
    run_command(command, cfg)
    require_file(output, "Failure mining output")
    failure_count = len(
        json.loads(output.read_text(encoding="utf-8")).get("failures", [])
    )
    mark_stage_complete(
        state, state_path, round_idx, "mine", output=str(output), failures=failure_count
    )
    print(f"round {round_idx}, have mined.", flush=True)


def latest_prompt_checkpoint(prompts: Path) -> tuple[Path | None, int]:
    metadata = prompts / "object_prompts.json"
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        latest = payload.get("latest")
        if latest and (prompts / latest).is_file():
            match = re.search(r"-(\d+)\.bin$", str(latest))
            return prompts / latest, int(match.group(1)) if match else 0
    checkpoints = sorted(
        (
            (int(match.group(1)), path)
            for path in prompts.glob("learned_embeds-*.bin")
            if (match := re.search(r"-(\d+)\.bin$", path.name))
        ),
        key=lambda item: item[0],
    )
    return (checkpoints[-1][1], checkpoints[-1][0]) if checkpoints else (None, 0)


def prompt_alignment_state(prompts: Path) -> dict[str, Any]:
    metadata_path = prompts / "object_prompts.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    alignment = payload.get("gradient_alignment", {})
    if not isinstance(alignment, dict):
        return {}
    cache = alignment.get("guide_cache") or {}
    return {
        "guide_cache": cache.get("cache_path"),
        "guide_metadata_hash": alignment.get("guide_metadata_hash"),
        "alignment_statistics": alignment.get("statistics"),
    }


def run_optimize(
    state: dict[str, Any],
    state_path: Path,
    cfg: dict[str, Any],
    round_idx: int,
    model: Path,
) -> None:
    record = round_state(state, round_idx)["optimize"]
    if record["status"] == "completed":
        return
    paths = data_paths(cfg)
    round_dir = Path(cfg["workspace"]) / f"round_{round_idx}"
    failures = round_dir / "failures.json"
    prompts = round_dir / "prompts"
    optimization = cfg["prompt_optimization"]
    maximum_steps = int(optimization["max_train_steps"])
    resume_token: Path | None = None
    initial_step = 0
    if record["status"] == "running":
        resume_token, initial_step = latest_prompt_checkpoint(prompts)
    elif optimization.get("resume_from_previous", False) and round_idx > 1:
        resume_token, _ = latest_prompt_checkpoint(
            Path(cfg["workspace"]) / f"round_{round_idx - 1}" / "prompts"
        )
    if initial_step >= maximum_steps and resume_token is not None:
        mark_stage_complete(
            state,
            state_path,
            round_idx,
            "optimize",
            output=str(prompts),
            latest=str(resume_token),
            completed_steps=maximum_steps,
            **prompt_alignment_state(prompts),
        )
        print(f"round {round_idx}, have optimized prompts.", flush=True)
        return

    mark_stage_running(
        state,
        state_path,
        round_idx,
        "optimize",
        output=str(prompts),
        model=str(model),
        completed_steps=initial_step,
    )
    inference = cfg["model"].get("inference", {})
    sd = cfg["stable_diffusion"]
    gradient = optimization["gradient_alignment"]
    guide_images, guide_labels = guide_data_paths(cfg)
    command = [
        sys.executable,
        "-m",
        "detection_gligen_cos.optimize_object_prompts_ultralytics",
        "--pretrained-model-name-or-path",
        str(sd["model"]),
        "--annotations",
        str(paths["train_annotations"]),
        "--images",
        str(paths["train_images"]),
        "--dataset-yaml",
        str(paths["dataset_yaml"]),
        "--failures",
        str(failures),
        "--model-family",
        str(cfg["model"]["family"]),
        "--ultralytics-root",
        str(cfg["model"]["ultralytics_root"]),
        "--weights",
        str(model),
        "--detector-image-size",
        str(cfg["model"]["image_size"]),
        "--detector-iou",
        str(inference.get("iou", 0.7)),
        "--detector-max-det",
        str(inference.get("max_det", 300)),
        "--guide-images",
        str(guide_images),
        "--guide-labels",
        str(guide_labels),
        "--alignment-cache-dir",
        str(prompts / "gradient_alignment"),
        "--alignment-image-size",
        str(gradient["image_size"]),
        "--guide-batch-size",
        str(gradient["guide_batch_size"]),
        "--guide-workers",
        str(gradient["workers"]),
        "--alignment-parameter-scope",
        str(gradient["parameter_scope"]),
        "--alignment-implementation",
        str(gradient["implementation"]),
        "--alignment-loss",
        str(gradient["loss"]),
        "--alignment-weight",
        str(gradient["weight"]),
        "--alignment-margin",
        str(gradient["margin"]),
        "--alignment-warmup-steps",
        str(gradient["warmup_steps"]),
        "--alignment-epsilon",
        str(gradient["epsilon"]),
        "--alignment-max-objects",
        str(gradient["max_objects_per_image"]),
        "--alignment-min-bbox-area-ratio",
        str(gradient["min_bbox_area_ratio"]),
        "--output-dir",
        str(prompts),
        "--prompt-scope",
        str(optimization["prompt_scope"]),
        "--num-new-tokens",
        str(optimization["num_new_tokens"]),
        "--initializer-token",
        str(optimization["initializer_token"]),
        "--init-mode",
        str(optimization["init_mode"]),
        "--resume-mode",
        str(optimization.get("resume_mode", "overwrite")),
        "--resolution",
        str(optimization["resolution"]),
        "--max-train-steps",
        str(maximum_steps),
        "--initial-step",
        str(initial_step),
        "--learning-rate",
        str(optimization["learning_rate"]),
        "--strength",
        str(optimization["strength"]),
        "--num-inference-steps",
        str(optimization["num_inference_steps"]),
        "--semantic-weight",
        str(optimization["semantic_weight"]),
        "--detector-weight",
        str(optimization["detector_weight"]),
        "--class-semantic-weight",
        str(optimization.get("class_semantic_weight", 0.0)),
        "--class-semantic-model",
        str(optimization.get("class_semantic_model", "openai/clip-vit-large-patch14")),
        "--save-steps",
        str(optimization["save_steps"]),
        "--log-every",
        str(optimization["log_every"]),
        "--seed",
        str(cfg["seed"]),
    ] + class_subset_args(cfg)
    command.append(
        "--alignment-enabled" if gradient.get("enabled", True) else "--no-alignment-enabled"
    )
    command.append(
        "--alignment-scaleup" if gradient.get("scaleup", False) else "--no-alignment-scaleup"
    )
    command.append(
        "--alignment-cache" if gradient.get("cache", True) else "--no-alignment-cache"
    )
    if gradient.get("enable_layout_filter", False):
        command.append("--alignment-enable-layout-filter")
    if gradient.get("max_guide_images") is not None:
        command += ["--max-guide-images", str(gradient["max_guide_images"])]
    if resume_token is not None:
        command += ["--resume-token", str(resume_token)]
    if optimization.get("include_class_name", False):
        command.append("--include-class-name")
    if optimization.get(
        "class_semantic_local_files_only", sd.get("local_files_only", True)
    ):
        command.append("--class-semantic-local-files-only")
    quality = optimization.get("instance_quality_filter", {})
    if quality.get("enabled", False):
        command.append("--use-instance-quality-filter")
        if quality.get("results"):
            command += ["--instance-quality-results", str(resolved(quality["results"]))]
        if quality.get("config"):
            command += ["--instance-quality-config", str(resolved(quality["config"]))]
    if sd.get("local_files_only", True):
        command.append("--local-files-only")

    print(f"round {round_idx}, begin to optimize prompts", flush=True)
    run_command(command, cfg)
    latest, completed_steps = latest_prompt_checkpoint(prompts)
    if latest is None or completed_steps < maximum_steps:
        raise RuntimeError(
            f"Prompt optimization stopped before step {maximum_steps}: latest={latest}, step={completed_steps}"
        )
    mark_stage_complete(
        state,
        state_path,
        round_idx,
        "optimize",
        output=str(prompts),
        latest=str(latest),
        completed_steps=completed_steps,
        **prompt_alignment_state(prompts),
    )
    print(f"round {round_idx}, have optimized prompts.", flush=True)


def manifest_count(path: Path) -> int:
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Generation manifest must be a list: {path}")
    return len(payload)


def run_generate(
    state: dict[str, Any], state_path: Path, cfg: dict[str, Any], round_idx: int
) -> None:
    record = round_state(state, round_idx)["generate"]
    if record["status"] == "completed":
        return
    paths = data_paths(cfg)
    round_dir = Path(cfg["workspace"]) / f"round_{round_idx}"
    prompts = round_dir / "prompts"
    output = round_dir / "synthetic"
    manifest = output / "manifest.json"
    if record["status"] == "pending" and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Fresh generation output is not empty; resume requires the state JSON: {output}"
        )
    generated = manifest_count(manifest)
    if generated >= int(cfg["syn_sample"]):
        mark_stage_complete(
            state,
            state_path,
            round_idx,
            "generate",
            output=str(output),
            manifest=str(manifest),
            generated_samples=generated,
            generation_target=int(cfg["syn_sample"]),
        )
        print(f"round {round_idx}, have generated {generated} samples.", flush=True)
        return
    mark_stage_running(
        state,
        state_path,
        round_idx,
        "generate",
        output=str(output),
        generated_samples=generated,
        generation_target=int(cfg["syn_sample"]),
    )
    generation = cfg["generation"]
    generation_model = generation.get("model") or generation.get("gligen_model")
    command = [
        sys.executable,
        "-m",
        "detection_gligen_cos.generate_gligen_sdedit_examples",
        "--pretrained-model-name-or-path",
        str(generation_model),
        "--annotations",
        str(paths["train_annotations"]),
        "--images",
        str(paths["train_images"]),
        "--dataset-yaml",
        str(paths["dataset_yaml"]),
        "--tokens-dir",
        str(prompts),
        "--output-dir",
        str(output),
        "--max-images",
        str(cfg["syn_sample"]),
        "--max-objects-per-image",
        str(generation["max_objects_per_image"]),
        "--width",
        str(generation["width"]),
        "--height",
        str(generation["height"]),
        "--strength",
        str(generation["strength"]),
        "--guidance-scale",
        str(generation["guidance_scale"]),
        "--num-inference-steps",
        str(generation["num_inference_steps"]),
        "--gligen-scheduled-sampling-beta",
        str(generation["gligen_scheduled_sampling_beta"]),
        "--global-prompt-source",
        str(generation["global_prompt_source"]),
        "--global-prompt-template",
        str(generation["global_prompt_template"]),
        "--negative-prompt",
        str(generation["negative_prompt"]),
        "--device",
        str(generation["device"]),
        "--seed",
        str(int(cfg["seed"]) + round_idx),
        "--state-json",
        str(state_path),
    ] + class_subset_args(cfg)
    if generated:
        command.append("--resume")
    if generation.get("variant") is not None:
        command += ["--variant", str(generation["variant"])]
    if generation.get("noise_timestep") is not None:
        command += ["--noise-timestep", str(generation["noise_timestep"])]
    include_name = generation.get("include_class_name")
    if include_name is True:
        command.append("--include-class-name")
    elif include_name is False:
        command.append("--no-include-class-name")
    if generation.get("enable_generation_filter", False):
        command += [
            "--enable-generation-filter",
            "--min-bbox-area-ratio",
            str(generation.get("min_bbox_area_ratio", 0.0)),
        ]
    if generation.get("save_visualization", False):
        command.append("--save-visualization")
    if generation.get("local_files_only", True):
        command.append("--local-files-only")

    print(f"round {round_idx}, begin to generate samples", flush=True)
    run_command(command, cfg)
    state.clear()
    state.update(read_json(state_path))
    generated = manifest_count(manifest)
    if generated < int(cfg["syn_sample"]):
        raise RuntimeError(
            f"Round {round_idx} generated {generated}/{cfg['syn_sample']} requested samples; candidate images were exhausted"
        )
    mark_stage_complete(
        state,
        state_path,
        round_idx,
        "generate",
        output=str(output),
        manifest=str(manifest),
        generated_samples=generated,
        generation_target=int(cfg["syn_sample"]),
    )
    print(f"round {round_idx}, have generated {generated} samples.", flush=True)


def retrain_dataset_yaml(cfg: dict[str, Any], round_idx: int) -> Path:
    paths = data_paths(cfg)
    _, source = dataset_root_from_yaml(paths["dataset_yaml"])
    train = ["images/train2017"]
    for index in range(1, round_idx + 1):
        train.append(
            str(
                (
                    Path(cfg["workspace"])
                    / f"round_{index}"
                    / "synthetic"
                    / "images"
                ).resolve()
            )
        )
    payload: dict[str, Any] = {
        "path": str(paths["root"]),
        "train": train,
        "val": "images/val2017",
        "names": source["names"],
    }
    if source.get("channels") is not None:
        payload["channels"] = source["channels"]
    output = Path(cfg["workspace"]) / f"round_{round_idx}" / "datasets" / "cos_prompt.yaml"
    write_yaml_atomic(output, payload)
    return output


def retrain_run_dir(cfg: dict[str, Any], round_idx: int) -> tuple[str, Path]:
    name = retrain_run_name(cfg, round_idx)
    return name, project_root(cfg) / name


def run_retrain(
    state: dict[str, Any], state_path: Path, cfg: dict[str, Any], round_idx: int
) -> None:
    retrain_record = round_state(state, round_idx)["retrain"]
    if retrain_record["status"] == "completed":
        return
    dataset_yaml = retrain_dataset_yaml(cfg, round_idx)
    name, run_dir = retrain_run_dir(cfg, round_idx)
    last = run_dir / "weights/last.pt"
    best = run_dir / "weights/best.pt"
    retrain_record = mark_stage_running(
        state,
        state_path,
        round_idx,
        "retrain",
        name=name,
        data=str(dataset_yaml),
        run_dir=str(run_dir),
        last=str(last),
        best=str(best),
        resuming=last.is_file(),
        epoch=int(retrain_record.get("epoch", 0)),
    )
    command = [
        sys.executable,
        "-m",
        "detection_gligen_cos.retrain_runner",
        "--internal-retrain",
        "--state-json",
        str(state_path),
        "--retrain-round",
        str(round_idx),
    ]
    run_command(command, cfg, quiet_stage_logs=False)
    state.clear()
    state.update(read_json(state_path))
    require_file(best, f"Retrained best checkpoint for {name}")
    mark_stage_complete(
        state,
        state_path,
        round_idx,
        "retrain",
        name=name,
        data=str(dataset_yaml),
        run_dir=str(run_dir),
        last=str(last),
        best=str(best),
    )
    print(f"round {round_idx}, retrain completed: {name}", flush=True)


def update_training_epoch(state_path: Path, round_idx: int, epoch: int) -> None:
    state = read_json(state_path)
    retrain_record = round_state(state, round_idx)["retrain"]
    retrain_record["epoch"] = int(epoch)
    state["current"] = {
        "round": round_idx,
        "stage": "retrain",
        "status": "running",
        "retrain_name": retrain_record["name"],
        "epoch": int(epoch),
    }
    write_json_atomic(state_path, state)


def internal_retrain(args: argparse.Namespace) -> None:
    state_path = Path(args.state_json).expanduser().resolve()
    state = read_json(state_path)
    cfg = state["config"]
    round_idx = int(args.retrain_round)
    retrain_record = round_state(state, round_idx)["retrain"]
    data_yaml = Path(retrain_record["data"])
    run_dir = Path(retrain_record["run_dir"])
    last = run_dir / "weights/last.pt"
    best = run_dir / "weights/best.pt"

    ultralytics_root = Path(cfg["model"]["ultralytics_root"])
    root_text = str(ultralytics_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import ultralytics
    from ultralytics import YOLO

    imported = Path(ultralytics.__file__).resolve()
    try:
        imported.relative_to(ultralytics_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported Ultralytics from {imported}, expected under {ultralytics_root}"
        ) from exc

    if last.is_file():
        model = YOLO(str(last))
    else:
        model = YOLO(str(cfg["model_yaml"]))

    def on_fit_epoch_end(trainer) -> None:
        update_training_epoch(state_path, round_idx, int(trainer.epoch) + 1)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    if last.is_file():
        try:
            model.train(resume=True)
        except AssertionError as exc:
            message = str(exc)
            if (
                best.is_file()
                and "training to" in message
                and "nothing to resume" in message
            ):
                return
            raise
        return

    retrain = cfg["retrain"]
    model.train(
        data=str(data_yaml),
        epochs=int(retrain["epochs"]),
        imgsz=int(retrain["image_size"]),
        device=str(cfg["model"]["device"]),
        batch=int(retrain["batch_size"]),
        project=str(project_root(cfg)),
        name=str(retrain_record["name"]),
        patience=int(retrain["patience"]),
        workers=int(retrain["workers"]),
        exist_ok=bool(retrain["exist_ok"]),
    )


def run_loop(state: dict[str, Any], state_path: Path) -> None:
    cfg = state["config"]
    ensure_missing_labels(cfg)
    for round_idx in range(1, int(cfg["rounds"]) + 1):
        record = round_state(state, round_idx)
        if record["status"] == "completed":
            continue
        model = model_for_round(state, cfg, round_idx)
        record["input_model"] = str(model)
        write_json_atomic(state_path, state)
        run_mine(state, state_path, cfg, round_idx, model)
        run_optimize(state, state_path, cfg, round_idx, model)
        run_generate(state, state_path, cfg, round_idx)
        run_retrain(state, state_path, cfg, round_idx)
        record = round_state(state, round_idx)
        record["status"] = "completed"
        record["completed_at"] = utc_now()
        write_json_atomic(state_path, state)

    state["status"] = "completed"
    state["current"] = {
        "round": int(cfg["rounds"]),
        "stage": "complete",
        "status": "completed",
        "final_model": round_state(state, int(cfg["rounds"]))["retrain"]["best"],
    }
    write_json_atomic(state_path, state)
    print(f"all {cfg['rounds']} rounds have completed.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine, optimize with Guide-gradient alignment, generate, and retrain one fresh detector per round."
    )
    parser.add_argument(
        "--overall-yaml",
        default=None,
        help=f"New-run configuration (default: {DEFAULT_OVERALL_YAML}).",
    )
    parser.add_argument(
        "--state-json",
        default=None,
        help="Resume using only this JSON; it contains the full configuration.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Override overall_yaml baseline .pt checkpoint.",
    )
    parser.add_argument(
        "--model-yaml",
        default=None,
        help="Override the from-scratch Ultralytics model YAML.",
    )
    parser.add_argument(
        "--dataset", default=None, help="Override the original dataset YAML."
    )
    parser.add_argument(
        "--round",
        dest="rounds",
        type=int,
        default=None,
        help="Override number of rounds.",
    )
    parser.add_argument(
        "--syn-smaple",
        "--syn-sample",
        dest="syn_sample",
        type=int,
        default=None,
        help="Generated samples per round (the misspelled requested form and corrected alias are both accepted).",
    )
    parser.add_argument(
        "--device", default=None, help="One physical GPU id, default 6, or cpu."
    )
    parser.add_argument(
        "--internal-retrain", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--retrain-round", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.internal_retrain:
        if not args.state_json or args.retrain_round is None:
            raise ValueError("Internal retrain requires state JSON and round")
        internal_retrain(args)
        return

    requested_state = (
        Path(args.state_json).expanduser().resolve() if args.state_json else None
    )
    if requested_state is not None and requested_state.is_file():
        state = read_json(requested_state)
        state_path = requested_state
        print(f"resume from {state_path}", flush=True)
    else:
        overall_yaml = resolved(args.overall_yaml or DEFAULT_OVERALL_YAML)
        require_file(overall_yaml, "Overall YAML")
        raw = yaml.safe_load(overall_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Overall YAML must contain a mapping: {overall_yaml}")
        cfg = normalize_effective_config(raw, args, overall_yaml)
        validate_effective_config(cfg)
        state_path = requested_state or Path(cfg["workspace"]) / "retrain_state.json"
        if state_path.exists():
            raise FileExistsError(
                f"State JSON already exists; resume it with --state-json: {state_path}"
            )
        state = initial_state(cfg, state_path)
        write_json_atomic(state_path, state)
        print(f"state json: {state_path}", flush=True)

    validate_effective_config(state["config"])
    run_loop(state, state_path)


if __name__ == "__main__":
    main()
