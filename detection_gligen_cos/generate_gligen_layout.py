# 新增独立 GLIGEN layout-conditioned generation 方案
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from .common import CocoMini, parse_class_subset, setup_logger, subset_label
from .generate_hard_examples import build_object_prompt, choose_group, install_embeddings, load_object_prompts


def resolved(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def cfg_get(config: dict, dotted: str, default=None):
    current = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def xywh_to_gligen_box(bbox, image_width: float, image_height: float) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0.0, min(1.0, x / image_width))
    y1 = max(0.0, min(1.0, y / image_height))
    x2 = max(0.0, min(1.0, (x + w) / image_width))
    y2 = max(0.0, min(1.0, (y + h) / image_height))
    return [x1, y1, max(x2, x1 + 1e-4), max(y2, y1 + 1e-4)]


def gligen_box_to_yolo_line(class_id: int, box: list[float]) -> str:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2
    yc = y1 + h / 2
    return f"{class_id} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"


def build_global_prompt(phrases: list[str], template: str) -> str:
    return template.format(phrases=", ".join(phrases))


def _state_dict_has_gligen_unet_keys(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        keys = load_file(str(path), device="cpu").keys()
    else:
        keys = torch.load(path, map_location="cpu").keys()
    return any(k.startswith("position_net") for k in keys) and any(".fuser." in k for k in keys)


def prepare_gligen_model_for_diffusers(model_path: str | Path, output_dir: Path,
                                       variant: str | None, logger) -> tuple[Path, str | None]:
    """Create a temporary load directory for old converted GLIGEN checkpoints.

    The official diffusers GLIGEN checkpoint may contain an old
    ``use_gated_attention`` config key. Modern diffusers expects
    ``attention_type="gated"`` to build the GLIGEN fusers and position_net.
    Some local fp16 variants also contain ordinary SD UNet weights without
    GLIGEN keys, while the full ``diffusion_pytorch_model.bin`` is correct.
    This helper patches only a temporary config and disables such a bad variant.
    """
    model_path = Path(model_path)
    unet_dir = model_path / "unet"
    config_path = unet_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    needs_config_patch = config.get("use_gated_attention") is True and config.get("attention_type") != "gated"
    load_variant = variant

    if variant:
        variant_candidates = [
            unet_dir / f"diffusion_pytorch_model.{variant}.safetensors",
            unet_dir / f"diffusion_pytorch_model.{variant}.bin",
        ]
        variant_weight = next((p for p in variant_candidates if p.exists()), None)
        if variant_weight is not None and not _state_dict_has_gligen_unet_keys(variant_weight):
            logger.warning(
                "GLIGEN variant '%s' disabled for loading: %s has no position_net/fuser weights; using full UNet weights and torch_dtype casting",
                variant, variant_weight,
            )
            load_variant = None

    if not needs_config_patch and load_variant == variant:
        return model_path, load_variant

    temp_dir = output_dir / "_gligen_model_load"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for child in model_path.iterdir():
        if child.name == "unet":
            continue
        os.symlink(child, temp_dir / child.name, target_is_directory=child.is_dir())

    temp_unet = temp_dir / "unet"
    temp_unet.mkdir()
    for child in unet_dir.iterdir():
        if child.name == "config.json":
            continue
        os.symlink(child, temp_unet / child.name, target_is_directory=child.is_dir())
    if needs_config_patch:
        config.pop("use_gated_attention", None)
        config["attention_type"] = "gated"
        logger.info("GLIGEN temporary UNet config patched: use_gated_attention -> attention_type=gated")
    (temp_unet / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("GLIGEN temporary model load dir: %s variant=%s", temp_dir, load_variant)
    return temp_dir, load_variant


def prepare_layout(anns: list[dict], dataset: CocoMini, groups: dict, prompt_scope: str,
                   include_class_name: bool, max_objects: int) -> tuple[list[str], list[list[float]], list[dict]]:
    phrases: list[str] = []
    boxes: list[list[float]] = []
    label_items: list[dict] = []
    for ann in anns:
        if len(phrases) >= max_objects:
            break
        class_id = int(ann["class_id"])
        group = choose_group(groups, class_id, prompt_scope)
        if group not in groups:
            continue
        info = dataset.images[int(ann["image_id"])]
        class_name = dataset.class_names[class_id]
        phrase = build_object_prompt(class_name, groups[group], include_class_name)
        box = xywh_to_gligen_box(ann["bbox"], float(info["width"]), float(info["height"]))
        phrases.append(phrase)
        boxes.append(box)
        label_items.append({
            "annotation_id": int(ann["id"]),
            "class_id": class_id,
            "class_name": class_name,
            "group": group,
            "phrase": phrase,
            "gligen_box": box,
        })
    return phrases, boxes, label_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full synthetic images with GLIGEN text+box layout conditioning.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pretrained-model-name-or-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--images", default=None)
    parser.add_argument("--tokens-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-scope", choices=["class", "class_failure"], default=None)
    parser.add_argument("--include-class-name", action="store_true", default=None)
    parser.add_argument("--no-include-class-name", action="store_false", dest="include_class_name")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-objects-per-image", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--gligen-scheduled-sampling-beta", type=float, default=None)
    parser.add_argument("--global-prompt-template", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    config = load_config(args.config)
    data_cfg = cfg_get(config, "data", {})
    gligen_cfg = cfg_get(config, "gligen_generation", {})
    class_cfg = cfg_get(config, "class_subset", {})

    model_path = args.pretrained_model_name_or_path or gligen_cfg.get("model")
    model_variant = gligen_cfg.get("variant")
    annotations = resolved(root, args.annotations or gligen_cfg.get("annotations") or data_cfg.get("train_annotations"))
    images = resolved(root, args.images or gligen_cfg.get("images") or data_cfg.get("train_images"))
    tokens_dir = resolved(root, args.tokens_dir or gligen_cfg.get("tokens_dir"))
    output = resolved(root, args.output_dir or gligen_cfg.get("output_dir") or "outputs/detection_gligen_sdedit/gligen_generation")
    if not model_path or annotations is None or images is None or tokens_dir is None or output is None:
        raise ValueError("model, annotations, images, tokens_dir and output_dir are required via CLI or config.")

    local_files_only = bool(args.local_files_only or gligen_cfg.get("local_files_only", False))
    max_images = int(args.max_images if args.max_images is not None else gligen_cfg.get("max_images", 500))
    max_objects = int(args.max_objects_per_image if args.max_objects_per_image is not None else gligen_cfg.get("max_objects_per_image", 8))
    height = int(args.height if args.height is not None else gligen_cfg.get("height", 512))
    width = int(args.width if args.width is not None else gligen_cfg.get("width", 512))
    steps = int(args.num_inference_steps if args.num_inference_steps is not None else gligen_cfg.get("num_inference_steps", 50))
    guidance = float(args.guidance_scale if args.guidance_scale is not None else gligen_cfg.get("guidance_scale", 7.5))
    beta = float(args.gligen_scheduled_sampling_beta if args.gligen_scheduled_sampling_beta is not None else gligen_cfg.get("gligen_scheduled_sampling_beta", 0.3))
    prompt_template = args.global_prompt_template or gligen_cfg.get("global_prompt_template", "{phrases}, photo, highly detailed, photorealistic")
    negative_prompt = args.negative_prompt or gligen_cfg.get("negative_prompt", "wrong class, duplicate object, deformed, cropped, text, watermark")
    seed = int(args.seed if args.seed is not None else gligen_cfg.get("seed", 17))
    device = args.device or gligen_cfg.get("device", "cuda:0")

    class_ids = args.class_ids if args.class_ids is not None else class_cfg.get("class_ids")
    classes_subset = args.classes_subset if args.classes_subset is not None else class_cfg.get("count")
    class_subset_enabled = bool(class_cfg.get("enabled", False)) or args.class_ids is not None or args.classes_subset is not None
    class_subset = parse_class_subset(class_subset_enabled, classes_subset, class_ids)

    if args.clean_output or bool(gligen_cfg.get("clean_output", False)):
        for child in (output / "images", output / "labels"):
            if child.exists():
                shutil.rmtree(child)
        for child in (output / "manifest.json", output / "selected_images.json"):
            child.unlink(missing_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)

    dataset = CocoMini(annotations, images, class_subset=class_subset)
    logger.info("GLIGEN generation dataset loaded: images=%d annotations=%d class_subset=%s",
                len(dataset.images), sum(len(v) for v in dataset.annotations.values()), subset_label(class_subset))
    meta, embeddings = load_object_prompts(tokens_dir)
    prompt_scope = args.prompt_scope or gligen_cfg.get("prompt_scope") or meta["prompt_scope"]
    include_class_name = bool(meta.get("include_class_name", False)) if args.include_class_name is None else bool(args.include_class_name)
    logger.info("object prompts loaded: tokens_dir=%s groups=%d latest=%s scope=%s include_class_name=%s",
                tokens_dir, len(meta["groups"]), meta["latest"], prompt_scope, include_class_name)

    from diffusers import StableDiffusionGLIGENPipeline

    logger.info("loading GLIGEN pipeline: model=%s variant=%s local_files_only=%s", model_path, model_variant, local_files_only)
    load_model_path, load_variant = prepare_gligen_model_for_diffusers(model_path, output, model_variant, logger)
    load_kwargs = {}
    if load_variant:
        load_kwargs["variant"] = str(load_variant)
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        load_model_path,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
        **load_kwargs,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    install_embeddings(pipe.tokenizer, pipe.text_encoder, meta["groups"], embeddings)
    logger.info("learned embeddings installed into GLIGEN text encoder: tokens=%d",
                sum(len(v) for v in meta["groups"].values()))

    candidates = [(image_id, path, anns) for image_id, path, anns in dataset.iter_images(None) if anns]
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(max_images, len(candidates)))
    (output / "selected_images.json").write_text(json.dumps([int(x[0]) for x in selected], indent=2), encoding="utf-8")
    logger.info("GLIGEN generation start: candidates=%d selected=%d max_objects_per_image=%d size=%dx%d steps=%d beta=%s seed=%d",
                len(candidates), len(selected), max_objects, width, height, steps, beta, seed)

    manifest = []
    skipped = 0
    for idx, (image_id, _path, anns) in enumerate(tqdm(selected, desc="gligen generation"), start=1):
        phrases, boxes, label_items = prepare_layout(
            anns, dataset, meta["groups"], prompt_scope, include_class_name, max_objects
        )
        if not phrases:
            skipped += 1
            continue
        prompt = build_global_prompt(phrases, prompt_template)
        generator = torch.Generator(device=device).manual_seed(seed + int(image_id))
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            gligen_scheduled_sampling_beta=beta,
            gligen_phrases=phrases,
            gligen_boxes=boxes,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]
        name = f"gligen_{int(image_id):012d}.jpg"
        image.save(output / "images" / name, quality=95)
        label_lines = [gligen_box_to_yolo_line(item["class_id"], item["gligen_box"]) for item in label_items]
        (output / "labels" / Path(name).with_suffix(".txt")).write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        manifest.append({
            "image_id": int(image_id),
            "file_name": name,
            "source": "gligen_layout",
            "prompt": prompt,
            "layout_objects": label_items,
            "labels_scope": "gligen_layout_boxes",
            "labels_inherited_from_real": False,
            "width": width,
            "height": height,
        })
        if len(manifest) == 1 or len(manifest) % 25 == 0 or idx == len(selected):
            logger.info("GLIGEN progress: processed=%d/%d synthetic=%d skipped=%d latest=%s objects=%d",
                        idx, len(selected), len(manifest), skipped, name, len(label_items))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("GLIGEN generation complete: synthetic=%d skipped=%d output=%s manifest=%s",
                len(manifest), skipped, output, output / "manifest.json")


if __name__ == "__main__":
    main()
