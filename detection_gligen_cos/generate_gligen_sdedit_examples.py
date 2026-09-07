from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .common import CocoMini, parse_class_subset, setup_logger, subset_label
from .generate_gligen_layout import (
    build_global_prompt,
    gligen_box_to_yolo_line,
    prepare_gligen_model_for_diffusers,
    xywh_to_gligen_box,
)
from .generate_hard_examples import choose_group, install_embeddings, load_object_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hard examples with full-image GLIGEN + SDEdit grounding instead of ROI inpainting."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="Ultralytics dataset YAML whose names define output class ids and COCO category mapping.",
    )
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-scope", choices=["class", "class_failure"], default=None)
    parser.add_argument("--include-class-name", action="store_true", default=None)
    parser.add_argument("--no-include-class-name", action="store_false", dest="include_class_name")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-objects-per-image", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--noise-timestep", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--gligen-scheduled-sampling-beta", type=float, default=0.3)
    parser.add_argument("--global-prompt-template", default="{phrases}, photo, highly detailed, photorealistic")
    parser.add_argument("--global-prompt-source", choices=["template", "category_count", "empty"], default="template")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark")
    parser.add_argument("--enable-generation-filter", action="store_true")
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from an atomically saved manifest.")
    parser.add_argument("--state-json", default=None, help="Optional retrain-runner state JSON updated after each image.")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--save-visualization", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def update_resume_state(
    path: Path | None, generated: int, target: int, complete: bool
) -> None:
    """Record image-level generation progress in the standalone runner state."""

    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    current = payload.setdefault("current", {})
    current["stage"] = "generate"
    current["generated_samples"] = int(generated)
    current["generation_target"] = int(target)
    current["generate_complete"] = bool(complete)
    round_idx = current.get("round")
    round_record = payload.get("rounds", {}).get(str(round_idx))
    if isinstance(round_record, dict) and isinstance(round_record.get("generate"), dict):
        round_record["generate"].update(
            {
                "status": "running",
                "generated_samples": int(generated),
                "generation_target": int(target),
                "generate_complete": bool(complete),
            }
        )
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, payload)


def bbox_area_ratio(ann: dict, info: dict) -> float:
    _, _, width, height = [float(v) for v in ann["bbox"]]
    image_area = max(float(info["width"]) * float(info["height"]), 1.0)
    return max(width, 0.0) * max(height, 0.0) / image_area


def number_word(count: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(count, str(count))


def article_for_class_name(class_name: str) -> str:
    return "an" if class_name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def pluralize_class_name(class_name: str) -> str:
    if class_name == "person":
        return "persons"
    if class_name.endswith(("bus", "class", "glass")):
        return class_name + "es"
    if class_name.endswith("y") and len(class_name) > 1 and class_name[-2].lower() not in {"a", "e", "i", "o", "u"}:
        return class_name[:-1] + "ies"
    if class_name.endswith(("s", "x", "z", "ch", "sh")):
        return class_name + "es"
    return class_name + "s"


def join_natural_language(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def build_category_count_prompt(layout_items: list[dict]) -> str:
    counts = Counter(item["class_name"] for item in layout_items)
    parts = []
    for class_name in sorted(counts):
        count = counts[class_name]
        if count == 1:
            parts.append(f"{article_for_class_name(class_name)} {class_name}")
        else:
            parts.append(f"{number_word(count)} {pluralize_class_name(class_name)}")
    objects = join_natural_language(parts) or "objects"
    # return f"A photo with {objects}"
    return f"An image with {objects}, photo, highly detailed, photorealistic"

def build_clean_object_phrase(class_name: str, tokens: list[str], include_class_name: bool) -> str:
    token_text = ",".join(tokens)
    if include_class_name:
        return f"{class_name}, {token_text}"
    return token_text


def phrase_for_ann(meta: dict, dataset: CocoMini, ann: dict,
                   prompt_scope: str, include_class_name: bool) -> tuple[str, str | None, str]:
    class_id = int(ann["class_id"])
    class_name = dataset.class_names[class_id]
    group = choose_group(meta["groups"], class_id, prompt_scope)
    if group in meta["groups"]:
        return build_clean_object_phrase(class_name, meta["groups"][group], include_class_name), group, "learned_prompt"
    return class_name, None, "class_name_fallback"


def priority_layout_annotations(edit_anns: list[dict], full_anns: list[dict]) -> list[tuple[dict, str]]:
    edit_keys = {int(ann.get("id")) for ann in edit_anns if ann.get("id") is not None}
    prioritized = [(ann, "target_gt") for ann in full_anns if int(ann.get("id", -1)) in edit_keys]
    prioritized += [(ann, "normal_gt") for ann in full_anns if int(ann.get("id", -1)) not in edit_keys]
    return prioritized


def prepare_layout_items(
    meta: dict,
    dataset: CocoMini,
    image_id: int,
    full_anns: list[dict],
    edit_anns: list[dict],
    prompt_scope: str,
    include_class_name: bool,
    max_objects: int,
    enable_filter: bool,
    min_bbox_area_ratio: float,
) -> tuple[list[str], list[list[float]], list[dict], list[dict]]:
    info = dataset.images[int(image_id)]
    phrases: list[str] = []
    boxes: list[list[float]] = []
    layout_items: list[dict] = []
    filtered: list[dict] = []

    for ann, source in priority_layout_annotations(edit_anns, full_anns):
        ratio = bbox_area_ratio(ann, info)
        if enable_filter and ratio < min_bbox_area_ratio:
            filtered.append({
                "annotation_id": ann.get("id"),
                "class_id": int(ann["class_id"]),
                "class_name": dataset.class_names[int(ann["class_id"])],
                "bbox": ann["bbox"],
                "bbox_area_ratio": ratio,
                "layout_source": source,
                "filter_reason": "small_bbox",
            })
            continue
        if len(layout_items) >= max_objects:
            break
        phrase, group, phrase_source = phrase_for_ann(meta, dataset, ann, prompt_scope, include_class_name)
        box = xywh_to_gligen_box(ann["bbox"], float(info["width"]), float(info["height"]))
        phrases.append(phrase)
        boxes.append(box)
        layout_items.append({
            "annotation_id": ann.get("id"),
            "class_id": int(ann["class_id"]),
            "class_name": dataset.class_names[int(ann["class_id"])],
            "bbox": ann["bbox"],
            "bbox_area_ratio": ratio,
            "gligen_box": box,
            "layout_source": source,
            "group": group,
            "phrase_source": phrase_source,
            "phrase": phrase,
        })
    return phrases, boxes, layout_items, filtered


def write_layout_label_file(path: Path, layout_items: list[dict]) -> None:
    lines = [gligen_box_to_yolo_line(int(item["class_id"]), item["gligen_box"]) for item in layout_items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def choose_global_prompt(phrases: list[str], layout_items: list[dict], args: argparse.Namespace) -> str:
    if args.global_prompt_source == "empty":
        return ""
    if args.global_prompt_source == "category_count":
        return build_category_count_prompt(layout_items)
    return build_global_prompt(phrases, args.global_prompt_template)


def prepare_gligen_grounding(pipe, gligen_phrases: list[str], gligen_boxes: list[list[float]],
                             do_classifier_free_guidance: bool, device: torch.device) -> dict:
    max_objs = 30
    if len(gligen_boxes) > max_objs:
        gligen_phrases = gligen_phrases[:max_objs]
        gligen_boxes = gligen_boxes[:max_objs]

    tokenizer_inputs = pipe.tokenizer(gligen_phrases, padding=True, return_tensors="pt").to(device)
    phrase_embeds = pipe.text_encoder(**tokenizer_inputs).pooler_output
    n_objs = len(gligen_boxes)

    boxes = torch.zeros(max_objs, 4, device=device, dtype=pipe.text_encoder.dtype)
    boxes[:n_objs] = torch.tensor(gligen_boxes, device=device, dtype=pipe.text_encoder.dtype)
    positive_embeddings = torch.zeros(max_objs, pipe.unet.config.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype)
    positive_embeddings[:n_objs] = phrase_embeds
    masks = torch.zeros(max_objs, device=device, dtype=pipe.text_encoder.dtype)
    masks[:n_objs] = 1

    boxes = boxes.unsqueeze(0)
    positive_embeddings = positive_embeddings.unsqueeze(0)
    masks = masks.unsqueeze(0)
    if do_classifier_free_guidance:
        boxes = torch.cat([boxes] * 2)
        positive_embeddings = torch.cat([positive_embeddings] * 2)
        masks = torch.cat([masks] * 2)
        masks[:1] = 0
    return {"gligen": {"boxes": boxes, "positive_embeddings": positive_embeddings, "masks": masks}}


def preprocess_source_image(pipe, image: Image.Image, width: int, height: int,
                            dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((width, height), Image.BICUBIC)
    tensor = pipe.image_processor.preprocess(resized)
    return tensor.to(device=device, dtype=dtype)


def select_sdedit_timesteps(scheduler, num_inference_steps: int, strength: float,
                            noise_timestep: int | None, device: torch.device) -> tuple[torch.Tensor, int, int]:
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    if noise_timestep is not None:
        start_index = int((timesteps.float() - float(noise_timestep)).abs().argmin().item())
    else:
        clipped = max(0.0, min(1.0, float(strength)))
        init_timestep = min(max(int(num_inference_steps * clipped), 1), num_inference_steps)
        start_index = max(num_inference_steps - init_timestep, 0)
    selected = timesteps[start_index:]
    return selected, start_index, int(selected[0].item())


def scheduler_step_kwargs(pipe, generator: torch.Generator, eta: float = 0.0) -> dict:
    parameters = set(inspect.signature(pipe.scheduler.step).parameters)
    kwargs = {}
    if "eta" in parameters:
        kwargs["eta"] = eta
    if "generator" in parameters:
        kwargs["generator"] = generator
    return kwargs


@torch.no_grad()
def run_gligen_sdedit(
    pipe,
    source_image: Image.Image,
    prompt: str,
    negative_prompt: str | None,
    phrases: list[str],
    boxes: list[list[float]],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[Image.Image, dict]:
    device = torch.device(args.device)
    do_cfg = args.guidance_scale > 1.0

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    image_tensor = preprocess_source_image(pipe, source_image, args.width, args.height, pipe.vae.dtype, device)
    latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
    latents = (latents * pipe.vae.config.scaling_factor).to(dtype=prompt_embeds.dtype)
    timesteps, start_index, actual_noise_timestep = select_sdedit_timesteps(
        pipe.scheduler, args.num_inference_steps, args.strength, args.noise_timestep, device
    )
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])

    cross_attention_kwargs = prepare_gligen_grounding(pipe, phrases, boxes, do_cfg, device)
    num_grounding_steps = int(args.gligen_scheduled_sampling_beta * len(timesteps))
    pipe.enable_fuser(True)
    extra_step_kwargs = scheduler_step_kwargs(pipe, generator)

    for step_idx, timestep in enumerate(timesteps):
        if step_idx == num_grounding_steps:
            pipe.enable_fuser(False)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]
    pipe.enable_fuser(False)
    return image, {
        "strength": float(args.strength),
        "requested_noise_timestep": args.noise_timestep,
        "actual_noise_timestep": actual_noise_timestep,
        "start_timestep_index": start_index,
        "denoise_steps": len(timesteps),
    }


def draw_layout(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    palette = [(255, 64, 64), (40, 160, 90), (65, 105, 225), (240, 160, 30), (155, 80, 200)]
    line = max(2, int(round(max(out.size) / 300)))
    for idx, item in enumerate(items):
        color = palette[idx % len(palette)]
        x, y, w, h = [float(v) for v in item["bbox"]]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line)
        draw.text((x + 4, max(0, y - 18)), f"{item['class_name']} {item['layout_source']}", fill=color)
    return out


def draw_result_layout(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    palette = [(255, 64, 64), (40, 160, 90), (65, 105, 225), (240, 160, 30), (155, 80, 200)]
    width, height = out.size
    line = max(2, int(round(max(out.size) / 300)))
    for idx, item in enumerate(items):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = [float(v) for v in item["gligen_box"]]
        left, top, right, bottom = x1 * width, y1 * height, x2 * width, y2 * height
        draw.rectangle([left, top, right, bottom], outline=color, width=line)
        draw.text((left + 4, max(0, top - 18)), f"{item['class_name']} {item['phrase_source']}", fill=color)
    return out


def write_visual_index(output: Path, manifest: list[dict]) -> None:
    cards = []
    for item in manifest:
        cards.append(
            f"""
<article>
  <h2>{item['file_name']} | image_id={item['image_id']}</h2>
  <p><b>prompt:</b> {item['global_prompt']}</p>
  <div class="media">
    <figure><figcaption>Original layout</figcaption><img src="{item.get('original_overlay_file', '')}"></figure>
    <figure><figcaption>Result</figcaption><img src="{item.get('result_file', '')}"></figure>
    <figure><figcaption>Result + bbox</figcaption><img src="{item.get('result_overlay_file', '')}"></figure>
  </div>
</article>
"""
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GLIGEN SDEdit Synthetic Generation</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #eef2f7; color: #1d2733; }}
header {{ position: sticky; top: 0; background: white; padding: 14px 18px; border-bottom: 1px solid #d7dee8; }}
main {{ max-width: 1400px; margin: 0 auto; padding: 18px; display: grid; gap: 16px; }}
article {{ background: white; border: 1px solid #d7dee8; border-radius: 8px; padding: 14px; }}
h1, h2 {{ margin: 0 0 8px; }}
p {{ margin: 0 0 10px; font-size: 13px; }}
.media {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 12px; color: #667085; margin-bottom: 5px; }}
img {{ width: 100%; border: 1px solid #d7dee8; border-radius: 6px; display: block; }}
</style></head><body>
<header><h1>GLIGEN SDEdit Synthetic Generation</h1><div>items={len(manifest)}</div></header>
<main>{''.join(cards)}</main></body></html>
"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if args.clean_output and args.resume:
        raise ValueError("--clean-output and --resume cannot be used together")
    if args.clean_output:
        for child in ("images", "labels", "visuals"):
            if (output / child).exists():
                shutil.rmtree(output / child)
        for child in ("manifest.json", "selected_images.json", "index.html"):
            (output / child).unlink(missing_ok=True)
    logger = setup_logger(output)
    images_out = output / "images"
    labels_out = output / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if args.save_visualization:
        for child in ("original_overlays", "result_overlays"):
            (output / "visuals" / child).mkdir(parents=True, exist_ok=True)

    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    edit_dataset = CocoMini(
        args.annotations,
        args.images,
        class_subset=class_subset,
        dataset_yaml=args.dataset_yaml,
    )
    full_dataset = CocoMini(
        args.annotations,
        args.images,
        class_subset=None,
        dataset_yaml=args.dataset_yaml,
    )
    meta, embeddings = load_object_prompts(Path(args.tokens_dir))
    prompt_class_names = meta.get("class_names")
    if prompt_class_names is not None and list(prompt_class_names) != full_dataset.class_names:
        raise ValueError(
            "Prompt checkpoint class names do not match the current dataset mapping: "
            f"prompts={list(prompt_class_names)} dataset={full_dataset.class_names}"
        )
    prompt_scope = args.prompt_scope or meta["prompt_scope"]
    include_class_name = bool(meta.get("include_class_name", False)) if args.include_class_name is None else bool(args.include_class_name)
    logger.info("GLIGEN SDEdit generation start: class_subset=%s candidates=%d tokens_dir=%s checkpoint=%s",
                subset_label(class_subset), len(edit_dataset.images), args.tokens_dir, meta.get("latest"))
    logger.info("generation params: max_images=%d max_objects=%d size=%dx%d strength=%.3f noise_timestep=%s steps=%d beta=%.3f guidance=%.3f filter=%s min_ratio=%.8f",
                args.max_images, args.max_objects_per_image, args.width, args.height, args.strength,
                args.noise_timestep, args.num_inference_steps, args.gligen_scheduled_sampling_beta,
                args.guidance_scale, args.enable_generation_filter, args.min_bbox_area_ratio)

    from diffusers import StableDiffusionGLIGENPipeline

    load_model_path, load_variant = prepare_gligen_model_for_diffusers(
        args.pretrained_model_name_or_path, output, args.variant, logger
    )
    load_kwargs = {}
    if load_variant:
        load_kwargs["variant"] = load_variant
    weight_dtype = torch.float32 if str(args.device).lower() == "cpu" else torch.float16
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        load_model_path,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
        **load_kwargs,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    install_embeddings(pipe.tokenizer, pipe.text_encoder, meta["groups"], embeddings)

    manifest_path = output / "manifest.json"
    selected_images_path = output / "selected_images.json"
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            raise ValueError(f"Resume manifest must be a list: {manifest_path}")
    else:
        manifest = []
    completed_image_ids = {int(item["image_id"]) for item in manifest}
    selected_images = [
        str((output / item["result_file"]).resolve())
        for item in manifest
        if item.get("result_file")
    ]
    candidates = [(image_id, path, anns) for image_id, path, anns in edit_dataset.iter_images(None) if anns]
    rng = random.Random(args.seed)
    selected = list(candidates)
    rng.shuffle(selected)
    skipped = 0
    update_resume_state(
        Path(args.state_json) if args.state_json else None,
        len(manifest),
        args.max_images,
        len(manifest) >= args.max_images,
    )
    for idx, (image_id, path, edit_anns) in enumerate(tqdm(selected, desc="gligen sdedit generation"), start=1):
        if len(manifest) >= args.max_images:
            break
        if int(image_id) in completed_image_ids:
            continue
        info = full_dataset.images[int(image_id)]
        full_anns = full_dataset.annotations.get(int(image_id), [])
        phrases, boxes, layout_items, filtered = prepare_layout_items(
            meta,
            full_dataset,
            int(image_id),
            full_anns,
            edit_anns,
            prompt_scope,
            include_class_name,
            args.max_objects_per_image,
            args.enable_generation_filter,
            args.min_bbox_area_ratio,
        )
        if not phrases:
            skipped += 1
            continue
        source = Image.open(path).convert("RGB")
        global_prompt = choose_global_prompt(phrases, layout_items, args)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + int(image_id))
        image, sdedit_info = run_gligen_sdedit(
            pipe=pipe,
            source_image=source,
            prompt=global_prompt,
            negative_prompt=args.negative_prompt,
            phrases=phrases,
            boxes=boxes,
            args=args,
            generator=generator,
        )

        name = f"hard_{int(image_id):012d}.jpg"
        result_file = images_out / name
        label_file = labels_out / Path(name).with_suffix(".txt")
        image.save(result_file, quality=95)
        write_layout_label_file(label_file, layout_items)
        selected_images.append(str(result_file.resolve()))

        item = {
            "image_id": int(image_id),
            "file_name": name,
            "source_file": str(path),
            "generation_mode": "gligen_sdedit",
            "global_prompt": global_prompt,
            "global_prompt_source": args.global_prompt_source,
            "gligen_phrases": phrases,
            "gligen_boxes": boxes,
            "layout_objects": layout_items,
            "filtered_layout_objects": filtered,
            "edited_objects": len(layout_items),
            "edit_annotations": len(edit_anns),
            "selected_edit_annotations": sum(1 for x in layout_items if x["layout_source"] == "target_gt"),
            "filtered_edit_annotations": sum(1 for x in filtered if x["layout_source"] == "target_gt"),
            "generation_filter_enabled": bool(args.enable_generation_filter),
            "min_bbox_area_ratio": args.min_bbox_area_ratio if args.enable_generation_filter else None,
            "original_gt_annotations": len(full_anns),
            "label_annotations": len(layout_items),
            "labels_inherited_from_real": False,
            "labels_scope": "gligen_layout_boxes",
            "dataset_yaml": str(full_dataset.dataset_yaml) if full_dataset.dataset_yaml is not None else None,
            "class_mapping": full_dataset.class_mapping if full_dataset.dataset_yaml is not None else None,
            "output_size": [args.width, args.height],
            "original_size": [int(info["width"]), int(info["height"])],
            "sdedit": sdedit_info,
            "result_file": str(result_file.relative_to(output)),
        }
        if args.save_visualization:
            original_overlay = output / "visuals" / "original_overlays" / name
            result_overlay = output / "visuals" / "result_overlays" / name
            draw_layout(source, layout_items).save(original_overlay, quality=95)
            draw_result_layout(image, layout_items).save(result_overlay, quality=95)
            item["original_overlay_file"] = str(original_overlay.relative_to(output))
            item["result_overlay_file"] = str(result_overlay.relative_to(output))
        manifest.append(item)
        completed_image_ids.add(int(image_id))
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(selected_images_path, selected_images)
        update_resume_state(
            Path(args.state_json) if args.state_json else None,
            len(manifest),
            args.max_images,
            False,
        )
        if len(manifest) == 1 or len(manifest) % 25 == 0 or idx == len(selected):
            logger.info("generation progress: processed=%d/%d synthetic=%d skipped=%d latest=%s objects=%d",
                        idx, len(selected), len(manifest), skipped, name, len(layout_items))

    write_json_atomic(manifest_path, manifest)
    write_json_atomic(selected_images_path, selected_images)
    update_resume_state(
        Path(args.state_json) if args.state_json else None,
        len(manifest),
        args.max_images,
        len(manifest) >= args.max_images,
    )
    if args.save_visualization:
        write_visual_index(output, manifest)
    logger.info(
        "GLIGEN SDEdit generation complete: target=%d synthetic=%d skipped=%d target_reached=%s output=%s",
        args.max_images,
        len(manifest),
        skipped,
        len(manifest) >= args.max_images,
        output,
    )


if __name__ == "__main__":
    main()
