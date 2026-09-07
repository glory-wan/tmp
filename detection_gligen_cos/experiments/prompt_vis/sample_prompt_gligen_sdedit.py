from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from detection_gligen_sdedit.common import CocoMini, load_failures, parse_class_subset, setup_logger, subset_label
from detection_gligen_sdedit.experiments.prompt_vis.sample_prompt_gligen_category_count import (
    bbox_area_ratio,
    choose_global_prompt,
    draw_gligen_layout,
    draw_layout,
    filter_small_gt_items,
    load_caption_map,
    ordered_failure_aware_gt,
    write_caption_html,
)
from detection_gligen_sdedit.experiments.prompt_vis.utils import (
    clean_output_dir,
    load_object_prompts_checkpoint,
    rel,
)
from detection_gligen_sdedit.generate_hard_examples import choose_group, install_embeddings
from detection_gligen_sdedit.generate_gligen_layout import build_global_prompt, prepare_gligen_model_for_diffusers, xywh_to_gligen_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDEdit-style GLIGEN visualization: original image latent + noise + GLIGEN box/text denoising."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--captions", default=None)
    parser.add_argument("--caption-selection", choices=["first", "random"], default="first")
    parser.add_argument("--images", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="learned_embeds-*.bin. Defaults to object_prompts.json latest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-scope", choices=["class", "class_failure"], default=None)
    parser.add_argument("--include-class-name", action="store_true", default=None)
    parser.add_argument("--no-include-class-name", action="store_false", dest="include_class_name")
    parser.add_argument("--max-scenes", type=int, default=32)
    parser.add_argument("--max-objects-per-scene", type=int, default=6)
    parser.add_argument(
        "--min-bbox-area-ratio",
        type=float,
        default=0.0,
        help="Filter out GT boxes whose bbox area divided by image area is below this value.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--gligen-scheduled-sampling-beta", type=float, default=0.3)
    parser.add_argument(
        "--global-prompt-source",
        choices=["template", "coco_caption", "category_count"],
        default="template",
        help="Global prompt source: template phrases, COCO caption, or category counts from final layout objects.",
    )
    parser.add_argument("--global-prompt-template", default="{phrases}, photo, highly detailed, photorealistic")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.65,
        help="SDEdit noise strength in [0, 1]. Larger values run more denoising steps from a noisier original latent.",
    )
    parser.add_argument(
        "--noise-timestep",
        type=int,
        default=None,
        help="Optional explicit scheduler timestep to add noise at. Overrides --strength by choosing the nearest inference timestep.",
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    return parser.parse_args()


def build_clean_object_phrase(class_name: str, tokens: list[str], include_class_name: bool) -> str:
    token_text = ",".join(tokens)
    if include_class_name:
        return f"{class_name}, {token_text}"
    return token_text


def phrase_for_gt_clean(meta: dict, dataset: CocoMini, ann: dict,
                        prompt_scope: str, include_class_name: bool | None) -> tuple[str, str | None, str]:
    class_id = int(ann["class_id"])
    class_name = dataset.class_names[class_id]
    group = choose_group(meta["groups"], class_id, prompt_scope)
    if group in meta["groups"]:
        use_class = bool(meta.get("include_class_name", False)) if include_class_name is None else bool(include_class_name)
        return build_clean_object_phrase(class_name, meta["groups"][group], use_class), group, "learned_prompt"
    return class_name, None, "class_name_fallback"


def prepare_gligen_grounding(pipe, gligen_phrases: list[str], gligen_boxes: list[list[float]],
                             batch_size: int, do_classifier_free_guidance: bool, device: torch.device) -> dict:
    max_objs = 30
    if len(gligen_boxes) > max_objs:
        gligen_phrases = gligen_phrases[:max_objs]
        gligen_boxes = gligen_boxes[:max_objs]

    tokenizer_inputs = pipe.tokenizer(gligen_phrases, padding=True, return_tensors="pt").to(device)
    text_embeddings_for_phrases = pipe.text_encoder(**tokenizer_inputs).pooler_output
    n_objs = len(gligen_boxes)

    boxes = torch.zeros(max_objs, 4, device=device, dtype=pipe.text_encoder.dtype)
    boxes[:n_objs] = torch.tensor(gligen_boxes, device=device, dtype=pipe.text_encoder.dtype)
    positive_embeddings = torch.zeros(
        max_objs, pipe.unet.config.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
    )
    positive_embeddings[:n_objs] = text_embeddings_for_phrases
    masks = torch.zeros(max_objs, device=device, dtype=pipe.text_encoder.dtype)
    masks[:n_objs] = 1

    boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1).clone()
    positive_embeddings = positive_embeddings.unsqueeze(0).expand(batch_size, -1, -1).clone()
    masks = masks.unsqueeze(0).expand(batch_size, -1).clone()
    if do_classifier_free_guidance:
        repeat_batch = batch_size * 2
        boxes = torch.cat([boxes] * 2)
        positive_embeddings = torch.cat([positive_embeddings] * 2)
        masks = torch.cat([masks] * 2)
        masks[: repeat_batch // 2] = 0

    return {"gligen": {"boxes": boxes, "positive_embeddings": positive_embeddings, "masks": masks}}


def preprocess_source_image(pipe, image: Image.Image, width: int, height: int, dtype: torch.dtype,
                            device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((width, height), Image.BICUBIC)
    tensor = pipe.image_processor.preprocess(resized)
    return tensor.to(device=device, dtype=dtype)


def encode_image_latents(pipe, image_tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
    return latents * pipe.vae.config.scaling_factor


def select_sdedit_timesteps(scheduler, num_inference_steps: int, strength: float,
                            noise_timestep: int | None, device: torch.device) -> tuple[torch.Tensor, int, int]:
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    if noise_timestep is not None:
        distances = (timesteps.float() - float(noise_timestep)).abs()
        start_index = int(distances.argmin().item())
    else:
        clipped_strength = max(0.0, min(1.0, float(strength)))
        init_timestep = min(max(int(num_inference_steps * clipped_strength), 1), num_inference_steps)
        start_index = max(num_inference_steps - init_timestep, 0)
    selected = timesteps[start_index:]
    return selected, start_index, int(selected[0].item())


@torch.no_grad()
def run_gligen_sdedit(
    pipe,
    source_image: Image.Image,
    prompt: str,
    negative_prompt: str | None,
    gligen_phrases: list[str],
    gligen_boxes: list[list[float]],
    width: int,
    height: int,
    num_inference_steps: int,
    strength: float,
    noise_timestep: int | None,
    guidance_scale: float,
    gligen_scheduled_sampling_beta: float,
    eta: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Image.Image, dict]:
    do_classifier_free_guidance = guidance_scale > 1.0
    batch_size = 1

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    image_tensor = preprocess_source_image(pipe, source_image, width, height, pipe.vae.dtype, device)
    latents = encode_image_latents(pipe, image_tensor, generator).to(dtype=prompt_embeds.dtype)
    timesteps, start_index, actual_noise_timestep = select_sdedit_timesteps(
        pipe.scheduler, num_inference_steps, strength, noise_timestep, device
    )
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    latents = pipe.scheduler.add_noise(latents, noise, timesteps[:1])

    cross_attention_kwargs = prepare_gligen_grounding(
        pipe, gligen_phrases, gligen_boxes, batch_size, do_classifier_free_guidance, device
    )
    num_grounding_steps = int(gligen_scheduled_sampling_beta * len(timesteps))
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    pipe.enable_fuser(True)

    for step_index, timestep in enumerate(timesteps):
        if step_index == num_grounding_steps:
            pipe.enable_fuser(False)

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]
    pipe.enable_fuser(False)
    return image, {
        "strength": float(strength),
        "requested_noise_timestep": noise_timestep,
        "actual_noise_timestep": actual_noise_timestep,
        "start_timestep_index": start_index,
        "denoise_steps": len(timesteps),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if args.clean_output:
        clean_output_dir(output)
    for child in ("originals", "masks", "results", "result_overlays"):
        (output / child).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)

    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    dataset = CocoMini(args.annotations, args.images, class_subset=None)
    captions_by_image = load_caption_map(args.captions)
    failures = load_failures(args.failures, class_subset=class_subset)
    by_image: dict[int, list[dict]] = defaultdict(list)
    for failure in failures:
        by_image[int(failure["image_id"])].append(failure)
    scenes = sorted(by_image.items())
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(scenes)
    scenes = scenes[:args.max_scenes]
    logger.info("GLIGEN SDEdit visualization start: scenes=%d failures=%d class_subset=%s",
                len(scenes), len(failures), subset_label(class_subset))
    logger.info("SDEdit settings: strength=%.3f noise_timestep=%s steps=%d",
                args.strength, args.noise_timestep, args.num_inference_steps)

    meta, ckpt_name, embeddings = load_object_prompts_checkpoint(Path(args.tokens_dir), args.checkpoint)
    prompt_scope = args.prompt_scope or meta["prompt_scope"]
    logger.info("object prompts loaded: tokens_dir=%s checkpoint=%s groups=%d scope=%s",
                args.tokens_dir, ckpt_name, len(meta["groups"]), prompt_scope)

    from diffusers import StableDiffusionGLIGENPipeline

    load_model_path, load_variant = prepare_gligen_model_for_diffusers(
        args.pretrained_model_name_or_path, output, args.variant, logger
    )
    load_kwargs = {}
    if load_variant:
        load_kwargs["variant"] = load_variant
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        load_model_path,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
        **load_kwargs,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    install_embeddings(pipe.tokenizer, pipe.text_encoder, meta["groups"], embeddings)

    device = torch.device(args.device)
    manifest = []
    skipped = 0
    for scene_idx, (image_id, failure_items) in enumerate(tqdm(scenes, desc="gligen sdedit scenes"), start=1):
        info = dataset.images[int(image_id)]
        phrases: list[str] = []
        boxes: list[list[float]] = []
        layout_items: list[dict] = []
        gt_items = ordered_failure_aware_gt(dataset.annotations.get(int(image_id), []), failure_items)
        gt_items, filtered_small_gt = filter_small_gt_items(gt_items, info, args.min_bbox_area_ratio)
        for ann, failure in gt_items:
            if len(phrases) >= args.max_objects_per_scene:
                break
            class_id = int(ann["class_id"])
            ratio = bbox_area_ratio(ann, info)
            prompt_for_box, group, phrase_source = phrase_for_gt_clean(meta, dataset, ann, prompt_scope, args.include_class_name)
            box = xywh_to_gligen_box(ann["bbox"], float(info["width"]), float(info["height"]))
            phrases.append(prompt_for_box)
            boxes.append(box)
            layout_items.append({
                "image_id": int(image_id),
                "annotation_id": ann.get("id"),
                "bbox": ann["bbox"],
                "bbox_area_ratio": ratio,
                "gligen_box": box,
                "class_id": class_id,
                "class_name": dataset.class_names[class_id],
                "failure_type": failure.get("failure_type") if failure else None,
                "failure_confidence": failure.get("confidence") if failure else None,
                "failure_top2_gap": failure.get("top2_gap") if failure else None,
                "layout_source": "failure" if failure else "normal_gt",
                "group": group,
                "phrase_source": phrase_source,
                "phrase": prompt_for_box,
            })
        if not phrases:
            skipped += 1
            continue

        prompt, global_prompt_source, coco_caption, category_count_prompt = choose_global_prompt(
            phrases, layout_items, int(image_id), args, captions_by_image
        )
        source = Image.open(dataset.image_path(image_id)).convert("RGB")
        generator = torch.Generator(device=args.device).manual_seed(args.seed + int(image_id))
        image, sdedit_info = run_gligen_sdedit(
            pipe=pipe,
            source_image=source,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            gligen_phrases=phrases,
            gligen_boxes=boxes,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            noise_timestep=args.noise_timestep,
            guidance_scale=args.guidance_scale,
            gligen_scheduled_sampling_beta=args.gligen_scheduled_sampling_beta,
            eta=args.eta,
            generator=generator,
            device=device,
        )

        layout = draw_layout(source, layout_items)
        result_overlay = draw_gligen_layout(image, layout_items)
        stem = f"{scene_idx:04d}_img{image_id}_objects{len(layout_items)}"
        original_file = output / "originals" / f"{stem}.jpg"
        mask_file = output / "masks" / f"{stem}.jpg"
        result_file = output / "results" / f"{stem}.jpg"
        result_overlay_file = output / "result_overlays" / f"{stem}.jpg"
        source.save(original_file, quality=95)
        layout.save(mask_file, quality=95)
        image.save(result_file, quality=95)
        result_overlay.save(result_overlay_file, quality=95)

        class_names = ", ".join(sorted({x["class_name"] for x in layout_items}))
        failure_layout_count = sum(1 for x in layout_items if x["layout_source"] == "failure")
        normal_layout_count = sum(1 for x in layout_items if x["layout_source"] == "normal_gt")
        manifest.append({
            "index": scene_idx,
            "mode": "gligen_sdedit",
            "image_id": int(image_id),
            "annotation_id": ",".join(str(x.get("annotation_id")) for x in layout_items),
            "bbox": [x["bbox"] for x in layout_items],
            "class_id": "multi",
            "class_name": class_names,
            "failure_type": "multi",
            "group": ",".join(sorted({str(x["group"]) for x in layout_items if x.get("group")})),
            "prompt": prompt,
            "global_prompt": prompt,
            "global_prompt_source": global_prompt_source,
            "coco_caption": coco_caption,
            "caption_selection": args.caption_selection if args.captions else None,
            "template_global_prompt": build_global_prompt(phrases, args.global_prompt_template),
            "category_count_prompt": category_count_prompt,
            "requested_global_prompt_source": args.global_prompt_source,
            "gligen_phrases": phrases,
            "gligen_boxes": boxes,
            "gt_annotations_total": len(dataset.annotations.get(int(image_id), [])),
            "failure_annotations_total": len(failure_items),
            "min_bbox_area_ratio": args.min_bbox_area_ratio,
            "filtered_small_bbox_count": len(filtered_small_gt),
            "filtered_small_bbox_items": filtered_small_gt,
            "failure_layout_count": failure_layout_count,
            "normal_layout_count": normal_layout_count,
            "checkpoint": ckpt_name,
            "layout_objects": layout_items,
            "sdedit": sdedit_info,
            "original_file": rel(original_file, output),
            "mask_file": rel(mask_file, output),
            "result_file": rel(result_file, output),
            "result_overlay_file": rel(result_overlay_file, output),
        })
        if scene_idx == 1 or scene_idx % 10 == 0 or scene_idx == len(scenes):
            logger.info("GLIGEN SDEdit progress: processed=%d/%d rendered=%d skipped=%d",
                        scene_idx, len(scenes), len(manifest), skipped)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_caption_html(output / "index.html", "GLIGEN SDEdit Layout Prompt Visualization", manifest)
    logger.info("GLIGEN SDEdit visualization complete: rendered=%d skipped=%d html=%s",
                len(manifest), skipped, output / "index.html")


if __name__ == "__main__":
    main()
