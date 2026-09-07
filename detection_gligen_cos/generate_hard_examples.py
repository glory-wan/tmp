from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import random
import torch
from PIL import Image
from tqdm.auto import tqdm

from .common import CocoMini, bbox_mask, load_depth_style_components, parse_class_subset, setup_logger, subset_label


def load_object_prompts(tokens_dir: Path):
    meta = json.loads((tokens_dir / "object_prompts.json").read_text(encoding="utf-8"))
    embeddings = torch.load(tokens_dir / meta["latest"], map_location="cpu")
    return meta, embeddings


def install_embeddings(tokenizer, text_encoder, groups, embeddings):
    tokens = [token for values in groups.values() for token in values]
    added = tokenizer.add_tokens(tokens)
    if added != len(tokens):
        raise ValueError("Token collision while installing learned object prompts.")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token in tokens:
        token_embeds[tokenizer.convert_tokens_to_ids(token)] = embeddings[token].to(dtype=token_embeds.dtype, device=token_embeds.device)


def write_label_file(path: Path, anns: list[dict], info: dict):
    lines = []
    width, height = float(info["width"]), float(info["height"])
    for ann in anns:
        x, y, w, h = map(float, ann["bbox"])
        lines.append(f"{ann['class_id']} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def choose_group(groups: dict, class_id: int, prompt_scope: str) -> str | None:
    if prompt_scope == "class":
        group = f"class_{class_id}"
        return group if group in groups else None
    prefix = f"class_{class_id}_"
    matches = sorted(group for group in groups if group.startswith(prefix))
    return matches[0] if matches else None


def build_object_prompt(class_name: str, tokens: list[str], include_class_name: bool) -> str:
    token_text = ",".join(tokens)
    if include_class_name:
        return f"{class_name}, {token_text}, photo, highly detailed, photorealistic"
    return f"{token_text}, photo, highly detailed, photorealistic"


def bbox_area_ratio(ann: dict, info: dict) -> float:
    image_area = max(float(info["width"]) * float(info["height"]), 1.0)
    _, _, width, height = map(float, ann["bbox"])
    return max(width, 0.0) * max(height, 0.0) / image_area


def filter_generation_annotations(anns: list[dict], info: dict, min_bbox_area_ratio: float) -> tuple[list[dict], list[dict]]:
    selected = []
    filtered = []
    for ann in anns:
        ratio = bbox_area_ratio(ann, info)
        if ratio >= min_bbox_area_ratio:
            selected.append(ann)
        else:
            filtered.append(ann)
    return selected, filtered


def manifest_instance(ann: dict, info: dict) -> dict:
    return {
        "annotation_id": ann.get("id"),
        "class_id": int(ann["class_id"]),
        "bbox": ann["bbox"],
        "bbox_area_ratio": bbox_area_ratio(ann, info),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate bbox-local hard examples with frozen object prompts.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-scope", choices=["class", "class_failure"], default=None)
    parser.add_argument("--include-class-name", action="store_true", default=None,
                        help="Prepend the explicit COCO class name to the learned-token object prompt.")
    parser.add_argument("--no-include-class-name", action="store_false", dest="include_class_name",
                        help="Do not prepend the explicit COCO class name, even if prompt metadata used it.")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-objects-per-image", type=int, default=3)
    parser.add_argument("--mask-padding", type=float, default=0.08)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--enable-generation-filter", action="store_true",
                        help="Only edit instances whose bbox area ratio passes --min-bbox-area-ratio.")
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.0,
                        help="Minimum bbox area divided by image area for selective generation editing.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark")
    parser.add_argument("--clean-output", action="store_true", help="Remove existing images/labels/manifest in output-dir before generation.")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if args.clean_output:
        for child in (output / "images", output / "labels"):
            if child.exists():
                shutil.rmtree(child)
        for child in (output / "manifest.json", output / "selected_images.json"):
            child.unlink(missing_ok=True)
    logger = setup_logger(output)
    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    edit_dataset = CocoMini(args.annotations, args.images, class_subset=class_subset)
    full_dataset = CocoMini(args.annotations, args.images, class_subset=None)
    logger.info("edit dataset loaded: images=%d annotations=%d classes=%d",
                len(edit_dataset.images), sum(len(v) for v in edit_dataset.annotations.values()), len(edit_dataset.class_names))
    logger.info("full-label dataset loaded: images=%d annotations=%d classes=%d",
                len(full_dataset.images), sum(len(v) for v in full_dataset.annotations.values()), len(full_dataset.class_names))
    logger.info("class subset: %s", subset_label(class_subset))
    meta, embeddings = load_object_prompts(Path(args.tokens_dir))
    prompt_scope = args.prompt_scope or meta["prompt_scope"]
    include_class_name = bool(meta.get("include_class_name", False)) if args.include_class_name is None else bool(args.include_class_name)
    logger.info("object prompts loaded: tokens_dir=%s groups=%d latest=%s scope=%s",
                args.tokens_dir, len(meta["groups"]), meta["latest"], prompt_scope)
    logger.info("prompt format: include_class_name=%s template='%s'",
                include_class_name,
                "{class_name}, {tokens}, photo, highly detailed, photorealistic" if include_class_name
                else "{tokens}, photo, highly detailed, photorealistic")
    logger.info("loading SD components: model=%s local_files_only=%s", args.pretrained_model_name_or_path, args.local_files_only)
    tokenizer, text_encoder, vae, unet, scheduler = load_depth_style_components(
        args.pretrained_model_name_or_path,
        dtype=torch.float16,
        local_files_only=args.local_files_only,
        text_encoder_dtype=torch.float16,
    )
    install_embeddings(tokenizer, text_encoder, meta["groups"], embeddings)
    logger.info("learned embeddings installed: tokens=%d", sum(len(v) for v in meta["groups"].values()))
    logger.info(
        "pipeline dtype check: text_encoder=%s vae=%s unet=%s",
        text_encoder.get_input_embeddings().weight.dtype,
        next(vae.parameters()).dtype,
        next(unet.parameters()).dtype,
    )

    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
    ).to(args.device)


    
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    manifest = []
    # candidates = [(image_id, path, anns) for image_id, path, anns in dataset.iter_images(None) if anns]
    # logger.info("generation start: candidates=%d max_images=%d max_objects_per_image=%d inference_steps=%d",
    #             len(candidates), args.max_images, args.max_objects_per_image, args.num_inference_steps)
    # skipped = 0
    # selected = candidates[:args.max_images]
    candidates = [
    (image_id, path, anns)
    for image_id, path, anns in edit_dataset.iter_images(None)
    if anns
    ]

    rng = random.Random(args.seed)

    select_num = min(args.max_images, len(candidates))
    selected = rng.sample(candidates, select_num)

    logger.info(
    "generation start: candidates=%d selected=%d max_objects_per_image=%d "
    "inference_steps=%d seed=%d class_subset=%s generation_filter=%s min_bbox_area_ratio=%.8f",
    len(candidates),
    len(selected),
    args.max_objects_per_image,
    args.num_inference_steps,
    args.seed,
    subset_label(class_subset),
    args.enable_generation_filter,
    args.min_bbox_area_ratio,
)

    skipped = 0

    



    for idx, (image_id, path, anns) in enumerate(tqdm(selected, desc="bbox generation"), start=1):
        image = Image.open(path).convert("RGB")
        image_info = edit_dataset.images[image_id]
        if args.enable_generation_filter:
            edit_anns, filtered_anns = filter_generation_annotations(anns, image_info, args.min_bbox_area_ratio)
        else:
            edit_anns, filtered_anns = list(anns), []
        edit_anns = edit_anns[:args.max_objects_per_image]
        edited = 0
        edited_instances = []
        for ann in edit_anns:
            group = choose_group(meta["groups"], ann["class_id"], prompt_scope)
            if group not in meta["groups"]:
                continue
            class_name = full_dataset.class_names[int(ann["class_id"])]
            prompt = build_object_prompt(class_name, meta["groups"][group], include_class_name)
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                image=image,
                mask_image=bbox_mask(image.size, ann["bbox"], args.mask_padding),
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0].resize(image.size)
            edited += 1
            edited_instances.append(manifest_instance(ann, image_info))
        if edited == 0:
            skipped += 1
            if idx == 1 or idx % 25 == 0 or idx == len(selected):
                logger.info("generation progress: processed=%d/%d synthetic=%d skipped=%d latest_skipped_image=%s",
                            idx, len(selected), len(manifest), skipped, image_id)
            continue
        name = f"hard_{image_id:012d}.jpg"
        image.save(output / "images" / name, quality=95)
        full_anns = full_dataset.annotations.get(image_id, [])
        write_label_file(output / "labels" / Path(name).with_suffix(".txt"), full_anns, full_dataset.images[image_id])
        manifest.append({
            "image_id": image_id,
            "file_name": name,
            "edited_objects": edited,
            "edit_annotations": len(anns),
            "selected_edit_annotations": len(edit_anns),
            "filtered_edit_annotations": len(filtered_anns),
            "generation_filter_enabled": bool(args.enable_generation_filter),
            "min_bbox_area_ratio": args.min_bbox_area_ratio if args.enable_generation_filter else None,
            "edited_instances": edited_instances,
            "filtered_instances": [manifest_instance(ann, image_info) for ann in filtered_anns],
            "inherited_annotations": len(full_anns),
            "labels_inherited_from_real": True,
            "labels_scope": "full_original_gt",
        })
        if len(manifest) == 1 or len(manifest) % 25 == 0 or idx == len(selected):
            logger.info("generation progress: processed=%d/%d synthetic=%d skipped=%d latest=%s edited=%d",
                        idx, len(selected), len(manifest), skipped, name, edited)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("generation complete: synthetic=%d skipped=%d output=%s manifest=%s",
                len(manifest), skipped, output, output / "manifest.json")


if __name__ == "__main__":
    main()
