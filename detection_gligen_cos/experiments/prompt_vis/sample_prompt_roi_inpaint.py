from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from detection.detector import YoloV7Detector
from detection_gligen_sdedit.common import (
    CocoMini,
    bbox_mask,
    load_depth_style_components,
    load_failures,
    parse_class_subset,
    patch_huggingface_hub_for_diffusers_024,
    setup_logger,
    subset_label,
)
from detection_gligen_sdedit.experiments.prompt_vis.utils import (
    clean_output_dir,
    crop_with_padding,
    draw_bbox_mask_overlay,
    load_object_prompts_checkpoint,
    matched_detector_score,
    prompt_for_failure,
    rel,
    write_html,
)
from detection_gligen_sdedit.generate_hard_examples import install_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI-level inpainting visualization for learned object prompts.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="learned_embeds-*.bin. Defaults to object_prompts.json latest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-scope", choices=["class", "class_failure"], default=None)
    parser.add_argument("--include-class-name", action="store_true", default=None)
    parser.add_argument("--no-include-class-name", action="store_false", dest="include_class_name")
    parser.add_argument("--max-instances", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--mask-padding", type=float, default=0.08)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark, no object, missing object")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--yolov7", default=None, help="Optional YOLOv7 repo for before/after detector scores.")
    parser.add_argument("--weights", default=None, help="Optional YOLOv7 weights for before/after detector scores.")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.001)
    parser.add_argument("--detector-match-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if args.clean_output:
        clean_output_dir(output)
    for child in ("originals", "masks", "results", "crops"):
        (output / child).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)

    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    dataset = CocoMini(args.annotations, args.images, class_subset=None)
    failures = load_failures(args.failures, class_subset=class_subset)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(failures)
    failures = failures[:args.max_instances]
    logger.info("ROI prompt visualization start: failures=%d class_subset=%s", len(failures), subset_label(class_subset))

    meta, ckpt_name, embeddings = load_object_prompts_checkpoint(Path(args.tokens_dir), args.checkpoint)
    prompt_scope = args.prompt_scope or meta["prompt_scope"]
    logger.info("object prompts loaded: tokens_dir=%s checkpoint=%s groups=%d scope=%s",
                args.tokens_dir, ckpt_name, len(meta["groups"]), prompt_scope)

    patch_huggingface_hub_for_diffusers_024()
    tokenizer, text_encoder, vae, unet, scheduler = load_depth_style_components(
        args.pretrained_model_name_or_path,
        dtype=torch.float16,
        local_files_only=args.local_files_only,
        text_encoder_dtype=torch.float16,
    )
    install_embeddings(tokenizer, text_encoder, meta["groups"], embeddings)

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

    detector = None
    if args.yolov7 and args.weights:
        logger.info("loading detector for scores: yolov7=%s weights=%s", args.yolov7, args.weights)
        detector = YoloV7Detector(args.yolov7, args.weights, args.detector_device, args.detector_image_size)

    manifest = []
    skipped = 0
    for idx, failure in enumerate(tqdm(failures, desc="roi inpaint"), start=1):
        image_id = int(failure["image_id"])
        class_id = int(failure["class_id"])
        group, prompt = prompt_for_failure(meta, dataset, failure, prompt_scope, args.include_class_name)
        if group is None or prompt is None:
            skipped += 1
            continue
        image = Image.open(dataset.image_path(image_id)).convert("RGB")
        mask = bbox_mask(image.size, failure["bbox"], args.mask_padding)
        label = f"{dataset.class_names[class_id]} | {failure.get('failure_type', '')}"
        overlay = draw_bbox_mask_overlay(image, mask, failure["bbox"], label)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx + image_id)

        result = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            mask_image=mask,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0].resize(image.size)

        stem = f"{idx:04d}_img{image_id}_cls{class_id}_{failure.get('failure_type', 'failure')}"
        original_file = output / "originals" / f"{stem}.jpg"
        mask_file = output / "masks" / f"{stem}.jpg"
        result_file = output / "results" / f"{stem}.jpg"
        crop_file = output / "crops" / f"{stem}.jpg"
        image.save(original_file, quality=95)
        overlay.save(mask_file, quality=95)
        result.save(result_file, quality=95)
        crop_with_padding(result, failure["bbox"]).save(crop_file, quality=95)

        detector_payload = {}
        if detector is not None:
            original_predictions = detector.predict(image, conf=args.detector_conf)
            result_predictions = detector.predict(result, conf=args.detector_conf)
            detector_payload = {
                "original": matched_detector_score(original_predictions, failure["bbox"], class_id, args.detector_match_iou),
                "generated": matched_detector_score(result_predictions, failure["bbox"], class_id, args.detector_match_iou),
            }
        else:
            detector_payload = {
                "original": {"confidence": failure.get("confidence"), "iou": None, "class_id": class_id},
                "generated": {"confidence": None, "iou": None, "class_id": None},
            }

        manifest.append({
            "index": idx,
            "image_id": image_id,
            "annotation_id": failure.get("annotation_id"),
            "bbox": failure["bbox"],
            "class_id": class_id,
            "class_name": dataset.class_names[class_id],
            "failure_type": failure.get("failure_type"),
            "group": group,
            "prompt": prompt,
            "checkpoint": ckpt_name,
            "detector": detector_payload,
            "original_file": rel(original_file, output),
            "mask_file": rel(mask_file, output),
            "result_file": rel(result_file, output),
            "crop_file": rel(crop_file, output),
        })
        if idx == 1 or idx % 25 == 0 or idx == len(failures):
            logger.info("ROI progress: processed=%d/%d rendered=%d skipped=%d", idx, len(failures), len(manifest), skipped)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(output / "index.html", "ROI Inpainting Prompt Visualization", manifest, "roi_inpaint")
    logger.info("ROI prompt visualization complete: rendered=%d skipped=%d html=%s", len(manifest), skipped, output / "index.html")


if __name__ == "__main__":
    main()
