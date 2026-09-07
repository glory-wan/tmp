# 最开始的GLIGEN版本，bbox只有一个，全局prompt是通用的
# "{phrases}, photo, highly detailed, photorealistic"

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from detection_gligen_sdedit.common import CocoMini, load_failures, parse_class_subset, setup_logger, subset_label
from detection_gligen_sdedit.experiments.prompt_vis.utils import (
    clean_output_dir,
    load_object_prompts_checkpoint,
    rel,
    write_html,
)
from detection_gligen_sdedit.generate_hard_examples import choose_group, install_embeddings
from detection_gligen_sdedit.generate_gligen_layout import build_global_prompt, prepare_gligen_model_for_diffusers, xywh_to_gligen_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLIGEN layout-conditioned visualization for learned object prompts.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--variant", default=None)
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
    parser.add_argument("--max-scenes", type=int, default=32)
    parser.add_argument("--max-objects-per-scene", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--gligen-scheduled-sampling-beta", type=float, default=0.3)
    parser.add_argument("--global-prompt-template", default="{phrases}, photo, highly detailed, photorealistic")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    return parser.parse_args()


def draw_layout(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    palette = [(255, 64, 64), (40, 160, 90), (65, 105, 225), (240, 160, 30), (155, 80, 200)]
    for idx, item in enumerate(items):
        color = palette[idx % len(palette)]
        x, y, w, h = [float(v) for v in item["bbox"]]
        line = max(2, int(round(max(out.size) / 300)))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line)
        draw.text((x + 4, max(0, y - 18)), f"{item['class_name']} | {item['group']}", fill=color)
    return out


def build_gligen_phrase(class_name: str, tokens: list[str], include_class_name: bool) -> str:
    token_text = ",".join(tokens)
    if include_class_name:
        return f"{class_name}, {token_text}"
    return token_text


def prompt_for_failure_gligen(meta: dict, dataset: CocoMini, failure: dict,
                              prompt_scope: str | None, include_class_name: bool | None) -> tuple[str | None, str | None]:
    scope = prompt_scope or meta["prompt_scope"]
    class_id = int(failure["class_id"])
    group = choose_group(meta["groups"], class_id, scope)
    if group not in meta["groups"]:
        return None, None
    use_class = bool(meta.get("include_class_name", False)) if include_class_name is None else bool(include_class_name)
    return group, build_gligen_phrase(dataset.class_names[class_id], meta["groups"][group], use_class)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if args.clean_output:
        clean_output_dir(output)
    for child in ("originals", "masks", "results"):
        (output / child).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)

    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    dataset = CocoMini(args.annotations, args.images, class_subset=None)
    failures = load_failures(args.failures, class_subset=class_subset)
    by_image: dict[int, list[dict]] = defaultdict(list)
    for failure in failures:
        by_image[int(failure["image_id"])].append(failure)
    scenes = sorted(by_image.items())
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(scenes)
    scenes = scenes[:args.max_scenes]
    logger.info("GLIGEN prompt visualization start: scenes=%d failures=%d class_subset=%s",
                len(scenes), len(failures), subset_label(class_subset))

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

    manifest = []
    skipped = 0
    for scene_idx, (image_id, items) in enumerate(tqdm(scenes, desc="gligen prompt scenes"), start=1):
        info = dataset.images[int(image_id)]
        phrases: list[str] = []
        boxes: list[list[float]] = []
        layout_items: list[dict] = []
        for failure in items:
            if len(phrases) >= args.max_objects_per_scene:
                break
            group, prompt = prompt_for_failure_gligen(meta, dataset, failure, prompt_scope, args.include_class_name)
            if group is None or prompt is None:
                continue
            class_id = int(failure["class_id"])
            box = xywh_to_gligen_box(failure["bbox"], float(info["width"]), float(info["height"]))
            phrases.append(prompt)
            boxes.append(box)
            layout_items.append({
                "image_id": int(image_id),
                "annotation_id": failure.get("annotation_id"),
                "bbox": failure["bbox"],
                "gligen_box": box,
                "class_id": class_id,
                "class_name": dataset.class_names[class_id],
                "failure_type": failure.get("failure_type"),
                "group": group,
                "phrase": prompt,
            })
        if not phrases:
            skipped += 1
            continue

        prompt = build_global_prompt(phrases, args.global_prompt_template)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + int(image_id))
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            gligen_scheduled_sampling_beta=args.gligen_scheduled_sampling_beta,
            gligen_phrases=phrases,
            gligen_boxes=boxes,
            negative_prompt=args.negative_prompt,
            generator=generator,
        ).images[0]

        source = Image.open(dataset.image_path(image_id)).convert("RGB")
        layout = draw_layout(source, layout_items)
        stem = f"{scene_idx:04d}_img{image_id}_objects{len(layout_items)}"
        original_file = output / "originals" / f"{stem}.jpg"
        mask_file = output / "masks" / f"{stem}.jpg"
        result_file = output / "results" / f"{stem}.jpg"
        source.save(original_file, quality=95)
        layout.save(mask_file, quality=95)
        image.save(result_file, quality=95)

        class_names = ", ".join(sorted({x["class_name"] for x in layout_items}))
        manifest.append({
            "index": scene_idx,
            "image_id": int(image_id),
            "annotation_id": ",".join(str(x.get("annotation_id")) for x in layout_items),
            "bbox": [x["bbox"] for x in layout_items],
            "class_id": "multi",
            "class_name": class_names,
            "failure_type": "multi",
            "group": ",".join(sorted({x["group"] for x in layout_items})),
            "prompt": prompt,
            "checkpoint": ckpt_name,
            "layout_objects": layout_items,
            "original_file": rel(original_file, output),
            "mask_file": rel(mask_file, output),
            "result_file": rel(result_file, output),
        })
        if scene_idx == 1 or scene_idx % 10 == 0 or scene_idx == len(scenes):
            logger.info("GLIGEN progress: processed=%d/%d rendered=%d skipped=%d", scene_idx, len(scenes), len(manifest), skipped)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(output / "index.html", "GLIGEN Layout Prompt Visualization", manifest, "gligen_layout")
    logger.info("GLIGEN prompt visualization complete: rendered=%d skipped=%d html=%s", len(manifest), skipped, output / "index.html")


if __name__ == "__main__":
    main()
