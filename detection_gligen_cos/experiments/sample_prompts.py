# 单独使用prompt 生成的可视化实验
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from detection_gligen_sdedit.common import CocoMini, load_depth_style_components, patch_huggingface_hub_for_diffusers_024, setup_logger
from detection_gligen_sdedit.generate_hard_examples import install_embeddings


def load_prompts(tokens_dir: Path, checkpoint: str | None):
    meta = json.loads((tokens_dir / "object_prompts.json").read_text(encoding="utf-8"))
    ckpt_name = checkpoint or meta["latest"]
    embeddings = torch.load(tokens_dir / ckpt_name, map_location="cpu")
    return meta, ckpt_name, embeddings


def class_name_for_group(dataset: CocoMini, group: str) -> tuple[int, str]:
    parts = group.split("_")
    class_id = int(parts[1])
    return class_id, dataset.class_names[class_id]


# def prompt_variants(class_name: str, tokens: list[str]) -> dict[str, str]:
#     token_text = ",".join(tokens)
#     return {
#         "class_only": f"{class_name}, photo, highly detailed, photorealistic",
#         "tokens_only": f"{token_text}, photo, highly detailed, photorealistic",
#         "class_tokens": f"{class_name}, {token_text}, photo, highly detailed, photorealistic",
#     }

def prompt_variants(class_name: str, tokens: list[str]) -> dict[str, str]:
    token_text = ",".join(tokens)
    return {
        "class_only": f"a photo of a {class_name}",
        "tokens_only": f"a photo of a {token_text}",
        "orign_combin": f"a photo of a {token_text} {class_name} ",

        "token_class": f"a photo of {token_text} object, {class_name}",
        "class_token": f"a photo of a {class_name} containing {token_text}",
        "only1": f"a {token_text}",
        "now": f"{class_name}, {tokens}, photo, highly detailed, photorealistic",
        "origin": f"a photo of {token_text},{class_name}, highly detailed, photorealistic"
    }


def annotate_grid(images: list[Image.Image], labels: list[str], cols: int, tile: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    label_h = 34
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    for i, (img, label) in enumerate(zip(images, labels)):
        x = (i % cols) * tile
        y = (i // cols) * (tile + label_h)
        canvas.paste(img.resize((tile, tile)), (x, y + label_h))
        draw.text((x + 4, y + 6), label[:70], fill=(0, 0, 0), font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative standalone prompt sampling for learned object tokens.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_minitrain2017.json")
    parser.add_argument("--images", default="data/coco/images/train2017")
    parser.add_argument("--tokens-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="learned_embeds-*.bin. Defaults to object_prompts.json latest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", default=None, help="Comma-separated groups, e.g. class_14,class_25. Defaults to all groups.")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark, no object, missing object")
    args = parser.parse_args()

    output = Path(args.output_dir)
    logger = setup_logger(output)
    patch_huggingface_hub_for_diffusers_024()
    dataset = CocoMini(args.annotations, args.images, class_subset=None)
    meta, ckpt_name, embeddings = load_prompts(Path(args.tokens_dir), args.checkpoint)
    groups = args.groups.split(",") if args.groups else sorted(meta["groups"])
    logger.info("standalone prompt sampling: tokens_dir=%s checkpoint=%s groups=%s num_images=%d",
                args.tokens_dir, ckpt_name, groups, args.num_images)

    tokenizer, text_encoder, vae, unet, scheduler = load_depth_style_components(
        args.pretrained_model_name_or_path,
        dtype=torch.float16,
        local_files_only=args.local_files_only,
        text_encoder_dtype=torch.float16,
    )
    install_embeddings(tokenizer, text_encoder, meta["groups"], embeddings)


    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
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

    manifest = []
    for group in tqdm(groups, desc="groups"):
        class_id, class_name = class_name_for_group(dataset, group)
        variants = prompt_variants(class_name, meta["groups"][group])
        group_dir = output / ckpt_name.replace(".bin", "") / group
        group_dir.mkdir(parents=True, exist_ok=True)
        grid_images, grid_labels = [], []
        for variant, prompt in variants.items():
            variant_dir = group_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(args.num_images):
                seed = args.seed + class_id * 1000 + idx
                generator = torch.Generator(device=args.device).manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
                name = f"{variant}_{idx:02d}_seed{seed}.jpg"
                image.save(variant_dir / name, quality=95)
                grid_images.append(image)
                grid_labels.append(f"{group} | {variant} | seed={seed}")
                manifest.append({
                    "group": group,
                    "class_id": class_id,
                    "class_name": class_name,
                    "variant": variant,
                    "seed": seed,
                    "prompt": prompt,
                    "file": str((variant_dir / name).relative_to(output)),
                    "checkpoint": ckpt_name,
                })
        grid = annotate_grid(grid_images, grid_labels, cols=args.num_images, tile=min(args.resolution, 512))
        grid.save(group_dir / "grid.jpg", quality=95)
        logger.info("saved group grid: %s", group_dir / "grid.jpg")

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("standalone prompt sampling complete: output=%s manifest=%s", output, output / "manifest.json")


if __name__ == "__main__":
    main()
