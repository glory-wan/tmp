from __future__ import annotations

import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from detection_gligen_sdedit.generate_hard_examples import build_object_prompt, choose_group, install_embeddings


def load_object_prompts_checkpoint(tokens_dir: Path, checkpoint: str | None) -> tuple[dict[str, Any], str, dict[str, torch.Tensor]]:
    meta = json.loads((tokens_dir / "object_prompts.json").read_text(encoding="utf-8"))
    ckpt_name = checkpoint or meta["latest"]
    embeddings = torch.load(tokens_dir / ckpt_name, map_location="cpu")
    return meta, ckpt_name, embeddings


def prompt_for_failure(meta: dict[str, Any], dataset, failure: dict[str, Any],
                       prompt_scope: str | None, include_class_name: bool | None) -> tuple[str | None, str | None]:
    scope = prompt_scope or meta["prompt_scope"]
    class_id = int(failure["class_id"])
    group = choose_group(meta["groups"], class_id, scope)
    if group not in meta["groups"]:
        return None, None
    use_class = bool(meta.get("include_class_name", False)) if include_class_name is None else bool(include_class_name)
    return group, build_object_prompt(dataset.class_names[class_id], meta["groups"][group], use_class)


def clean_output_dir(output: Path) -> None:
    for child in ("originals", "masks", "results", "crops", "images"):
        path = output / child
        if path.exists():
            shutil.rmtree(path)
    for name in ("index.html", "manifest.json", "selected_images.json", "pipeline.log"):
        (output / name).unlink(missing_ok=True)


def xywh_iou(box: list[float] | tuple[float, ...], other: list[float] | tuple[float, ...]) -> float:
    ax, ay, aw, ah = [float(x) for x in box]
    bx, by, bw, bh = [float(x) for x in other]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return inter / max(union, 1e-9)


def matched_detector_score(predictions: list[dict[str, Any]], bbox, class_id: int, min_iou: float) -> dict[str, Any]:
    best = None
    best_iou = 0.0
    for pred in predictions:
        iou = xywh_iou(bbox, pred["bbox"])
        if int(pred.get("class_id", -1)) == int(class_id) and iou > best_iou:
            best = pred
            best_iou = iou
    if best is None or best_iou < min_iou:
        return {"confidence": None, "iou": best_iou, "class_id": None}
    return {"confidence": float(best.get("confidence", 0.0)), "iou": best_iou, "class_id": int(best["class_id"])}


def draw_bbox_mask_overlay(image: Image.Image, mask: Image.Image, bbox, label: str, color=(255, 64, 64)) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste((*color, 110), mask=mask.convert("L"))
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(out)
    x, y, w, h = [float(v) for v in bbox]
    line = max(2, int(round(max(base.size) / 300)))
    draw.rectangle([x, y, x + w, y + h], outline=(*color, 255), width=line)
    draw.text((x + 4, max(0, y - 18)), label, fill=(*color, 255))
    return out.convert("RGB")


def crop_with_padding(image: Image.Image, bbox, padding: float = 0.25) -> Image.Image:
    x, y, w, h = [float(v) for v in bbox]
    pad_x, pad_y = w * padding, h * padding
    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(image.width, int(math.ceil(x + w + pad_x)))
    bottom = min(image.height, int(math.ceil(y + h + pad_y)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_html(path: Path, title: str, manifest: list[dict[str, Any]], mode: str) -> None:
    cards = []
    for item in manifest:
        prompt = html.escape(str(item.get("prompt", "")))
        group = html.escape(str(item.get("group", "")))
        cls = html.escape(f"{item.get('class_id')} {item.get('class_name', '')}")
        score = item.get("detector", {})
        score_text = ""
        if isinstance(score, dict):
            orig = score.get("original", {})
            gen = score.get("generated", {})
            score_text = (
                f"orig score={orig.get('confidence')} iou={orig.get('iou')}<br>"
                f"gen score={gen.get('confidence')} iou={gen.get('iou')}"
            )
        images = []
        for key, label in (("original_file", "Original"), ("mask_file", "Mask/Bbox"), ("result_file", "Result"), ("crop_file", "Result crop")):
            if item.get(key):
                images.append(f"""
                <figure>
                  <figcaption>{label}</figcaption>
                  <img src="{html.escape(item[key])}" loading="lazy">
                </figure>
                """)
        cards.append(f"""
        <article class="card" data-class="{html.escape(str(item.get('class_id')))}" data-group="{group}">
          <div class="media">{''.join(images)}</div>
          <div class="meta">
            <b>{cls}</b> | group {group}<br>
            image {item.get('image_id')} | failure {html.escape(str(item.get('failure_type', '')))} | ann {item.get('annotation_id')}<br>
            bbox {html.escape(str(item.get('bbox')))}<br>
            {score_text}<br>
            <span class="prompt">{prompt}</span>
          </div>
        </article>
        """)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fa; color: #1d2733; }}
header {{ position: sticky; top: 0; z-index: 2; background: white; border-bottom: 1px solid #d7dee8; padding: 14px 18px; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
.muted {{ color: #667085; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 14px; padding: 16px; }}
.card {{ background: white; border: 1px solid #d7dee8; border-radius: 8px; overflow: hidden; }}
.media {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 10px; }}
figure {{ margin: 0; }}
figcaption {{ color: #475467; font-size: 12px; margin-bottom: 4px; }}
img {{ width: 100%; display: block; border: 1px solid #d7dee8; border-radius: 6px; background: #111; }}
.meta {{ border-top: 1px solid #e4e9f0; padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
.prompt {{ color: #344054; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="muted">mode={html.escape(mode)} | items={len(manifest)}</div>
</header>
<main class="grid">
{''.join(cards)}
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")
