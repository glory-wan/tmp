# 现在这个质量筛选用的是 COCO 标注的 segmentation mask，不是 SAM3 mask

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from detection_gligen_sdedit.common import CocoMini, parse_class_subset, subset_label


@dataclass
class FilterRules:
    min_mask_area_px: int = 512
    min_mask_area_ratio: float = 0.0005
    min_bbox_width_px: float = 24.0
    min_bbox_height_px: float = 24.0
    min_bbox_area_px: float = 1024.0
    min_bbox_area_ratio: float = 0.0008
    min_mask_bbox_fill_ratio: float = 0.08
    max_bbox_area_ratio: float | None = None
    large_bbox_area_ratio: float = 0.20


def resolved(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def cfg_get(config: dict[str, Any], dotted: str, default=None):
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def image_to_data_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 82) -> str:
    buffer = BytesIO()
    if fmt.upper() == "JPEG":
        image = image.convert("RGB")
        image.save(buffer, format=fmt, quality=quality)
    else:
        image.save(buffer, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def draw_coco_segmentation(segmentation, size: tuple[int, int]) -> Image.Image | None:
    if not segmentation:
        return None
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not polygon or len(polygon) < 6:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
            draw.polygon(points, fill=255)
        return mask
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils

            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, size[1], size[0])
            arr = mask_utils.decode(rle)
            if arr.ndim == 3:
                arr = arr.any(axis=2)
            return Image.fromarray((arr > 0).astype("uint8") * 255, mode="L")
        except Exception:
            return None
    return None


class SamMaskStore:
    def __init__(self, path: Path | None):
        self.path = path
        self.payload: Any = None
        self.by_ann_id: dict[str, Any] = {}
        if path is None:
            return
        if path.is_file() and path.suffix.lower() == ".json":
            self.payload = json.loads(path.read_text(encoding="utf-8"))
            self.by_ann_id = self._index_json(self.payload)

    def _index_json(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
            items = payload["annotations"]
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            return {str(k): v for k, v in payload.items()}
        else:
            return {}
        indexed = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = item.get("annotation_id", item.get("ann_id", item.get("coco_annotation_id", item.get("id"))))
            if key is not None:
                indexed[str(key)] = item
        return indexed

    def _load_mask_path(self, path: Path, size: tuple[int, int]) -> Image.Image | None:
        if not path.exists():
            return None
        mask = Image.open(path).convert("L")
        if mask.size != size:
            mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        return mask.point(lambda v: 255 if v > 0 else 0)

    def _candidate_paths(self, ann: dict) -> list[Path]:
        assert self.path is not None
        root = self.path if self.path.is_dir() else self.path.parent
        ann_id = str(ann.get("id", ""))
        image_id = str(ann.get("image_id", ""))
        stem_candidates = [
            ann_id,
            f"ann_{ann_id}",
            f"annotation_{ann_id}",
            f"{image_id}_{ann_id}",
            f"image_{image_id}_ann_{ann_id}",
        ]
        paths = []
        for stem in stem_candidates:
            for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                paths.append(root / f"{stem}{suffix}")
        return paths

    def load(self, ann: dict, size: tuple[int, int]) -> tuple[Image.Image | None, str]:
        if self.path is None:
            return None, "none"
        if self.path.is_dir():
            for path in self._candidate_paths(ann):
                mask = self._load_mask_path(path, size)
                if mask is not None:
                    return mask, f"sam:{path.name}"
            return None, "sam_missing"
        item = self.by_ann_id.get(str(ann.get("id")))
        if item is None:
            return None, "sam_missing"
        if isinstance(item, str):
            mask = self._load_mask_path(resolved(self.path.parent, item), size)
            return mask, "sam:path" if mask is not None else "sam_missing"
        if isinstance(item, dict):
            for key in ("mask_path", "file_name", "path"):
                if item.get(key):
                    mask = self._load_mask_path(resolved(self.path.parent, item[key]), size)
                    if mask is not None:
                        return mask, f"sam:{key}"
            for key in ("segmentation", "mask"):
                if item.get(key) is not None:
                    mask = draw_coco_segmentation(item[key], size)
                    if mask is not None:
                        return mask, f"sam:{key}"
        return None, "sam_unsupported"


def bbox_fallback_mask(size: tuple[int, int], bbox: list[float]) -> Image.Image:
    x, y, w, h = [float(v) for v in bbox]
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle([x, y, x + w, y + h], fill=255)
    return mask


def mask_pixel_area(mask: Image.Image) -> int:
    hist = mask.convert("1").histogram()
    return int(hist[255]) if len(hist) > 255 else int(sum(hist[1:]))


def classify_instance(row: dict[str, Any], rules: FilterRules) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    tags: list[str] = []
    if row["mask_area_px"] < rules.min_mask_area_px:
        reasons.append("mask_area_px")
    if row["mask_area_ratio"] < rules.min_mask_area_ratio:
        reasons.append("mask_area_ratio")
    if row["bbox_w"] < rules.min_bbox_width_px:
        reasons.append("bbox_width")
    if row["bbox_h"] < rules.min_bbox_height_px:
        reasons.append("bbox_height")
    if row["bbox_area_px"] < rules.min_bbox_area_px:
        reasons.append("bbox_area_px")
    if row["bbox_area_ratio"] < rules.min_bbox_area_ratio:
        reasons.append("bbox_area_ratio")
    if row["mask_bbox_fill_ratio"] < rules.min_mask_bbox_fill_ratio:
        reasons.append("mask_bbox_fill_ratio")
    if rules.max_bbox_area_ratio is not None and row["bbox_area_ratio"] > rules.max_bbox_area_ratio:
        reasons.append("max_bbox_area_ratio")

    if row["bbox_area_px"] < 32 * 32:
        tags.append("coco_small")
    elif row["bbox_area_px"] < 96 * 96:
        tags.append("coco_medium")
    else:
        tags.append("coco_large")
    if row["bbox_area_ratio"] >= rules.large_bbox_area_ratio:
        tags.append("large")
    if reasons:
        tags.append("small" if any("min" in r or r in {"bbox_width", "bbox_height"} for r in reasons) else "filtered")
    return ("filtered" if reasons else "keep"), reasons, tags


def overlay_instance(image: Image.Image, mask: Image.Image, bbox: list[float], status: str, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
    out = image.convert("RGBA")
    color = (33, 150, 83, 112) if status == "keep" else (220, 38, 38, 120)
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    overlay.paste(color, mask=mask)
    out = Image.alpha_composite(out, overlay)
    draw = ImageDraw.Draw(out)
    x, y, w, h = [float(v) * scale for v in bbox]
    line = max(2, int(3 * scale))
    box_color = (46, 204, 113, 255) if status == "keep" else (255, 77, 77, 255)
    draw.rectangle([x, y, x + w, y + h], outline=box_color, width=line)
    return out.convert("RGB")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manual_invalid(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("annotation_ids", payload.get("invalid_annotation_ids", []))
        else:
            values = payload
        return {int(x) for x in values}
    invalid = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                invalid.add(int(line.split(",")[0]))
    return invalid


def build_html(path: Path, rows: list[dict[str, Any]], rules: FilterRules) -> None:
    counts = {
        "all": len(rows),
        "keep": sum(1 for r in rows if r["status"] == "keep"),
        "filtered": sum(1 for r in rows if r["status"] == "filtered"),
        "small": sum(1 for r in rows if "small" in r["tags"] or "coco_small" in r["tags"]),
        "large": sum(1 for r in rows if "large" in r["tags"] or "coco_large" in r["tags"]),
        "manual_invalid": sum(1 for r in rows if r["manual_invalid"]),
    }
    cards = []
    for row in rows:
        attrs = " ".join([row["status"], *row["tags"], "manual_invalid" if row["manual_invalid"] else ""])
        reason = ", ".join(row["filter_reasons"]) if row["filter_reasons"] else "none"
        cards.append(f"""
        <article class="card {html.escape(attrs)}" data-status="{row['status']}" data-tags="{html.escape(attrs)}">
          <img src="{row['overlay_uri']}" loading="lazy" alt="annotation {row['annotation_id']}">
          <div class="meta">
            <div class="topline"><strong>{html.escape(row['class_name'])}</strong><span>ann {row['annotation_id']}</span></div>
            <div>image {row['image_id']} | {html.escape(row['file_name'])}</div>
            <div>status <b class="{row['status']}">{row['status']}</b> | source {html.escape(row['mask_source'])}</div>
            <div>bbox {row['bbox_w']:.1f}x{row['bbox_h']:.1f}, area {row['bbox_area_px']:.0f} ({row['bbox_area_ratio']:.4%})</div>
            <div>mask {row['mask_area_px']} px ({row['mask_area_ratio']:.4%}), fill {row['mask_bbox_fill_ratio']:.3f}</div>
            <div>reasons: {html.escape(reason)}</div>
            <div>tags: {html.escape(', '.join(row['tags']))}</div>
          </div>
        </article>
        """)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>COCO Instance Quality</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #18202a; }}
header {{ position: sticky; top: 0; z-index: 2; background: white; border-bottom: 1px solid #d8dde5; padding: 14px 18px; }}
h1 {{ margin: 0 0 10px; font-size: 20px; }}
.buttons {{ display: flex; gap: 8px; flex-wrap: wrap; }}
button {{ border: 1px solid #b8c0cc; background: #fff; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
button.active {{ background: #18202a; color: white; border-color: #18202a; }}
.rules {{ margin-top: 10px; color: #4b5563; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px; padding: 16px; }}
.card {{ background: white; border: 1px solid #d8dde5; border-radius: 8px; overflow: hidden; }}
.card img {{ width: 100%; display: block; background: #111; }}
.meta {{ padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
.topline {{ display: flex; justify-content: space-between; gap: 8px; font-size: 15px; }}
.keep {{ color: #16803c; }}
.filtered {{ color: #b42318; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>COCO Instance Quality Analysis</h1>
  <div class="buttons">
    <button class="active" data-filter="all">All ({counts['all']})</button>
    <button data-filter="keep">Keep ({counts['keep']})</button>
    <button data-filter="filtered">Filtered ({counts['filtered']})</button>
    <button data-filter="small">Small ({counts['small']})</button>
    <button data-filter="large">Large ({counts['large']})</button>
    <button data-filter="manual_invalid">Manual Invalid ({counts['manual_invalid']})</button>
  </div>
  <div class="rules">Rules: {html.escape(json.dumps(rules.__dict__, ensure_ascii=False))}</div>
</header>
<main class="grid">
{''.join(cards)}
</main>
<script>
const buttons = document.querySelectorAll('button[data-filter]');
const cards = document.querySelectorAll('.card');
buttons.forEach(btn => btn.addEventListener('click', () => {{
  buttons.forEach(x => x.classList.remove('active'));
  btn.classList.add('active');
  const filter = btn.dataset.filter;
  cards.forEach(card => {{
    const tags = card.dataset.tags || '';
    const show = filter === 'all' || tags.split(/\\s+/).includes(filter);
    card.classList.toggle('hidden', !show);
  }});
}}));
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def make_filtered_annotations(annotation_path: Path, output_path: Path, keep_ids: set[int]) -> None:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"] = [ann for ann in data.get("annotations", []) if int(ann.get("id", -1)) in keep_ids]
    used_images = {int(ann["image_id"]) for ann in data["annotations"]}
    data["images"] = [img for img in data.get("images", []) if int(img.get("id", -1)) in used_images]
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and filter COCO instance quality with SAM/segmentation masks.")
    parser.add_argument("--config", default=None, help="Optional YAML config; CLI args override it.")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--images", default=None)
    parser.add_argument("--sam-masks", default=None, help="Optional SAM/SAM3 mask JSON or mask PNG directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--thumb-size", type=int, default=520)
    parser.add_argument("--manual-invalid", default=None, help="JSON/list/csv with annotation ids judged invalid by humans.")
    parser.add_argument("--write-filtered-annotations", action="store_true")
    parser.add_argument("--min-mask-area-px", type=int, default=None)
    parser.add_argument("--min-mask-area-ratio", type=float, default=None)
    parser.add_argument("--min-bbox-width-px", type=float, default=None)
    parser.add_argument("--min-bbox-height-px", type=float, default=None)
    parser.add_argument("--min-bbox-area-px", type=float, default=None)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=None)
    parser.add_argument("--min-mask-bbox-fill-ratio", type=float, default=None)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=None)
    parser.add_argument("--large-bbox-area-ratio", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    config = load_config(resolved(root, args.config))
    quality_cfg = cfg_get(config, "instance_quality", {})
    data_cfg = cfg_get(config, "data", {})

    annotations = resolved(root, args.annotations or quality_cfg.get("annotations") or data_cfg.get("train_annotations"))
    images = resolved(root, args.images or quality_cfg.get("images") or data_cfg.get("train_images"))
    output_dir = resolved(root, args.output_dir or quality_cfg.get("output_dir") or "outputs/instance_quality")
    sam_masks = resolved(root, args.sam_masks or quality_cfg.get("sam_masks"))
    manual_invalid_path = resolved(root, args.manual_invalid or quality_cfg.get("manual_invalid"))
    if annotations is None or images is None or output_dir is None:
        raise ValueError("--annotations, --images, and --output-dir are required unless provided by config.")

    rules_cfg = quality_cfg.get("rules", {})
    rules = FilterRules(
        min_mask_area_px=args.min_mask_area_px if args.min_mask_area_px is not None else int(rules_cfg.get("min_mask_area_px", 512)),
        min_mask_area_ratio=args.min_mask_area_ratio if args.min_mask_area_ratio is not None else float(rules_cfg.get("min_mask_area_ratio", 0.0005)),
        min_bbox_width_px=args.min_bbox_width_px if args.min_bbox_width_px is not None else float(rules_cfg.get("min_bbox_width_px", 24.0)),
        min_bbox_height_px=args.min_bbox_height_px if args.min_bbox_height_px is not None else float(rules_cfg.get("min_bbox_height_px", 24.0)),
        min_bbox_area_px=args.min_bbox_area_px if args.min_bbox_area_px is not None else float(rules_cfg.get("min_bbox_area_px", 1024.0)),
        min_bbox_area_ratio=args.min_bbox_area_ratio if args.min_bbox_area_ratio is not None else float(rules_cfg.get("min_bbox_area_ratio", 0.0008)),
        min_mask_bbox_fill_ratio=args.min_mask_bbox_fill_ratio if args.min_mask_bbox_fill_ratio is not None else float(rules_cfg.get("min_mask_bbox_fill_ratio", 0.08)),
        max_bbox_area_ratio=args.max_bbox_area_ratio if args.max_bbox_area_ratio is not None else rules_cfg.get("max_bbox_area_ratio"),
        large_bbox_area_ratio=args.large_bbox_area_ratio if args.large_bbox_area_ratio is not None else float(rules_cfg.get("large_bbox_area_ratio", 0.20)),
    )
    if rules.max_bbox_area_ratio is not None:
        rules.max_bbox_area_ratio = float(rules.max_bbox_area_ratio)

    class_ids = args.class_ids if args.class_ids is not None else quality_cfg.get("class_ids")
    classes_subset = args.classes_subset if args.classes_subset is not None else quality_cfg.get("classes_subset")
    class_subset = parse_class_subset(class_ids is not None or classes_subset is not None, classes_subset, class_ids)
    max_instances = args.max_instances if args.max_instances is not None else quality_cfg.get("max_instances")
    max_instances = int(max_instances) if max_instances is not None else None
    thumb_size = int(args.thumb_size if args.thumb_size is not None else quality_cfg.get("thumb_size", 520))

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CocoMini(annotations, images, class_subset=class_subset)
    sam_store = SamMaskStore(sam_masks)
    manual_invalid = load_manual_invalid(manual_invalid_path)

    rows: list[dict[str, Any]] = []
    seen = 0
    for image_id, image_path, anns in tqdm(list(dataset.iter_images(None)), desc="instances"):
        image = Image.open(image_path).convert("RGB")
        image_area = float(image.width * image.height)
        for ann in anns:
            if max_instances is not None and seen >= max_instances:
                break
            seen += 1
            mask, mask_source = sam_store.load(ann, image.size)
            if mask is None:
                mask = draw_coco_segmentation(ann.get("segmentation"), image.size)
                mask_source = "coco_segmentation" if mask is not None else mask_source
            if mask is None:
                mask = bbox_fallback_mask(image.size, ann["bbox"])
                mask_source = "bbox_fallback"

            x, y, w, h = [float(v) for v in ann["bbox"]]
            bbox_area = max(0.0, w) * max(0.0, h)
            mask_area = mask_pixel_area(mask)
            row = {
                "annotation_id": int(ann["id"]),
                "image_id": int(image_id),
                "file_name": dataset.images[int(image_id)]["file_name"],
                "category_id": int(ann["category_id"]),
                "class_id": int(ann["class_id"]),
                "class_name": dataset.class_names[int(ann["class_id"])],
                "image_width": image.width,
                "image_height": image.height,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox_area_px": bbox_area,
                "bbox_area_ratio": bbox_area / max(image_area, 1.0),
                "mask_area_px": mask_area,
                "mask_area_ratio": mask_area / max(image_area, 1.0),
                "mask_bbox_fill_ratio": mask_area / max(bbox_area, 1.0),
                "mask_source": mask_source,
                "manual_invalid": int(ann["id"]) in manual_invalid,
            }
            status, reasons, tags = classify_instance(row, rules)
            if row["manual_invalid"]:
                tags.append("manual_invalid")
            row["status"] = status
            row["filter_reasons"] = reasons
            row["tags"] = tags
            row["overlay_uri"] = image_to_data_uri(overlay_instance(image, mask, ann["bbox"], status, thumb_size))
            rows.append(row)
        if max_instances is not None and seen >= max_instances:
            break

    json_rows = [{k: v for k, v in row.items() if k != "overlay_uri"} for row in rows]
    keep_ids = {int(row["annotation_id"]) for row in rows if row["status"] == "keep"}
    summary = {
        "annotations": str(annotations),
        "images": str(images),
        "sam_masks": str(sam_masks) if sam_masks else None,
        "class_subset": subset_label(class_subset),
        "rules": rules.__dict__,
        "total": len(rows),
        "keep": sum(1 for row in rows if row["status"] == "keep"),
        "filtered": sum(1 for row in rows if row["status"] == "filtered"),
        "manual_invalid": len(manual_invalid),
        "manual_invalid_kept": [row["annotation_id"] for row in rows if row["manual_invalid"] and row["status"] == "keep"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "instances_quality.json").write_text(json.dumps(json_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "instances_quality.csv", json_rows)
    build_html(output_dir / "index.html", rows, rules)
    if args.write_filtered_annotations or bool(quality_cfg.get("write_filtered_annotations", False)):
        make_filtered_annotations(annotations, output_dir / "filtered_annotations.json", keep_ids)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"HTML: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
