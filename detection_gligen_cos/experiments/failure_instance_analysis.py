# 失败实例分析
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from detection_gligen_sdedit.common import CocoMini, parse_class_subset, setup_logger, subset_label


def load_failures(path: str | Path) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    failures = payload.get("failures", [])
    if not isinstance(failures, list):
        raise ValueError(f"invalid failures.json: {path}")
    return payload.get("meta", {}), failures


def size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def failure_enriched_rows(failures: list[dict], dataset: CocoMini) -> list[dict]:
    rows = []
    for idx, item in enumerate(failures):
        class_id = int(item["class_id"])
        x, y, w, h = [safe_float(v) for v in item["bbox"]]
        area = max(w, 0.0) * max(h, 0.0)
        aspect_ratio = w / max(h, 1e-9)
        image_id = int(item["image_id"])
        info = dataset.images.get(image_id, {})
        image_area = safe_float(info.get("width"), 0.0) * safe_float(info.get("height"), 0.0)
        rows.append({
            "failure_index": idx,
            "image_id": image_id,
            "file_name": info.get("file_name", ""),
            "class_id": class_id,
            "class_name": dataset.class_names[class_id] if 0 <= class_id < len(dataset.class_names) else str(class_id),
            "failure_type": str(item.get("failure_type", "unknown")),
            "confidence": safe_float(item.get("confidence", 0.0)),
            "top2_gap": safe_float(item.get("top2_gap", 1.0)),
            "forgotten_count": int(item.get("forgotten_count", 0)),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "bbox_area": area,
            "bbox_area_ratio": area / max(image_area, 1e-9),
            "bbox_aspect_ratio": aspect_ratio,
            "bbox_size_bucket": size_bucket(area),
            "source": item.get("source", ""),
        })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def count_rows(counter: Counter, key_name: str, value_name: str = "count") -> list[dict]:
    total = sum(counter.values())
    return [
        {key_name: key, value_name: count, "ratio": count / total if total else 0.0}
        for key, count in counter.most_common()
    ]


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def q(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    return {
        "count": len(values),
        "min": min(values),
        "p25": q(0.25),
        "median": median(values),
        "mean": mean(values),
        "p75": q(0.75),
        "max": max(values),
    }


def build_summary(meta: dict, rows: list[dict], dataset: CocoMini) -> dict:
    by_class = Counter(row["class_id"] for row in rows)
    by_type = Counter(row["failure_type"] for row in rows)
    by_size = Counter(row["bbox_size_bucket"] for row in rows)
    by_image = Counter(row["image_id"] for row in rows)
    class_type = defaultdict(Counter)
    for row in rows:
        class_type[row["class_id"]][row["failure_type"]] += 1
    return {
        "meta": meta,
        "total_failures": len(rows),
        "unique_images": len(by_image),
        "class_subset_seen": sorted(by_class),
        "class_distribution": [
            {
                "class_id": class_id,
                "class_name": dataset.class_names[class_id],
                "count": count,
                "ratio": count / len(rows) if rows else 0.0,
            }
            for class_id, count in by_class.most_common()
        ],
        "failure_type_distribution": count_rows(by_type, "failure_type"),
        "bbox_size_bucket_distribution": count_rows(by_size, "bbox_size_bucket"),
        "bbox_area_stats": quantiles([row["bbox_area"] for row in rows]),
        "bbox_area_ratio_stats": quantiles([row["bbox_area_ratio"] for row in rows]),
        "bbox_aspect_ratio_stats": quantiles([row["bbox_aspect_ratio"] for row in rows]),
        "failures_per_image_stats": quantiles(list(by_image.values())),
        "class_failure_type_distribution": {
            str(class_id): {
                "class_name": dataset.class_names[class_id],
                "failure_types": dict(counter),
                "total": sum(counter.values()),
            }
            for class_id, counter in sorted(class_type.items())
        },
    }


def table_outputs(rows: list[dict], summary: dict, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(tables / "failures_enriched.csv", rows)
    write_csv(tables / "class_distribution.csv", summary["class_distribution"],
              ["class_id", "class_name", "count", "ratio"])
    write_csv(tables / "failure_type_distribution.csv", summary["failure_type_distribution"],
              ["failure_type", "count", "ratio"])
    write_csv(tables / "bbox_size_bucket_distribution.csv", summary["bbox_size_bucket_distribution"],
              ["bbox_size_bucket", "count", "ratio"])

    per_image = Counter(row["image_id"] for row in rows)
    per_image_rows = [
        {"image_id": image_id, "failure_count": count}
        for image_id, count in per_image.most_common()
    ]
    write_csv(tables / "failures_per_image.csv", per_image_rows, ["image_id", "failure_count"])

    class_type_rows = []
    for class_id, payload in summary["class_failure_type_distribution"].items():
        for failure_type, count in payload["failure_types"].items():
            class_type_rows.append({
                "class_id": class_id,
                "class_name": payload["class_name"],
                "failure_type": failure_type,
                "count": count,
                "class_total": payload["total"],
                "ratio_in_class": count / payload["total"] if payload["total"] else 0.0,
            })
    write_csv(tables / "class_failure_type_distribution.csv", class_type_rows,
              ["class_id", "class_name", "failure_type", "count", "class_total", "ratio_in_class"])


def load_font(size: int = 18):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return None


def draw_failure(image: Image.Image, row: dict, *, line_width: int = 4, max_side: int | None = None) -> Image.Image:
    image = image.convert("RGB").copy()
    sx = sy = 1.0
    if max_side is not None and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        sx = image.width / safe_float(row.get("image_width_original", image.width), image.width)
        sy = image.height / safe_float(row.get("image_height_original", image.height), image.height)
    draw = ImageDraw.Draw(image)
    font = load_font(16)
    x = row["bbox_x"] * sx
    y = row["bbox_y"] * sy
    w = row["bbox_w"] * sx
    h = row["bbox_h"] * sy
    color = (255, 48, 48)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
    label = (
        f"{row['class_name']} | {row['failure_type']} | "
        f"conf={row['confidence']:.3f} | area={row['bbox_size_bucket']}"
    )
    text_box = draw.textbbox((0, 0), label, font=font)
    tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
    tx, ty = int(max(0, min(x, image.width - tw - 8))), int(max(0, y - th - 8))
    draw.rectangle([tx, ty, tx + tw + 8, ty + th + 6], fill=(255, 48, 48))
    draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255), font=font)
    return image


def open_failure_image(row: dict, dataset: CocoMini) -> Image.Image | None:
    path = dataset.image_path(int(row["image_id"]))
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def annotate_rows_with_image_size(rows: list[dict], dataset: CocoMini) -> None:
    for row in rows:
        info = dataset.images.get(int(row["image_id"]), {})
        row["image_width_original"] = safe_float(info.get("width"), 0.0)
        row["image_height_original"] = safe_float(info.get("height"), 0.0)


def make_grid(images: list[Image.Image], labels: list[str], cols: int, tile_size: int = 320) -> Image.Image:
    if not images:
        return Image.new("RGB", (tile_size, tile_size), "white")
    rows = math.ceil(len(images) / cols)
    label_h = 52
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    for idx, (image, label) in enumerate(zip(images, labels)):
        col = idx % cols
        row = idx // cols
        x0 = col * tile_size
        y0 = row * (tile_size + label_h)
        thumb = image.copy()
        thumb.thumbnail((tile_size, tile_size))
        px = x0 + (tile_size - thumb.width) // 2
        py = y0 + label_h + (tile_size - thumb.height) // 2
        canvas.paste(thumb, (px, py))
        draw.text((x0 + 6, y0 + 6), label[:90], fill=(0, 0, 0), font=font)
    return canvas


def save_random_instances(rows: list[dict], dataset: CocoMini, output: Path, sample_count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(sample_count, len(rows))) if rows else []
    out_dir = output / "visualizations" / "random_instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, row in enumerate(tqdm(chosen, desc="random failure visualizations"), start=1):
        image = open_failure_image(row, dataset)
        if image is None:
            continue
        rendered = draw_failure(image, row, max_side=1400)
        name = f"{rank:04d}_img{row['image_id']}_cls{row['class_id']}_{row['failure_type']}.jpg"
        rendered.save(out_dir / name, quality=95)
        manifest.append({**row, "visualization": str((out_dir / name).relative_to(output))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def save_class_grids(rows: list[dict], dataset: CocoMini, output: Path, samples_per_class: int, seed: int) -> None:
    rng = random.Random(seed)
    out_dir = output / "visualizations" / "class_grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    all_grid_images, all_grid_labels = [], []
    for class_id in sorted(by_class):
        chosen = rng.sample(by_class[class_id], min(samples_per_class, len(by_class[class_id])))
        images, labels = [], []
        for row in chosen:
            image = open_failure_image(row, dataset)
            if image is None:
                continue
            rendered = draw_failure(image, row, line_width=5, max_side=900)
            label = f"{row['class_name']} | {row['failure_type']} | conf={row['confidence']:.2f}"
            images.append(rendered)
            labels.append(label)
            all_grid_images.append(rendered)
            all_grid_labels.append(f"{class_id}:{label}")
        if images:
            grid = make_grid(images, labels, cols=min(4, len(images)), tile_size=320)
            grid.save(out_dir / f"class_{class_id}_{dataset.class_names[class_id]}.jpg", quality=95)
    if all_grid_images:
        overview = make_grid(all_grid_images, all_grid_labels, cols=min(5, len(all_grid_images)), tile_size=280)
        overview.save(out_dir / "all_classes_overview.jpg", quality=95)


def save_plots(rows: list[dict], summary: dict, output: Path, top_k: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output / "visualizations" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dist = summary["class_distribution"][:top_k]
    plt.figure(figsize=(max(8, len(class_dist) * 0.7), 5))
    plt.bar([f"{x['class_id']}:{x['class_name']}" for x in class_dist], [x["count"] for x in class_dist])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Failure instances")
    plt.title("Failure count by class")
    plt.tight_layout()
    plt.savefig(out_dir / "class_frequency_bar.png", dpi=180)
    plt.close()

    type_dist = summary["failure_type_distribution"]
    plt.figure(figsize=(8, 5))
    plt.bar([x["failure_type"] for x in type_dist], [x["count"] for x in type_dist])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Failure instances")
    plt.title("Failure type distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_type_bar.png", dpi=180)
    plt.close()

    areas = [row["bbox_area"] for row in rows]
    area_ratios = [row["bbox_area_ratio"] for row in rows]
    aspect = [row["bbox_aspect_ratio"] for row in rows if row["bbox_aspect_ratio"] > 0]
    per_image = list(Counter(row["image_id"] for row in rows).values())

    plt.figure(figsize=(8, 5))
    plt.hist(areas, bins=50)
    plt.xlabel("BBox area in pixels")
    plt.ylabel("Count")
    plt.title("BBox area distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "bbox_area_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(area_ratios, bins=50)
    plt.xlabel("BBox area / image area")
    plt.ylabel("Count")
    plt.title("BBox relative area distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "bbox_area_ratio_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist([min(x, 10.0) for x in aspect], bins=50)
    plt.xlabel("BBox aspect ratio w/h, clipped at 10")
    plt.ylabel("Count")
    plt.title("BBox aspect ratio distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "bbox_aspect_ratio_hist.png", dpi=180)
    plt.close()

    size_dist = summary["bbox_size_bucket_distribution"]
    order = ["small", "medium", "large"]
    size_counts = {x["bbox_size_bucket"]: x["count"] for x in size_dist}
    plt.figure(figsize=(6, 5))
    plt.bar(order, [size_counts.get(k, 0) for k in order])
    plt.ylabel("Count")
    plt.title("COCO-style bbox size buckets")
    plt.tight_layout()
    plt.savefig(out_dir / "bbox_size_bucket_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    bins = range(1, max(per_image, default=1) + 2)
    plt.hist(per_image, bins=bins, align="left")
    plt.xlabel("Failures per image")
    plt.ylabel("Image count")
    plt.title("Failure instances per image")
    plt.tight_layout()
    plt.savefig(out_dir / "failures_per_image_hist.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and visualize instance-level YOLO failure mining results.")
    parser.add_argument("--failures", required=True, help="Path to failures.json produced by mine_failures.py.")
    parser.add_argument("--annotations", required=True, help="COCO annotation json used for mining.")
    parser.add_argument("--images", required=True, help="COCO image directory used for mining.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    parser.add_argument("--random-samples", type=int, default=80)
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--top-k-classes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output = Path(args.output_dir)
    logger = setup_logger(output)
    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None,
                                      args.classes_subset, args.class_ids)
    dataset = CocoMini(args.annotations, args.images, class_subset=None)
    meta, failures = load_failures(args.failures)
    if class_subset is not None:
        failures = [x for x in failures if int(x["class_id"]) in class_subset]
    logger.info(
        "failure analysis start: failures=%s total_after_filter=%d annotations=%s images=%s class_filter=%s",
        args.failures, len(failures), args.annotations, args.images, subset_label(class_subset),
    )

    rows = failure_enriched_rows(failures, dataset)
    annotate_rows_with_image_size(rows, dataset)
    summary = build_summary(meta, rows, dataset)
    table_outputs(rows, summary, output)
    logger.info("tables written: %s", output / "tables")

    save_random_instances(rows, dataset, output, args.random_samples, args.seed)
    save_class_grids(rows, dataset, output, args.samples_per_class, args.seed)
    save_plots(rows, summary, output, args.top_k_classes)
    logger.info("visualizations written: %s", output / "visualizations")
    logger.info(
        "failure analysis complete: failures=%d unique_images=%d summary=%s",
        summary["total_failures"], summary["unique_images"], output / "tables" / "summary.json",
    )


if __name__ == "__main__":
    main()
