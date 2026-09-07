from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from detection_gligen_sdedit.visualize_yolo_labels import COCO_NAMES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def parse_yolo_labels(path: Path, width: int, height: int) -> list[dict]:
    if not path.exists():
        return []
    boxes = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = [float(x) for x in parts[1:]]
        box_w = bw * width
        box_h = bh * height
        x = xc * width - box_w / 2
        y = yc * height - box_h / 2
        area = max(0.0, box_w) * max(0.0, box_h)
        boxes.append({
            "index": idx,
            "class_id": class_id,
            "class_name": COCO_NAMES[class_id] if 0 <= class_id < len(COCO_NAMES) else f"class_{class_id}",
            "x": x,
            "y": y,
            "w": box_w,
            "h": box_h,
            "area": area,
            "area_ratio": area / max(width * height, 1),
            "size": size_bucket(area),
            "yolo": [class_id, xc, yc, bw, bh],
        })
    return boxes


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel)


def build_records(pool: Path, output: Path, copy_images: bool, max_images: int | None) -> tuple[list[dict], dict]:
    images_dir = pool / "images"
    labels_dir = pool / "labels"
    assets_dir = output / "assets" / "images"
    candidates = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    if max_images is not None:
        candidates = candidates[:max_images]

    records = []
    class_counts: Counter[str] = Counter()
    image_object_counts: Counter[int] = Counter()
    size_counts: Counter[str] = Counter()

    for image_path in tqdm(candidates, desc="synthetic images"):
        label_path = labels_dir / f"{image_path.stem}.txt"
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            print(f"skip unreadable image: {image_path} ({exc})")
            continue

        asset_name = image_path.name
        link_or_copy(image_path, assets_dir / asset_name, copy_images)
        boxes = parse_yolo_labels(label_path, width, height)
        for box in boxes:
            class_counts[f"{box['class_id']}:{box['class_name']}"] += 1
            size_counts[box["size"]] += 1
        image_object_counts[len(boxes)] += 1
        records.append({
            "file_name": image_path.name,
            "stem": image_path.stem,
            "image_url": f"assets/images/{asset_name}",
            "label_file": str(label_path),
            "width": width,
            "height": height,
            "object_count": len(boxes),
            "classes": sorted({box["class_name"] for box in boxes}),
            "class_ids": sorted({box["class_id"] for box in boxes}),
            "sizes": sorted({box["size"] for box in boxes}),
            "boxes": boxes,
        })

    stats = {
        "image_count": len(records),
        "box_count": sum(r["object_count"] for r in records),
        "class_distribution": [
            {"class": key.split(":", 1)[1], "class_id": int(key.split(":", 1)[0]), "count": count}
            for key, count in class_counts.most_common()
        ],
        "objects_per_image": [
            {"object_count": count, "image_count": image_count}
            for count, image_count in sorted(image_object_counts.items())
        ],
        "size_distribution": [
            {"size": key, "count": size_counts.get(key, 0)}
            for key in ("small", "medium", "large")
        ],
    }
    return records, stats


def write_data_js(path: Path, payload: dict) -> None:
    path.write_text(
        "window.GEN_VIS_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def write_html(path: Path) -> None:
    path.write_text("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic Pool Viewer</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <aside class="sidebar">
    <div class="brand">
      <h1>Synthetic Pool Viewer</h1>
      <div id="datasetPath" class="muted"></div>
    </div>

    <section>
      <h2>Filters</h2>
      <label>File name
        <input id="fileSearch" type="search" placeholder="r1_hard_000000">
      </label>
      <label>Class filter
        <select id="classFilter"></select>
      </label>
      <label>Class mode
        <select id="classMode">
          <option value="contains">Contains selected class</option>
          <option value="only">Only selected class</option>
        </select>
      </label>
      <div class="inline">
        <label>Min boxes <input id="minBoxes" type="number" min="0" placeholder="0"></label>
        <label>Max boxes <input id="maxBoxes" type="number" min="0" placeholder="any"></label>
      </div>
      <label>Object size
        <select id="sizeFilter">
          <option value="">Any size</option>
          <option value="small">Has small</option>
          <option value="medium">Has medium</option>
          <option value="large">Has large</option>
        </select>
      </label>
      <div class="inline">
        <label>Min area <input id="minArea" type="number" min="0" placeholder="px"></label>
        <label>Max area <input id="maxArea" type="number" min="0" placeholder="px"></label>
      </div>
      <button id="resetFilters">Reset</button>
    </section>

    <section>
      <h2>Display</h2>
      <label class="check"><input id="showBoxes" type="checkbox" checked> Show boxes</label>
      <label class="check"><input id="showLabels" type="checkbox" checked> Show class names</label>
    </section>

    <section>
      <h2>Current Image</h2>
      <div id="imageMeta" class="meta"></div>
      <div id="objectList" class="object-list"></div>
    </section>
  </aside>

  <main>
    <header class="toolbar">
      <div>
        <button id="prevBtn">Previous</button>
        <button id="nextBtn">Next</button>
      </div>
      <div id="position" class="position"></div>
    </header>

    <section class="viewer">
      <canvas id="canvas"></canvas>
    </section>

    <section class="stats">
      <div class="stat-card"><h3>Summary</h3><div id="summaryStats"></div></div>
      <div class="stat-card"><h3>Class Distribution</h3><div id="classStats"></div></div>
      <div class="stat-card"><h3>Objects Per Image</h3><div id="countStats"></div></div>
      <div class="stat-card"><h3>Object Sizes</h3><div id="sizeStats"></div></div>
    </section>

    <section>
      <div class="thumb-head">
        <h2>Thumbnails</h2>
        <span id="filteredCount"></span>
      </div>
      <div id="thumbs" class="thumbs"></div>
    </section>
  </main>

  <script src="data.js"></script>
  <script src="app.js"></script>
</body>
</html>
""", encoding="utf-8")


def write_css(path: Path) -> None:
    path.write_text("""* { box-sizing: border-box; }
body { margin: 0; display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; font-family: Arial, sans-serif; color: #1f2933; background: #f4f6f8; }
.sidebar { height: 100vh; overflow: auto; padding: 16px; background: #ffffff; border-right: 1px solid #d6dde6; }
.brand h1 { margin: 0 0 6px; font-size: 20px; }
.muted { color: #667085; font-size: 12px; word-break: break-all; }
section { margin-top: 18px; }
h2 { margin: 0 0 10px; font-size: 15px; }
h3 { margin: 0 0 8px; font-size: 14px; }
label { display: block; margin: 9px 0; font-size: 13px; color: #344054; }
input, select, button { width: 100%; margin-top: 4px; padding: 8px; border: 1px solid #c8d0da; border-radius: 6px; background: #fff; font-size: 13px; }
button { cursor: pointer; background: #1f2933; color: white; border-color: #1f2933; }
.inline { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.check { display: flex; gap: 8px; align-items: center; }
.check input { width: auto; margin: 0; }
main { min-width: 0; height: 100vh; overflow: auto; }
.toolbar { position: sticky; top: 0; z-index: 3; display: grid; grid-template-columns: 240px 1fr; gap: 12px; align-items: center; padding: 12px 16px; background: rgba(255,255,255,.96); border-bottom: 1px solid #d6dde6; }
.toolbar div:first-child { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.position { font-size: 13px; color: #475467; }
.viewer { padding: 16px; display: flex; justify-content: center; align-items: center; background: #111827; min-height: 560px; }
canvas { max-width: 100%; max-height: 78vh; background: #0b0f16; }
.meta { font-size: 13px; line-height: 1.5; }
.object-list { display: grid; gap: 8px; }
.object-item { border: 1px solid #d6dde6; border-radius: 6px; padding: 8px; font-size: 12px; line-height: 1.45; background: #f8fafc; }
.stats { padding: 16px; display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }
.stat-card { background: white; border: 1px solid #d6dde6; border-radius: 8px; padding: 12px; font-size: 13px; }
.bar-row { display: grid; grid-template-columns: 110px 1fr 48px; gap: 8px; align-items: center; margin: 5px 0; }
.bar { height: 8px; background: #e5eaf0; border-radius: 8px; overflow: hidden; }
.bar span { display: block; height: 100%; background: #2e90fa; }
.thumb-head { display: flex; justify-content: space-between; align-items: center; padding: 0 16px; }
.thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; padding: 12px 16px 24px; }
.thumb { border: 2px solid transparent; border-radius: 8px; background: white; padding: 6px; cursor: pointer; text-align: left; }
.thumb.active { border-color: #2e90fa; }
.thumb img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; border-radius: 5px; background: #111; }
.thumb div { margin-top: 5px; font-size: 11px; color: #344054; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
@media (max-width: 980px) { body { grid-template-columns: 1fr; } .sidebar { height: auto; border-right: 0; border-bottom: 1px solid #d6dde6; } main { height: auto; } .stats { grid-template-columns: 1fr; } }
""", encoding="utf-8")


def write_js(path: Path) -> None:
    path.write_text("""const DATA = window.GEN_VIS_DATA;
let records = DATA.records || [];
let filtered = records.slice();
let current = 0;
let image = new Image();

const els = {
  datasetPath: document.getElementById('datasetPath'),
  fileSearch: document.getElementById('fileSearch'),
  classFilter: document.getElementById('classFilter'),
  classMode: document.getElementById('classMode'),
  minBoxes: document.getElementById('minBoxes'),
  maxBoxes: document.getElementById('maxBoxes'),
  sizeFilter: document.getElementById('sizeFilter'),
  minArea: document.getElementById('minArea'),
  maxArea: document.getElementById('maxArea'),
  resetFilters: document.getElementById('resetFilters'),
  showBoxes: document.getElementById('showBoxes'),
  showLabels: document.getElementById('showLabels'),
  prevBtn: document.getElementById('prevBtn'),
  nextBtn: document.getElementById('nextBtn'),
  position: document.getElementById('position'),
  canvas: document.getElementById('canvas'),
  imageMeta: document.getElementById('imageMeta'),
  objectList: document.getElementById('objectList'),
  thumbs: document.getElementById('thumbs'),
  filteredCount: document.getElementById('filteredCount'),
  summaryStats: document.getElementById('summaryStats'),
  classStats: document.getElementById('classStats'),
  countStats: document.getElementById('countStats'),
  sizeStats: document.getElementById('sizeStats'),
};
const ctx = els.canvas.getContext('2d');

function colorFor(classId) {
  const colors = ['#ff4d4f', '#2e90fa', '#12b76a', '#f79009', '#7a5af8', '#06aed4', '#f04438', '#66c61c'];
  return colors[Math.abs(classId) % colors.length];
}

function uniqueClasses() {
  const map = new Map();
  records.forEach(r => r.boxes.forEach(b => map.set(b.class_id, b.class_name)));
  return [...map.entries()].sort((a, b) => a[0] - b[0]);
}

function initControls() {
  els.datasetPath.textContent = DATA.pool;
  els.classFilter.innerHTML = '<option value="">Any class</option>' + uniqueClasses().map(([id, name]) => `<option value="${id}">${id} - ${name}</option>`).join('');
  ['fileSearch','classFilter','classMode','minBoxes','maxBoxes','sizeFilter','minArea','maxArea'].forEach(id => els[id].addEventListener('input', applyFilters));
  els.showBoxes.addEventListener('change', renderCurrent);
  els.showLabels.addEventListener('change', renderCurrent);
  els.prevBtn.addEventListener('click', () => selectIndex(current - 1));
  els.nextBtn.addEventListener('click', () => selectIndex(current + 1));
  els.resetFilters.addEventListener('click', resetFilters);
}

function passes(record) {
  const q = els.fileSearch.value.trim().toLowerCase();
  if (q && !record.file_name.toLowerCase().includes(q)) return false;
  const classId = els.classFilter.value === '' ? null : Number(els.classFilter.value);
  if (classId !== null) {
    const has = record.boxes.some(b => b.class_id === classId);
    const only = record.boxes.length > 0 && record.boxes.every(b => b.class_id === classId);
    if (els.classMode.value === 'only' ? !only : !has) return false;
  }
  const minBoxes = els.minBoxes.value === '' ? null : Number(els.minBoxes.value);
  const maxBoxes = els.maxBoxes.value === '' ? null : Number(els.maxBoxes.value);
  if (minBoxes !== null && record.object_count < minBoxes) return false;
  if (maxBoxes !== null && record.object_count > maxBoxes) return false;
  const size = els.sizeFilter.value;
  if (size && !record.boxes.some(b => b.size === size)) return false;
  const minArea = els.minArea.value === '' ? null : Number(els.minArea.value);
  const maxArea = els.maxArea.value === '' ? null : Number(els.maxArea.value);
  if (minArea !== null && !record.boxes.some(b => b.area >= minArea)) return false;
  if (maxArea !== null && !record.boxes.some(b => b.area <= maxArea)) return false;
  return true;
}

function applyFilters() {
  filtered = records.filter(passes);
  current = 0;
  renderThumbs();
  renderCurrent();
}

function resetFilters() {
  els.fileSearch.value = '';
  els.classFilter.value = '';
  els.classMode.value = 'contains';
  els.minBoxes.value = '';
  els.maxBoxes.value = '';
  els.sizeFilter.value = '';
  els.minArea.value = '';
  els.maxArea.value = '';
  applyFilters();
}

function selectIndex(index) {
  if (!filtered.length) return;
  current = (index + filtered.length) % filtered.length;
  renderCurrent();
  renderThumbs(false);
}

function renderCurrent() {
  if (!filtered.length) {
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    els.position.textContent = 'No images match current filters';
    els.imageMeta.textContent = '';
    els.objectList.innerHTML = '';
    return;
  }
  const record = filtered[current];
  image.onload = () => draw(record);
  image.src = record.image_url;
  els.position.textContent = `${current + 1} / ${filtered.length} filtered, ${records.length} total`;
  els.imageMeta.innerHTML = `<b>${record.file_name}</b><br>${record.width} x ${record.height}<br>${record.object_count} objects<br>classes: ${record.classes.join(', ') || 'none'}`;
  els.objectList.innerHTML = record.boxes.map(b => `<div class="object-item"><b>${b.class_name}</b> [${b.class_id}] ${b.size}<br>x=${b.x.toFixed(1)}, y=${b.y.toFixed(1)}, w=${b.w.toFixed(1)}, h=${b.h.toFixed(1)}<br>area=${Math.round(b.area)} px, ratio=${(b.area_ratio * 100).toFixed(3)}%</div>`).join('');
}

function draw(record) {
  els.canvas.width = record.width;
  els.canvas.height = record.height;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.drawImage(image, 0, 0);
  if (!els.showBoxes.checked) return;
  ctx.lineWidth = Math.max(2, Math.round(Math.max(record.width, record.height) / 360));
  ctx.font = `${Math.max(14, Math.round(record.width / 45))}px Arial`;
  record.boxes.forEach(b => {
    const color = colorFor(b.class_id);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    if (els.showLabels.checked) {
      const text = `${b.class_name} [${b.class_id}]`;
      const metrics = ctx.measureText(text);
      const th = parseInt(ctx.font, 10) + 8;
      const tx = Math.max(0, b.x);
      const ty = Math.max(th, b.y);
      ctx.fillRect(tx, ty - th, metrics.width + 10, th);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, tx + 5, ty - 6);
    }
  });
}

function renderThumbs(rebuild = true) {
  els.filteredCount.textContent = `${filtered.length} shown`;
  if (rebuild) {
    els.thumbs.innerHTML = filtered.map((r, i) => `<button class="thumb" data-i="${i}"><img src="${r.image_url}" loading="lazy"><div>${r.file_name}</div><div>${r.object_count} boxes</div></button>`).join('');
    els.thumbs.querySelectorAll('.thumb').forEach(btn => btn.addEventListener('click', () => selectIndex(Number(btn.dataset.i))));
  }
  els.thumbs.querySelectorAll('.thumb').forEach((btn, i) => btn.classList.toggle('active', i === current));
}

function barRows(items, labelKey, valueKey, maxValue) {
  return items.map(item => {
    const label = item[labelKey];
    const value = item[valueKey];
    const pct = maxValue ? Math.max(2, (value / maxValue) * 100) : 0;
    return `<div class="bar-row"><span>${label}</span><div class="bar"><span style="width:${pct}%"></span></div><span>${value}</span></div>`;
  }).join('');
}

function renderStats() {
  const stats = DATA.stats;
  els.summaryStats.innerHTML = `images: <b>${stats.image_count}</b><br>boxes: <b>${stats.box_count}</b>`;
  const maxClass = Math.max(1, ...stats.class_distribution.map(x => x.count));
  els.classStats.innerHTML = barRows(stats.class_distribution.slice(0, 20).map(x => ({label: `${x.class_id} ${x.class}`, count: x.count})), 'label', 'count', maxClass);
  const maxCount = Math.max(1, ...stats.objects_per_image.map(x => x.image_count));
  els.countStats.innerHTML = barRows(stats.objects_per_image.map(x => ({label: `${x.object_count} boxes`, count: x.image_count})), 'label', 'count', maxCount);
  const maxSize = Math.max(1, ...stats.size_distribution.map(x => x.count));
  els.sizeStats.innerHTML = barRows(stats.size_distribution.map(x => ({label: x.size, count: x.count})), 'label', 'count', maxSize);
}

initControls();
renderStats();
applyFilters();
""", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML browser for synthetic images and YOLO bbox labels.")
    parser.add_argument("--synthetic-pool", required=True, help="Directory containing images/ and labels/.")
    parser.add_argument("--output-dir", default="outputs/gen_vis")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating relative symlinks.")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = Path(args.synthetic_pool).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, stats = build_records(pool, output, args.copy_images, args.max_images)
    write_data_js(output / "data.js", {"pool": str(pool), "records": records, "stats": stats})
    write_html(output / "index.html")
    write_css(output / "style.css")
    write_js(output / "app.js")
    (output / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"images: {stats['image_count']} boxes: {stats['box_count']}")
    print(f"HTML: {output / 'index.html'}")
    print(f"Serve: cd {output} && python -m http.server 8000")


if __name__ == "__main__":
    main()
