from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path


STEP_RE = re.compile(r"(?:round_\d+_step_|learned_embeds-)(\d+)")
CLASS_RE = re.compile(r"class_(\d+)$")


COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML grid comparing class grid.jpg outputs across embedding steps."
    )
    parser.add_argument(
        "--qual-dir",
        default="outputs/prompt_ablation/detection_gligen_sdedit/prompt_ablation_0807_30class/qual",
        help="Directory containing round_*_step_*/learned_embeds-*/class_*/grid.jpg outputs.",
    )
    parser.add_argument("--output", default=None, help="HTML output path. Defaults to QUAL_DIR/index_embedding_steps.html.")
    parser.add_argument("--title", default="Embedding Step Qualitative Comparison")
    parser.add_argument("--class-ids", default=None, help="Optional comma-separated class ids to include.")
    parser.add_argument("--sort-classes", choices=["id", "name"], default="id")
    parser.add_argument("--image-name", default="grid.jpg")
    return parser.parse_args()


def extract_step(path: Path) -> int | None:
    for part in path.parts:
        match = STEP_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def extract_class_id(path: Path) -> int | None:
    for parent in [path.parent, *path.parents]:
        match = CLASS_RE.match(parent.name)
        if match:
            return int(match.group(1))
    return None


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_id]
    return f"class_{class_id}"


def scan_qual_dir(qual_dir: Path, image_name: str) -> tuple[dict[int, dict[int, Path]], list[int], list[int]]:
    by_class: dict[int, dict[int, Path]] = defaultdict(dict)
    for image_path in sorted(qual_dir.glob(f"round_*_step_*/learned_embeds-*/class_*/{image_name}")):
        step = extract_step(image_path)
        class_id = extract_class_id(image_path)
        if step is None or class_id is None:
            continue
        by_class[class_id][step] = image_path
    steps = sorted({step for values in by_class.values() for step in values})
    classes = sorted(by_class)
    return dict(by_class), classes, steps


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_html(qual_dir: Path, title: str, by_class: dict[int, dict[int, Path]],
               classes: list[int], steps: list[int]) -> str:
    nav = []
    for class_id in classes:
        nav.append(
            f'<a href="#class-{class_id}">class {class_id} · {html.escape(class_name(class_id))}</a>'
        )

    cards = []
    for class_id in classes:
        cells = []
        for step in steps:
            image_path = by_class.get(class_id, {}).get(step)
            if image_path is None:
                cells.append(
                    f"""
                    <section class="cell missing">
                      <div class="step">step {step}</div>
                      <div class="missing-text">missing</div>
                    </section>
                    """
                )
                continue
            image_rel = html.escape(rel(image_path, qual_dir))
            cells.append(
                f"""
                <section class="cell">
                  <div class="step">step {step}</div>
                  <a href="{image_rel}" target="_blank" rel="noreferrer">
                    <img src="{image_rel}" loading="lazy">
                  </a>
                </section>
                """
            )
        cards.append(
            f"""
            <article class="class-card" id="class-{class_id}" data-class-id="{class_id}" data-class-name="{html.escape(class_name(class_id))}">
              <header class="class-head">
                <div>
                  <h2>class {class_id}</h2>
                  <p>{html.escape(class_name(class_id))}</p>
                </div>
                <span>{sum(1 for step in steps if step in by_class.get(class_id, {}))}/{len(steps)} steps</span>
              </header>
              <div class="step-grid" style="grid-template-columns: repeat({max(len(steps), 1)}, minmax(320px, 1fr));">
                {''.join(cells)}
              </div>
            </article>
            """
        )

    manifest = {
        "qual_dir": str(qual_dir.resolve()),
        "steps": steps,
        "classes": [{"class_id": class_id, "class_name": class_name(class_id)} for class_id in classes],
    }

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --line: #d7dee8; --text: #1d2733; --muted: #667085; --panel: #fff; --soft: #f3f6fb; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, sans-serif; background: #eef2f7; color: var(--text); }}
.top {{ position: sticky; top: 0; z-index: 5; background: rgba(255,255,255,.97); border-bottom: 1px solid var(--line); }}
.top-inner {{ padding: 14px 18px; display: grid; gap: 10px; }}
h1 {{ margin: 0; font-size: 22px; }}
h2 {{ margin: 0; font-size: 18px; }}
p {{ margin: 0; color: var(--muted); font-size: 13px; }}
.toolbar {{ display: grid; grid-template-columns: minmax(220px, 360px) minmax(0, 1fr); gap: 12px; align-items: start; }}
input {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; font-size: 14px; }}
.nav {{ display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px; }}
.nav a {{ white-space: nowrap; text-decoration: none; color: #344054; background: var(--soft); border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
.meta {{ color: var(--muted); font-size: 12px; }}
main {{ max-width: 1800px; margin: 0 auto; padding: 18px; display: grid; gap: 18px; }}
.class-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
.class-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 12px 14px; border-bottom: 1px solid var(--line); }}
.class-head span {{ background: var(--soft); border: 1px solid var(--line); border-radius: 999px; padding: 5px 9px; font-size: 12px; color: #344054; }}
.step-grid {{ display: grid; gap: 12px; padding: 12px; overflow-x: auto; }}
.cell {{ min-width: 320px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; overflow: hidden; }}
.step {{ padding: 8px 10px; font-size: 13px; font-weight: 700; border-bottom: 1px solid var(--line); background: white; }}
img {{ width: 100%; display: block; background: #111; }}
.missing {{ display: grid; grid-template-rows: auto 1fr; min-height: 260px; }}
.missing-text {{ display: grid; place-items: center; color: var(--muted); font-size: 14px; min-height: 220px; }}
@media (max-width: 760px) {{ .toolbar {{ grid-template-columns: 1fr; }} main {{ padding: 10px; }} }}
</style>
</head>
<body>
<header class="top">
  <div class="top-inner">
    <div>
      <h1>{html.escape(title)}</h1>
      <p>classes={len(classes)} · steps={', '.join(str(x) for x in steps)} · source={html.escape(str(qual_dir))}</p>
    </div>
    <div class="toolbar">
      <input id="filter" placeholder="Filter class id or name, e.g. 14 or bird">
      <nav class="nav">{''.join(nav)}</nav>
    </div>
    <div class="meta">Each row compares the same class across embedding checkpoints. Click any grid to open the source image.</div>
  </div>
</header>
<main id="content">
{''.join(cards)}
</main>
<script type="application/json" id="manifest">{html.escape(json.dumps(manifest, ensure_ascii=False))}</script>
<script>
const input = document.getElementById('filter');
const cards = Array.from(document.querySelectorAll('.class-card'));
input.addEventListener('input', () => {{
  const q = input.value.trim().toLowerCase();
  for (const card of cards) {{
    const id = card.dataset.classId || '';
    const name = (card.dataset.className || '').toLowerCase();
    card.style.display = (!q || id.includes(q) || name.includes(q)) ? '' : 'none';
  }}
}});
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    qual_dir = Path(args.qual_dir).resolve()
    output = Path(args.output).resolve() if args.output else qual_dir / "index_embedding_steps.html"
    by_class, classes, steps = scan_qual_dir(qual_dir, args.image_name)
    if args.class_ids:
        keep = {int(x) for x in args.class_ids.split(",") if x.strip()}
        classes = [class_id for class_id in classes if class_id in keep]
    if args.sort_classes == "name":
        classes = sorted(classes, key=lambda class_id: (class_name(class_id), class_id))
    if not by_class or not steps:
        raise FileNotFoundError(f"No {args.image_name} files found under {qual_dir}/round_*_step_*/learned_embeds-*/class_*/")
    html_text = build_html(qual_dir, args.title, by_class, classes, steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(f"HTML written: {output}")
    print(f"classes={len(classes)} steps={steps}")


if __name__ == "__main__":
    main()
