# GLIGEN的第三个版本。
# 新增参数：
# --global-prompt-source template|coco_caption|category_count
# template：保持原来的 {phrases}, photo, highly detailed, photorealistic
# coco_caption：优先使用 COCO caption，没有 caption 时回退 template
# category_count：基于最终实际进入 GLIGEN 的 layout_items 统计类别数量生成 global prompt

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from detection_gligen_sdedit.common import CocoMini, load_failures, parse_class_subset, setup_logger, subset_label
from detection_gligen_sdedit.experiments.prompt_vis.utils import (
    clean_output_dir,
    load_object_prompts_checkpoint,
    rel,
)
from detection_gligen_sdedit.generate_hard_examples import build_object_prompt, choose_group, install_embeddings
from detection_gligen_sdedit.generate_gligen_layout import build_global_prompt, prepare_gligen_model_for_diffusers, xywh_to_gligen_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLIGEN caption+layout-conditioned visualization for learned object prompts.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--captions", default=None,
                        help="Optional COCO captions json. If provided, each scene uses the image caption as global prompt.")
    parser.add_argument("--caption-selection", choices=["first", "random"], default="first",
                        help="How to select one caption when an image has multiple captions. random is seeded and reproducible.")
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
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.0,
                        help="Filter out GT boxes whose bbox area divided by image area is below this value.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--gligen-scheduled-sampling-beta", type=float, default=0.3)
    parser.add_argument("--global-prompt-source", choices=["template", "coco_caption", "category_count"], default="template",
                        help="Global prompt source: template phrases, COCO caption, or category counts from final layout objects.")
    parser.add_argument("--global-prompt-template", default="{phrases}, photo, highly detailed, photorealistic")
    parser.add_argument("--negative-prompt", default="wrong class, duplicate object, deformed, cropped, text, watermark")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    return parser.parse_args()


def load_caption_map(path: str | None) -> dict[int, list[str]]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    captions: dict[int, list[str]] = defaultdict(list)
    for ann in data.get("annotations", []):
        caption = str(ann.get("caption", "")).strip()
        if caption:
            captions[int(ann["image_id"])].append(caption)
    return dict(captions)


def select_caption(captions_by_image: dict[int, list[str]], image_id: int, selection: str, seed: int) -> str | None:
    captions = captions_by_image.get(int(image_id), [])
    if not captions:
        return None
    if selection == "random":
        return random.Random(seed + int(image_id)).choice(captions)
    return captions[0]


def article_for_class_name(class_name: str) -> str:
    return "an" if class_name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def number_word(count: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(count, str(count))


def pluralize_class_name(class_name: str) -> str:
    if class_name.endswith("person"):
        return class_name[:-6] + "persons"
    if class_name.endswith(("bus", "class", "glass")):
        return class_name + "es"
    if class_name.endswith("y") and len(class_name) > 1 and class_name[-2].lower() not in {"a", "e", "i", "o", "u"}:
        return class_name[:-1] + "ies"
    if class_name.endswith(("s", "x", "z", "ch", "sh")):
        return class_name + "es"
    return class_name + "s"


def join_natural_language(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def build_category_count_prompt(layout_items: list[dict]) -> str:
    counts = Counter(item["class_name"] for item in layout_items)
    phrases = []
    for class_name in sorted(counts):
        count = counts[class_name]
        if count == 1:
            phrases.append(f"{article_for_class_name(class_name)} {class_name}")
        else:
            phrases.append(f"{number_word(count)} {pluralize_class_name(class_name)}")
    objects = join_natural_language(phrases) or "objects"
    # return f"An image with {objects}, photo, highly detailed, photorealistic"
    return f"An photo with {objects}"

def choose_global_prompt(phrases: list[str], layout_items: list[dict], image_id: int, args: argparse.Namespace,
                         captions_by_image: dict[int, list[str]]) -> tuple[str, str, str | None, str]:
    template_prompt = build_global_prompt(phrases, args.global_prompt_template)
    category_count_prompt = build_category_count_prompt(layout_items)
    caption = select_caption(captions_by_image, image_id, args.caption_selection, args.seed)
    if args.global_prompt_source == "coco_caption" and caption:
        return caption, "coco_caption", caption, category_count_prompt
    if args.global_prompt_source == "coco_caption":
        return template_prompt, "template_fallback_no_caption", None, category_count_prompt
    if args.global_prompt_source == "category_count":
        return category_count_prompt, "category_count", caption, category_count_prompt
    return template_prompt, "template", caption, category_count_prompt


def bbox_key(bbox) -> tuple[float, float, float, float]:
    return tuple(round(float(x), 3) for x in bbox)


def failure_lookup(failures: list[dict]) -> tuple[dict[int, dict], dict[tuple[int, tuple[float, float, float, float]], dict]]:
    by_ann = {}
    by_class_bbox = {}
    for failure in failures:
        if failure.get("annotation_id") is not None:
            by_ann[int(failure["annotation_id"])] = failure
        by_class_bbox[(int(failure["class_id"]), bbox_key(failure["bbox"]))] = failure
    return by_ann, by_class_bbox


def match_failure(ann: dict, failures_by_ann: dict[int, dict],
                  failures_by_class_bbox: dict[tuple[int, tuple[float, float, float, float]], dict]) -> dict | None:
    ann_id = ann.get("id")
    if ann_id is not None and int(ann_id) in failures_by_ann:
        return failures_by_ann[int(ann_id)]
    return failures_by_class_bbox.get((int(ann["class_id"]), bbox_key(ann["bbox"])))


def ordered_failure_aware_gt(anns: list[dict], failures: list[dict]) -> list[tuple[dict, dict | None]]:
    failures_by_ann, failures_by_class_bbox = failure_lookup(failures)
    annotated = [(ann, match_failure(ann, failures_by_ann, failures_by_class_bbox)) for ann in anns]
    failures_first = [item for item in annotated if item[1] is not None]
    normal = [item for item in annotated if item[1] is None]
    return failures_first + normal


def bbox_area_ratio(ann: dict, info: dict) -> float:
    _, _, width, height = [float(v) for v in ann["bbox"]]
    image_area = max(float(info["width"]) * float(info["height"]), 1.0)
    return max(width, 0.0) * max(height, 0.0) / image_area


def filter_small_gt_items(items: list[tuple[dict, dict | None]], info: dict,
                          min_bbox_area_ratio: float) -> tuple[list[tuple[dict, dict | None]], list[dict]]:
    if min_bbox_area_ratio <= 0:
        return items, []
    kept = []
    filtered = []
    for ann, failure in items:
        ratio = bbox_area_ratio(ann, info)
        if ratio >= min_bbox_area_ratio:
            kept.append((ann, failure))
        else:
            filtered.append({
                "annotation_id": ann.get("id"),
                "bbox": ann["bbox"],
                "bbox_area_ratio": ratio,
                "class_id": int(ann["class_id"]),
                "layout_source": "failure" if failure else "normal_gt",
                "filter_reason": "small_bbox",
            })
    return kept, filtered


def phrase_for_gt(meta: dict, dataset: CocoMini, ann: dict,
                  prompt_scope: str, include_class_name: bool | None) -> tuple[str, str | None, str]:
    class_id = int(ann["class_id"])
    class_name = dataset.class_names[class_id]
    group = choose_group(meta["groups"], class_id, prompt_scope)
    if group in meta["groups"]:
        use_class = bool(meta.get("include_class_name", False)) if include_class_name is None else bool(include_class_name)
        return build_object_prompt(class_name, meta["groups"][group], use_class), group, "learned_prompt"
    return class_name, None, "class_name_fallback"


def draw_layout(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    palette = [(255, 64, 64), (40, 160, 90), (65, 105, 225), (240, 160, 30), (155, 80, 200)]
    for idx, item in enumerate(items):
        color = palette[idx % len(palette)]
        x, y, w, h = [float(v) for v in item["bbox"]]
        line = max(2, int(round(max(out.size) / 300)))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line)
        label = f"{item['class_name']} | {item['layout_source']} | {item['phrase_source']}"
        draw.text((x + 4, max(0, y - 18)), label, fill=color)
    return out


def draw_gligen_layout(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    palette = [(255, 64, 64), (40, 160, 90), (65, 105, 225), (240, 160, 30), (155, 80, 200)]
    width, height = out.size
    for idx, item in enumerate(items):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = [float(v) for v in item["gligen_box"]]
        left = x1 * width
        top = y1 * height
        right = x2 * width
        bottom = y2 * height
        line = max(2, int(round(max(out.size) / 300)))
        draw.rectangle([left, top, right, bottom], outline=color, width=line)
        label = f"{item['class_name']} | {item['layout_source']} | {item['phrase_source']}"
        draw.text((left + 4, max(0, top - 18)), label, fill=color)
    return out


def write_caption_html(path: Path, title: str, manifest: list[dict]) -> None:
    cards = []
    for item in manifest:
        figures = []
        for key, label in (
            ("original_file", "Original"),
            ("mask_file", "Mask/Bbox"),
            ("result_file", "Result"),
            ("result_overlay_file", "Bbox+Result"),
        ):
            if item.get(key):
                figures.append(f"""
                <figure>
                  <figcaption>{label}</figcaption>
                  <img src="{html.escape(item[key])}" loading="lazy">
                </figure>
                """)
        object_rows = []
        for idx, obj in enumerate(item.get("layout_objects", []), start=1):
            source_class = "failure" if obj.get("layout_source") == "failure" else "normal"
            phrase_class = "learned" if obj.get("phrase_source") == "learned_prompt" else "fallback"
            object_rows.append(f"""
            <tr>
              <td>{idx}</td>
              <td><span class="pill {source_class}">{html.escape(str(obj.get('layout_source', '')))}</span></td>
              <td>{html.escape(str(obj.get('class_name', '')))}</td>
              <td><span class="pill {phrase_class}">{html.escape(str(obj.get('phrase_source', '')))}</span></td>
              <td class="mono">{html.escape(str(obj.get('bbox')))}</td>
              <td class="mono">{html.escape(str(obj.get('gligen_box')))}</td>
              <td>{html.escape(str(obj.get('phrase', '')))}</td>
            </tr>
            """)
        cards.append(f"""
        <article class="card">
          <section class="summary">
            <div>
              <h2>Image {item.get('image_id')}</h2>
              <p class="classes">{html.escape(str(item.get('class_name', '')))}</p>
            </div>
            <div class="stats">
              <span>objects <b>{len(item.get('layout_objects', []))}</b></span>
              <span>GT <b>{item.get('gt_annotations_total')}</b></span>
              <span>filtered <b>{item.get('filtered_small_bbox_count')}</b></span>
              <span>failure <b>{item.get('failure_layout_count')}</b></span>
              <span>normal <b>{item.get('normal_layout_count')}</b></span>
            </div>
          </section>
          <section class="prompt-panel">
            <div class="prompt-main">
              <span class="label">used global prompt</span>
              <span class="source-tag">{html.escape(str(item.get('global_prompt_source', '')))}</span>
              <p>{html.escape(str(item.get('global_prompt', item.get('prompt', ''))))}</p>
            </div>
            <div class="prompt-grid">
              <div><span class="label">category count</span><p>{html.escape(str(item.get('category_count_prompt') or ''))}</p></div>
              <div><span class="label">COCO caption</span><p>{html.escape(str(item.get('coco_caption') or ''))}</p></div>
              <div><span class="label">template</span><p>{html.escape(str(item.get('template_global_prompt') or ''))}</p></div>
            </div>
          </section>
          <div class="media">{''.join(figures)}</div>
          <section class="objects">
            <div class="table-wrap">
              <table>
                <thead><tr><th>#</th><th>source</th><th>class</th><th>phrase source</th><th>bbox</th><th>gligen box</th><th>phrase</th></tr></thead>
                <tbody>{''.join(object_rows)}</tbody>
              </table>
            </div>
          </section>
        </article>
        """)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --line: #d7dee8; --text: #1d2733; --muted: #667085; --panel: #ffffff; --soft: #f3f6fb; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, sans-serif; background: #eef2f7; color: var(--text); }}
header {{ position: sticky; top: 0; z-index: 2; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--line); padding: 16px 22px; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
h2 {{ margin: 0; font-size: 18px; }}
.muted {{ color: var(--muted); font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; padding: 18px; max-width: 1500px; margin: 0 auto; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.summary {{ display: flex; justify-content: space-between; gap: 16px; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
.classes {{ margin: 5px 0 0; color: var(--muted); font-size: 13px; }}
.stats {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; align-content: center; }}
.stats span {{ background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; font-size: 12px; color: #344054; }}
.prompt-panel {{ display: grid; grid-template-columns: minmax(260px, .9fr) minmax(360px, 1.4fr); gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--line); background: #fbfcfe; }}
.prompt-main, .prompt-grid > div {{ border: 1px solid var(--line); border-radius: 8px; background: white; padding: 10px 12px; }}
.prompt-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
.label {{ display: inline-block; margin-bottom: 6px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
.source-tag {{ display: inline-block; margin-left: 8px; background: #e0f2fe; color: #075985; border-radius: 999px; padding: 2px 8px; font-size: 12px; }}
p {{ margin: 0; line-height: 1.45; font-size: 13px; }}
.media {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 12px; }}
figure {{ margin: 0; }}
figcaption {{ color: #475467; font-size: 12px; margin-bottom: 5px; }}
img {{ width: 100%; display: block; border: 1px solid var(--line); border-radius: 6px; background: #111; }}
.objects {{ padding: 0 12px 14px; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 8px 9px; border-bottom: 1px solid #e4e9f0; vertical-align: top; text-align: left; }}
th {{ background: #f8fafc; color: #475467; font-weight: 600; }}
tr:last-child td {{ border-bottom: 0; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #475467; white-space: nowrap; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 2px 7px; font-size: 11px; white-space: nowrap; }}
.pill.failure {{ background: #fee4e2; color: #b42318; }}
.pill.normal {{ background: #e0f2fe; color: #026aa2; }}
.pill.learned {{ background: #dcfae6; color: #067647; }}
.pill.fallback {{ background: #f2f4f7; color: #475467; }}
@media (max-width: 1100px) {{ .prompt-panel, .prompt-grid, .media {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 760px) {{ .summary, .prompt-panel, .prompt-grid, .media {{ grid-template-columns: 1fr; display: grid; }} .stats {{ justify-content: flex-start; }} }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="muted">mode=gligen_category_count_layout | items={len(manifest)}</div>
</header>
<main class="grid">
{''.join(cards)}
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


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
    logger.info("GLIGEN prompt visualization start: scenes=%d failures=%d class_subset=%s",
                len(scenes), len(failures), subset_label(class_subset))
    logger.info("global prompt source: source=%s captions=%s images_with_captions=%d selection=%s",
                args.global_prompt_source, args.captions, len(captions_by_image), args.caption_selection)

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
    for scene_idx, (image_id, failure_items) in enumerate(tqdm(scenes, desc="gligen prompt scenes"), start=1):
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
            prompt, group, phrase_source = phrase_for_gt(meta, dataset, ann, prompt_scope, args.include_class_name)
            box = xywh_to_gligen_box(ann["bbox"], float(info["width"]), float(info["height"]))
            phrases.append(prompt)
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
                "phrase": prompt,
            })
        if not phrases:
            skipped += 1
            continue

        prompt, global_prompt_source, coco_caption, category_count_prompt = choose_global_prompt(
            phrases, layout_items, int(image_id), args, captions_by_image
        )
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
            "original_file": rel(original_file, output),
            "mask_file": rel(mask_file, output),
            "result_file": rel(result_file, output),
            "result_overlay_file": rel(result_overlay_file, output),
        })
        if scene_idx == 1 or scene_idx % 10 == 0 or scene_idx == len(scenes):
            logger.info("GLIGEN progress: processed=%d/%d rendered=%d skipped=%d", scene_idx, len(scenes), len(manifest), skipped)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_caption_html(output / "index.html", "GLIGEN Category Count Layout Prompt Visualization", manifest)
    logger.info("GLIGEN prompt visualization complete: rendered=%d skipped=%d html=%s", len(manifest), skipped, output / "index.html")


if __name__ == "__main__":
    main()
