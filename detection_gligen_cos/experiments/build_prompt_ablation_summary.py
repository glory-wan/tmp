from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import yaml


STEP_RE = re.compile(
    r"step=(?P<step>\d+)/(?P<total>\d+).*?"
    r"loss=(?P<loss>-?\d+(?:\.\d+)?) "
    r"semantic=(?P<semantic>-?\d+(?:\.\d+)?) "
    r"class_semantic=(?P<class_semantic>-?\d+(?:\.\d+)?) "
    r"det=(?P<det>-?\d+(?:\.\d+)?)"
)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def experiment_kind(name: str) -> str:
    if name.startswith("quick_steps_"):
        return "steps"
    if name.startswith("quick_dw_"):
        return "detector_weight"
    if name.startswith("quick_csw_"):
        return "class_semantic_weight"
    return "other"


def parse_losses(log_path: Path) -> dict:
    if not log_path.exists():
        return {"status": "missing"}
    matches = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if match:
            item = {k: float(v) for k, v in match.groupdict().items()}
            matches.append(item)
    if not matches:
        return {"status": "no_steps"}
    last = matches[-1]
    tail = matches[-20:]
    avg = {
        key: sum(item[key] for item in tail) / len(tail)
        for key in ("loss", "semantic", "class_semantic", "det")
    }
    return {"status": "ok", "last": last, "tail20_avg": avg, "logged_steps": len(matches)}


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def load_generation_summary(exp_dir: Path) -> dict:
    manifest_path = exp_dir / "round_1" / "synthetic" / "manifest.json"
    if not manifest_path.exists():
        return {"synthetic_images": image_count(exp_dir / "round_1" / "synthetic" / "images"), "manifest_count": 0}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        manifest = []
    edited = [int(x.get("edited_objects", x.get("layout_objects", []).__len__() if isinstance(x.get("layout_objects"), list) else 0)) for x in manifest if isinstance(x, dict)]
    return {
        "synthetic_images": image_count(exp_dir / "round_1" / "synthetic" / "images"),
        "manifest_count": len(manifest) if isinstance(manifest, list) else 0,
        "avg_edited_objects": sum(edited) / len(edited) if edited else None,
    }


def grid_images(root: Path, name: str) -> list[Path]:
    qdir = root / "qual" / name
    if not qdir.exists():
        return []
    return sorted(qdir.glob("**/grid.jpg"))


def collect_experiment(root: Path, exp_dir: Path) -> dict:
    name = exp_dir.name
    cfg = load_yaml(root / "configs" / f"{name}.yaml")
    po = cfg.get("prompt_optimization", {})
    gen = cfg.get("generation", {})
    latest_meta = {}
    meta_path = exp_dir / "round_1" / "prompts" / "object_prompts.json"
    if meta_path.exists():
        try:
            latest_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            latest_meta = {}
    gen_vis = root / "gen_vis" / name / "index.html"
    grids = grid_images(root, name)
    return {
        "name": name,
        "kind": experiment_kind(name),
        "config": rel(root / "configs" / f"{name}.yaml", root) if (root / "configs" / f"{name}.yaml").exists() else None,
        "steps": po.get("max_train_steps"),
        "detector_weight": po.get("detector_weight"),
        "class_semantic_weight": po.get("class_semantic_weight"),
        "semantic_weight": po.get("semantic_weight"),
        "num_new_tokens": po.get("num_new_tokens"),
        "generation_max_images": gen.get("max_images"),
        "generation": load_generation_summary(exp_dir),
        "losses": parse_losses(exp_dir / "round_1" / "prompts" / "pipeline.log"),
        "latest_checkpoint": latest_meta.get("latest"),
        "gen_vis": rel(gen_vis, root) if gen_vis.exists() else None,
        "prompt_log": rel(exp_dir / "round_1" / "prompts" / "pipeline.log", root) if (exp_dir / "round_1" / "prompts" / "pipeline.log").exists() else None,
        "generate_log": rel(exp_dir / "logs" / "round_1_generate.log", root) if (exp_dir / "logs" / "round_1_generate.log").exists() else None,
        "grids": [{"class": p.parent.name, "path": rel(p, root)} for p in grids],
    }


def discover(root: Path) -> list[dict]:
    experiments = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("quick_"):
            continue
        if not (path / "round_1").exists():
            continue
        experiments.append(collect_experiment(root, path))
    order = {"steps": 0, "detector_weight": 1, "class_semantic_weight": 2, "other": 3}
    return sorted(experiments, key=lambda x: (order.get(x["kind"], 99), str(x.get("steps")), str(x.get("detector_weight")), str(x.get("class_semantic_weight")), x["name"]))


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def stat_cell(exp: dict) -> str:
    losses = exp["losses"]
    if losses.get("status") != "ok":
        return html.escape(losses.get("status", "missing"))
    last = losses["last"]
    avg = losses["tail20_avg"]
    return (
        f"last loss {fmt(last['loss'])}<br>"
        f"last semantic {fmt(last['semantic'])}<br>"
        f"last class_sem {fmt(last['class_semantic'])}<br>"
        f"last det {fmt(last['det'])}<br>"
        f"tail20 loss {fmt(avg['loss'])}"
    )


def card(exp: dict) -> str:
    chips = [
        f"steps={exp.get('steps')}",
        f"dw={exp.get('detector_weight')}",
        f"csw={exp.get('class_semantic_weight')}",
        f"syn={exp['generation'].get('synthetic_images', 0)}",
    ]
    links = []
    if exp.get("gen_vis"):
        links.append(f'<a href="{html.escape(exp["gen_vis"])}" target="_blank">Synthetic viewer</a>')
    if exp.get("config"):
        links.append(f'<a href="{html.escape(exp["config"])}" target="_blank">Config</a>')
    if exp.get("prompt_log"):
        links.append(f'<a href="{html.escape(exp["prompt_log"])}" target="_blank">Prompt log</a>')
    if exp.get("generate_log"):
        links.append(f'<a href="{html.escape(exp["generate_log"])}" target="_blank">Generate log</a>')
    grids = "\n".join(
        f'<figure><figcaption>{html.escape(item["class"])}</figcaption><a href="{html.escape(item["path"])}" target="_blank"><img src="{html.escape(item["path"])}" loading="lazy"></a></figure>'
        for item in exp["grids"]
    )
    return f"""
    <article class="card" data-kind="{html.escape(exp['kind'])}" data-name="{html.escape(exp['name'])}">
      <div class="card-head">
        <h3>{html.escape(exp['name'])}</h3>
        <div class="kind">{html.escape(exp['kind'])}</div>
      </div>
      <div class="chips">{''.join(f'<span>{html.escape(c)}</span>' for c in chips)}</div>
      <div class="metrics">
        <div>{stat_cell(exp)}</div>
        <div>
          latest {html.escape(str(exp.get('latest_checkpoint') or '-'))}<br>
          manifest {fmt(exp['generation'].get('manifest_count'))}<br>
          avg edited {fmt(exp['generation'].get('avg_edited_objects'), 2)}
        </div>
      </div>
      <div class="links">{' '.join(links)}</div>
      <div class="grids">{grids}</div>
    </article>
    """


def table_rows(experiments: list[dict]) -> str:
    rows = []
    for exp in experiments:
        losses = exp["losses"]
        last = losses.get("last", {}) if losses.get("status") == "ok" else {}
        rows.append(f"""
        <tr data-kind="{html.escape(exp['kind'])}">
          <td>{html.escape(exp['name'])}</td>
          <td>{html.escape(exp['kind'])}</td>
          <td>{fmt(exp.get('steps'))}</td>
          <td>{fmt(exp.get('detector_weight'))}</td>
          <td>{fmt(exp.get('class_semantic_weight'))}</td>
          <td>{fmt(last.get('loss'))}</td>
          <td>{fmt(last.get('semantic'))}</td>
          <td>{fmt(last.get('class_semantic'))}</td>
          <td>{fmt(last.get('det'))}</td>
          <td>{fmt(exp['generation'].get('synthetic_images'))}</td>
         
          <td>{'<a href="' + html.escape(exp['gen_vis']) + '" target="_blank">open</a>' if exp.get('gen_vis') else '-'}</td>
        </tr>
        """)
    return "\n".join(rows)


def write_html(root: Path, experiments: list[dict]) -> Path:
    counts = {}
    for exp in experiments:
        counts[exp["kind"]] = counts.get(exp["kind"], 0) + 1
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt Ablation Summary</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fa; color: #1d2733; }}
    header {{ position: sticky; top: 0; z-index: 3; background: white; border-bottom: 1px solid #d7dee8; padding: 14px 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .muted {{ color: #667085; font-size: 13px; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    button {{ border: 1px solid #b9c4d0; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
    button.active {{ background: #1d2733; color: white; border-color: #1d2733; }}
    main {{ padding: 16px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .stat {{ background: white; border: 1px solid #d7dee8; border-radius: 8px; padding: 12px; }}
    .stat b {{ font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee8; border-radius: 8px; overflow: hidden; margin-bottom: 18px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e9f0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; position: sticky; top: 92px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d7dee8; border-radius: 8px; padding: 12px; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
    .card h3 {{ margin: 0; font-size: 17px; }}
    .kind {{ color: #475467; font-size: 13px; }}
    .chips {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }}
    .chips span {{ background: #eef2f6; border: 1px solid #d7dee8; padding: 4px 7px; border-radius: 999px; font-size: 12px; }}
    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px; line-height: 1.45; }}
    .links {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; font-size: 13px; }}
    .grids {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #475467; margin-bottom: 4px; }}
    figure img {{ width: 100%; display: block; border: 1px solid #d7dee8; border-radius: 6px; background: #111; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
<header>
  <h1>Prompt Ablation Summary</h1>
  <div class="muted">{html.escape(str(root))}</div>
  <div class="controls">
    <button class="active" data-filter="all">All ({len(experiments)})</button>
    <button data-filter="steps">Steps ({counts.get('steps', 0)})</button>
    <button data-filter="detector_weight">Detector weight ({counts.get('detector_weight', 0)})</button>
    <button data-filter="class_semantic_weight">Class semantic weight ({counts.get('class_semantic_weight', 0)})</button>
  </div>
</header>
<main>
  <section class="summary">
    <div class="stat"><b>{len(experiments)}</b><br>experiments</div>
    <div class="stat"><b>{sum(exp['generation'].get('synthetic_images', 0) for exp in experiments)}</b><br>synthetic images</div>
    <div class="stat"><b>{sum(len(exp['grids']) for exp in experiments)}</b><br>prompt grids</div>
  </section>
  <table>
    <thead><tr><th>Name</th><th>Kind</th><th>Steps</th><th>DW</th><th>CSW</th><th>Loss</th><th>Semantic</th><th>Class sem</th><th>Det</th><th>Images</th><th>Viewer</th></tr></thead>
    <tbody>{table_rows(experiments)}</tbody>
  </table>
  <section class="cards">{''.join(card(exp) for exp in experiments)}</section>
</main>
<script>
const buttons = document.querySelectorAll('button[data-filter]');
const cards = document.querySelectorAll('.card');
const rows = document.querySelectorAll('tbody tr');
buttons.forEach(btn => btn.addEventListener('click', () => {{
  buttons.forEach(x => x.classList.remove('active'));
  btn.classList.add('active');
  const f = btn.dataset.filter;
  [...cards, ...rows].forEach(el => {{
    const show = f === 'all' || el.dataset.kind === f;
    el.classList.toggle('hidden', !show);
  }});
}}));
</script>
</body>
</html>
"""
    out = root / "index.html"
    out.write_text(body, encoding="utf-8")
    (root / "summary_data.json").write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single HTML summary for prompt ablation quick experiments.")
    parser.add_argument("--root", required=True, help="Ablation output root, e.g. outputs/prompt_ablation/prompt_ablation_0729_ab")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    experiments = discover(root)
    if not experiments:
        raise RuntimeError(f"no quick_* experiments found under {root}")
    out = write_html(root, experiments)
    print(f"experiments: {len(experiments)}")
    print(f"HTML: {out}")


if __name__ == "__main__":
    main()
