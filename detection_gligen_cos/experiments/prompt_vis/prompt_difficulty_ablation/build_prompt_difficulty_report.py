from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


DEFAULT_CONDITIONS = ["source_original", "class", "learned", "class_learned"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector-centric prompt difficulty ablation report.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--max-grid-images", type=int, default=80)
    parser.add_argument("--max-feature-points", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def load_feature_sets(root: Path, conditions: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_parts = []
    condition_parts = []
    class_parts = []
    failure_parts = []
    expected_dim = None
    for condition in conditions:
        path = root / "analysis" / condition / "backbone_roi_features.npz"
        if not path.exists():
            continue
        data = np.load(path)
        features = data["features"].astype(np.float32)
        if not len(features):
            continue
        expected_dim = features.shape[1] if expected_dim is None else expected_dim
        if features.shape[1] != expected_dim:
            raise ValueError(f"Feature dimension mismatch in {path}: {features.shape[1]} != {expected_dim}")
        feature_parts.append(features)
        condition_parts.append(np.asarray([condition] * len(features)))
        class_parts.append(data["class_id"])
        failure_parts.append(data["false_negative"])
    if not feature_parts:
        return np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0)
    return (
        np.concatenate(feature_parts),
        np.concatenate(condition_parts),
        np.concatenate(class_parts),
        np.concatenate(failure_parts),
    )


def standardized_pca(features: np.ndarray, dimensions: int = 32) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    count = max(1, min(dimensions, scaled.shape[0] - 1, scaled.shape[1]))
    return PCA(n_components=count, random_state=0).fit_transform(scaled)


def mean_nearest_distance(points: np.ndarray, reference: np.ndarray) -> float | None:
    if not len(points) or not len(reference):
        return None
    chunks = []
    for start in range(0, len(points), 256):
        block = points[start:start + 256]
        squared = ((block[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.sqrt(squared.min(axis=1)))
    return float(np.concatenate(chunks).mean())


def compute_alignment(pca_features: np.ndarray, labels: np.ndarray, failures: np.ndarray,
                      conditions: list[str]) -> dict[str, dict]:
    reference_failure = pca_features[(labels == "source_original") & (failures == 1)]
    reference_detected = pca_features[(labels == "source_original") & (failures == 0)]
    rows = {}
    for condition in conditions:
        if condition == "source_original":
            continue
        points = pca_features[labels == condition]
        failure_distance = mean_nearest_distance(points, reference_failure)
        detected_distance = mean_nearest_distance(points, reference_detected)
        rows[condition] = {
            "points": int(len(points)),
            "distance_to_source_failure": failure_distance,
            "distance_to_source_detected": detected_distance,
            "failure_affinity": (
                detected_distance - failure_distance
                if failure_distance is not None and detected_distance is not None else None
            ),
        }
    return rows


def save_embedding_plot(features: np.ndarray, conditions: np.ndarray, failures: np.ndarray,
                        output: Path, max_points: int, seed: int) -> str | None:
    if len(features) < 3:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    selected = features[indices]
    selected_conditions = conditions[indices]
    selected_failures = failures[indices]
    perplexity = max(2, min(30, (len(selected) - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(selected)
    palette = {
        "source_original": "#111827",
        "class": "#2563eb",
        "learned": "#dc2626",
        "class_learned": "#059669",
    }
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    for condition in sorted(set(selected_conditions.tolist())):
        mask = selected_conditions == condition
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], s=13,
            c=palette.get(condition, "#7c3aed"), alpha=0.55, label=condition,
        )
    failure_mask = selected_failures == 1
    ax.scatter(
        embedding[failure_mask, 0], embedding[failure_mask, 1],
        s=30, facecolors="none", edgecolors="#f59e0b", linewidths=0.7, label="detector FN",
    )
    ax.set_title("YOLOv7 backbone ROI feature t-SNE")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output / "feature_tsne.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def metric_value(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def find_posttrain_metrics(root: Path, condition: str):
    candidates = [
        root / "posttrain" / condition / "metrics.json",
        root / "finetune" / condition / "evaluation" / "metrics.json",
    ]
    return next((load_json(path) for path in candidates if path.exists()), None)


def build_html(root: Path, conditions: list[str], detector_metrics: dict, alignment: dict,
               plot_name: str | None, baseline: dict | None, max_grid_images: int) -> str:
    rows = []
    for condition in conditions:
        metrics = detector_metrics.get(condition, {})
        overall = metrics.get("overall", {})
        align = alignment.get(condition, {})
        post = find_posttrain_metrics(root, condition)
        delta_ap = None if not post or not baseline else post.get("AP", 0) - baseline.get("AP", 0)
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(condition)}</b></td>"
            f"<td>{overall.get('instances', 0)}</td>"
            f"<td>{metric_value(overall.get('mean_matched_confidence'))}</td>"
            f"<td>{metric_value(overall.get('false_negative_rate'))}</td>"
            f"<td>{metric_value(overall.get('class_error_rate'))}</td>"
            f"<td>{metric_value(align.get('distance_to_source_failure'))}</td>"
            f"<td>{metric_value(align.get('failure_affinity'))}</td>"
            f"<td>{metric_value(post.get('AP') if post else None)}</td>"
            f"<td>{metric_value(delta_ap)}</td>"
            "</tr>"
        )

    generated_conditions = [condition for condition in conditions if condition != "source_original"]
    manifests = {
        condition: load_json(root / "generated" / condition / "manifest.json", [])
        for condition in generated_conditions
    }
    names = []
    if manifests:
        name_sets = [{item["file_name"] for item in items} for items in manifests.values()]
        names = sorted(set.intersection(*name_sets))[:max_grid_images] if name_sets else []
    manifest_maps = {condition: {item["file_name"]: item for item in items} for condition, items in manifests.items()}
    cards = []
    for name in names:
        first = manifest_maps[generated_conditions[0]][name]
        classes = ", ".join(item["class_name"] for item in first.get("layout_objects", []))
        figures = []
        for condition in generated_conditions:
            rel = Path("generated") / condition / "images" / name
            figures.append(
                f'<figure><figcaption>{html.escape(condition)}</figcaption><img loading="lazy" src="../{rel.as_posix()}"></figure>'
            )
        cards.append(
            f'<article><h3>{html.escape(name)}</h3><p>{html.escape(classes)}</p>'
            f'<div class="image-row">{"".join(figures)}</div></article>'
        )
    plot = f'<img class="plot" src="{plot_name}">' if plot_name else "<p>Feature files not available.</p>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Prompt Difficulty Ablation</title>
<style>
:root {{ color-scheme: light; }} body {{ margin:0; font-family:Arial,sans-serif; color:#172033; background:#f3f5f8; }}
header {{ padding:24px max(24px,4vw); background:#fff; border-bottom:1px solid #d8dee8; }}
main {{ max-width:1500px; margin:auto; padding:22px; }} section {{ margin-bottom:28px; }}
.panel, article {{ background:#fff; border:1px solid #d8dee8; border-radius:8px; padding:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:9px; border-bottom:1px solid #e5e9f0; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
.plot {{ display:block; max-width:100%; max-height:820px; margin:auto; }} .grid {{ display:grid; gap:14px; }}
.image-row {{ display:grid; grid-template-columns:repeat({max(len(generated_conditions), 1)},minmax(0,1fr)); gap:10px; }}
figure {{ margin:0; }} figcaption {{ font-weight:700; font-size:13px; margin-bottom:6px; }} img {{ width:100%; display:block; }}
h1,h2,h3,p {{ margin-top:0; }} .note {{ color:#536174; max-width:1000px; line-height:1.5; }}
</style></head><body>
<header><h1>Learned Prompt: Detector Difficulty Ablation</h1>
<p class="note">Lower confidence and higher FN indicate harder generated instances. Lower distance to source failure and higher failure affinity indicate closer backbone features to the real detector-failure region. Training AP delta must be interpreted together with class semantics and image quality.</p></header>
<main><section class="panel"><h2>Detector summary</h2><table><thead><tr><th>Condition</th><th>Instances</th><th>Mean conf</th><th>FN rate</th><th>Class error</th><th>Distance to failure</th><th>Failure affinity</th><th>Post-train AP</th><th>AP delta</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="panel"><h2>Backbone feature distribution</h2>{plot}</section>
<section><h2>Paired generated images</h2><div class="grid">{''.join(cards)}</div></section></main></body></html>"""


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_dir)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    detector_metrics = {
        condition: load_json(root / "analysis" / condition / "metrics.json", {})
        for condition in conditions
    }
    features, feature_conditions, _class_ids, failures = load_feature_sets(root, conditions)
    alignment = {}
    plot_name = None
    if len(features):
        pca_features = standardized_pca(features)
        alignment = compute_alignment(pca_features, feature_conditions, failures, conditions)
        plot_name = save_embedding_plot(
            pca_features, feature_conditions, failures, report_dir, args.max_feature_points, args.seed
        )
    baseline = load_json(Path(args.baseline_metrics), None) if args.baseline_metrics else None
    summary = {"detector_metrics": detector_metrics, "feature_alignment": alignment, "baseline": baseline}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    page = build_html(root, conditions, detector_metrics, alignment, plot_name, baseline, args.max_grid_images)
    (report_dir / "index.html").write_text(page, encoding="utf-8")
    print(report_dir / "index.html")


if __name__ == "__main__":
    main()
