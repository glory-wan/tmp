from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    root = path.parents[1]
    cfg["project_root"] = str(root)
    for section, keys in {
        "paths": ("workspace", "dataset", "yolov7", "weights"),
    }.items():
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                cfg[section][key] = str((root / value).resolve())
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    workspace = Path(cfg["paths"]["workspace"])
    for name in ("failures", "prompts", "synthetic", "datasets", "runs", "state"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
