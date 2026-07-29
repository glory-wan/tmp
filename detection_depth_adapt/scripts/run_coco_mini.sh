#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-py10}"
CONFIG="${CONFIG:-$ROOT/detection_depth_adapt/configs/coco_mini.yaml}"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

conda run -n "$ENV_NAME" python -m detection_depth_adapt.runner all --config "$CONFIG"
