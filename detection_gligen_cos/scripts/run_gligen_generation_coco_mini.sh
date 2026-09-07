#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-py10}"

cd "$ROOT"

conda run -n "$ENV_NAME" python -m detection_gligen_sdedit.generate_gligen_layout \
  --config detection_gligen_sdedit/configs/gligen_generation_coco_mini.yaml \
  "$@"
