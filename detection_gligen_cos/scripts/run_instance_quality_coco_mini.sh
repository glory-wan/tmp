#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-py10}"

cd "$ROOT"

conda run -n "$ENV_NAME" python -m detection_gligen_sdedit.analyze_instance_quality \
  --config detection_gligen_sdedit/configs/instance_quality_coco_mini.yaml \
  "$@"
