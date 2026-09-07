#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-py10}"
SYNTHETIC_POOL="${1:-outputs/detection_gligen_sdedit/exp_0728/round_1/synthetic_pool}"
OUTPUT_DIR="${2:-outputs/gen_vis}"

cd "$ROOT"

conda run -n "$ENV_NAME" python -m detection_gligen_sdedit.experiments.build_synthetic_pool_html \
  --synthetic-pool "$SYNTHETIC_POOL" \
  --output-dir "$OUTPUT_DIR"
