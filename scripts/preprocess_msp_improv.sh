#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/minhnhutngnn/MSP-IMPROV}"
OUTPUT_ROOT="${OUTPUT_ROOT:-metadata}"
SEED="${SEED:-32}"

python src/data_processing/preprocess.py \
  --dataset msp-improv \
  --data_root "${DATA_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --seed "${SEED}" \
  "$@"
