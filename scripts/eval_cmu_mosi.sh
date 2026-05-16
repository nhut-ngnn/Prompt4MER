#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/minhnhutngnn/data/cmu-mosi}"
MOSI_FILE="${MOSI_FILE:-}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/mosi_4mser_concat_finetune.pt}"
EVAL_CSV="${EVAL_CSV:-./checkpoints/mosi_4mser_concat_finetune_eval.csv}"

resolve_data_path() {
  if [[ -n "${DATA_PATH:-}" ]]; then
    echo "${DATA_PATH}"
    return
  fi
  if [[ -n "${MOSI_FILE}" ]]; then
    echo "${DATA_ROOT}/${MOSI_FILE}"
    return
  fi
  local candidates=(
    "mosi_data.pkl"
    "mosi_raw.pkl"
    "aligned_50.pkl"
    "unaligned_50.pkl"
    "Copy of aligned_50.pkl"
  )
  local filename
  for filename in "${candidates[@]}"; do
    if [[ -f "${DATA_ROOT}/${filename}" ]]; then
      echo "${DATA_ROOT}/${filename}"
      return
    fi
  done
  echo "${DATA_ROOT}"
}

DATA_PATH="$(resolve_data_path)"
if [[ ! -e "${DATA_PATH}" ]]; then
  echo "MOSI data path does not exist: ${DATA_PATH}" >&2
  echo "Set DATA_PATH=/path/to/file.pkl or put a MOSI .pkl file under ${DATA_ROOT}" >&2
  exit 2
fi
echo "Using MOSI data: ${DATA_PATH}"

python main.py \
  --eval_only \
  --dataset mosi \
  --data_path "${DATA_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --linear_layer_output 512,256 \
  --eval_split test \
  --eval_modalities atv,t,a,v,at,av,tv \
  --eval_csv "${EVAL_CSV}" \
  "$@"
