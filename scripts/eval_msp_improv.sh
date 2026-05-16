#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-feature/}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/msp_improv_4mser_concat_finetune.pt}"
EVAL_CSV="${EVAL_CSV:-./checkpoints/msp_improv_4mser_concat_finetune_eval.csv}"

python main.py \
  --eval_only \
  --dataset msp-improv \
  --data_path "${DATA_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --linear_layer_output 512,256 \
  --eval_split test \
  --eval_modalities atv,t,a,v,at,av,tv \
  --eval_csv "${EVAL_CSV}" \
  "$@"
