#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-feature/}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-./checkpoints/msp_improv_4mser_concat_pretrain.pt}"
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT:-./checkpoints/msp_improv_4mser_concat_finetune.pt}"
STAGE="${STAGE:-all}"

COMMON_ARGS=(
  --dataset msp-improv
  --data_path "${DATA_PATH}"
  --linear_layer_output 512,256
  --optim AdamW
  --lr 5e-4
  --weight_decay 1e-4
  --when 5
  --scheduler_factor 0.5
)

if [[ "${STAGE}" != "pretrain" && "${STAGE}" != "finetune" && "${STAGE}" != "all" ]]; then
  echo "Invalid STAGE='${STAGE}'. Use STAGE=pretrain, STAGE=finetune, or STAGE=all." >&2
  exit 2
fi

if [[ "${STAGE}" == "pretrain" || "${STAGE}" == "all" ]]; then
  python main.py \
    "${COMMON_ARGS[@]}" \
    --max_missing_prob 0 \
    --double_missing_prob 0 \
    --name "${PRETRAIN_CHECKPOINT}" \
    "$@"
fi

if [[ "${STAGE}" == "finetune" || "${STAGE}" == "all" ]]; then
  python main.py \
    "${COMMON_ARGS[@]}" \
    --pretrained_model "${PRETRAIN_CHECKPOINT}" \
    --name "${FINETUNE_CHECKPOINT}" \
    "$@"
fi
