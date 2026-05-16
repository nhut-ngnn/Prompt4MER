#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-feature/}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/iemocap_missing_ablation}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${OUTPUT_DIR}/iemocap_ablation_pretrain.pt}"
EVAL_MODALITIES="${EVAL_MODALITIES:-atv,t,a,v,at,av,tv}"
MAX_MISSING_VALUES="${MAX_MISSING_VALUES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
DOUBLE_MISSING_PROB="${DOUBLE_MISSING_PROB:-0.25}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SEED="${SEED:-32}"
NUM_SEEDS="${NUM_SEEDS:-1}"

COMMON_ARGS=(
  --dataset iemocap
  --data_path "${DATA_PATH}"
  --linear_layer_output 512,256
  --optim AdamW
  --lr 5e-4
  --weight_decay 1e-4
  --when 5
  --scheduler_factor 0.5
  --seed "${SEED}"
  --num_seeds "${NUM_SEEDS}"
)

mkdir -p "${OUTPUT_DIR}"

tag_value() {
  local value="$1"
  echo "${value}" | tr '.' 'p'
}

if [[ "${RUN_PRETRAIN}" == "1" ]]; then
  python main.py \
    "${COMMON_ARGS[@]}" \
    --max_missing_prob 0 \
    --double_missing_prob 0 \
    --name "${PRETRAIN_CHECKPOINT}" \
    "$@"
fi

for max_missing_prob in ${MAX_MISSING_VALUES}; do
  double_missing_prob="${DOUBLE_MISSING_PROB}"
  tag="max$(tag_value "${max_missing_prob}")_double$(tag_value "${double_missing_prob}")"
  finetune_checkpoint="${OUTPUT_DIR}/iemocap_ablation_${tag}_finetune.pt"
  eval_csv="${OUTPUT_DIR}/iemocap_ablation_${tag}_eval.csv"

  if [[ "${RUN_FINETUNE}" == "1" ]]; then
    python main.py \
      "${COMMON_ARGS[@]}" \
      --pretrained_model "${PRETRAIN_CHECKPOINT}" \
      --max_missing_prob "${max_missing_prob}" \
      --double_missing_prob "${double_missing_prob}" \
      --name "${finetune_checkpoint}" \
      "$@"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    python main.py \
      --eval_only \
      --dataset iemocap \
      --data_path "${DATA_PATH}" \
      --checkpoint "${finetune_checkpoint}" \
      --linear_layer_output 512,256 \
      --eval_split test \
      --eval_modalities "${EVAL_MODALITIES}" \
      --eval_csv "${eval_csv}" \
      --seed "${SEED}" \
      --num_seeds "${NUM_SEEDS}" \
      "$@"
  fi
done

if [[ "${RUN_EVAL}" == "1" ]]; then
  python - "${OUTPUT_DIR}" "${DOUBLE_MISSING_PROB}" ${MAX_MISSING_VALUES} <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
double_missing_prob = sys.argv[2]
max_missing_values = sys.argv[3:]
summary_path = output_dir / "summary.csv"

rows = []
fieldnames = {"max_missing_prob", "double_missing_prob", "row_type", "modality"}

def tag_value(value: str) -> str:
    return value.replace(".", "p")

for max_missing_prob in max_missing_values:
    tag = f"max{tag_value(max_missing_prob)}_double{tag_value(double_missing_prob)}"
    csv_path = output_dir / f"iemocap_ablation_{tag}_eval.csv"
    if not csv_path.exists():
        continue
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("row_type") not in {"mean", "std"}:
                continue
            merged = {
                "max_missing_prob": max_missing_prob,
                "double_missing_prob": double_missing_prob,
                **row,
            }
            rows.append(merged)
            fieldnames.update(merged.keys())

ordered_fields = [
    "max_missing_prob",
    "double_missing_prob",
    "row_type",
    "modality",
    "loss",
    "metrics.acc",
    "metrics.f1",
]
ordered_fields += sorted(fieldnames - set(ordered_fields))

with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=ordered_fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved ablation summary to {summary_path}")
PY
fi
