#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/minhnhutngnn/data/cmu-mosi}"
MOSI_FILE="${MOSI_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/mosi_missing_ablation}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${OUTPUT_DIR}/mosi_ablation_pretrain.pt}"
EVAL_MODALITIES="${EVAL_MODALITIES:-atv,t,a,v,at,av,tv}"
MAX_MISSING_VALUES="${MAX_MISSING_VALUES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
DOUBLE_MISSING_PROB="${DOUBLE_MISSING_PROB:-0.25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU_ID="${GPU_ID:-0}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-2048}"
USE_CPU="${USE_CPU:-0}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SEED="${SEED:-32}"
NUM_SEEDS="${NUM_SEEDS:-1}"

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

COMMON_ARGS=(
  --dataset mosi
  --data_path "${DATA_PATH}"
  --linear_layer_output 512,256
  --optim AdamW
  --lr 5e-4
  --weight_decay 1e-4
  --when 5
  --scheduler_factor 0.5
  --batch_size "${BATCH_SIZE}"
  --skip_epoch_test_eval
  --skip_final_eval
  --seed "${SEED}"
  --num_seeds "${NUM_SEEDS}"
)

if [[ "${USE_CPU}" == "1" ]]; then
  COMMON_ARGS+=(--no_cuda)
else
  COMMON_ARGS+=(--gpu_id "${GPU_ID}")
fi

mkdir -p "${OUTPUT_DIR}"

tag_value() {
  local value="$1"
  echo "${value}" | tr '.' 'p'
}

run_training_stage() {
  local checkpoint="$1"
  shift
  set +e
  python main.py "$@"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    if [[ "${status}" -eq 137 && -s "${checkpoint}" ]]; then
      echo "WARNING: training process was killed after writing ${checkpoint}; continuing." >&2
      return 0
    fi
    return "${status}"
  fi
}

check_gpu_memory() {
  if [[ "${USE_CPU}" == "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  local free_mb
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ -z "${free_mb}" ]]; then
    return
  fi
  if (( free_mb < GPU_MIN_FREE_MB )); then
    echo "CUDA GPU ${GPU_ID} has only ${free_mb} MB free; need at least ${GPU_MIN_FREE_MB} MB." >&2
    echo "Current GPU processes:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null >&2 || true
    echo "Stop the process using the GPU, choose another GPU with GPU_ID=<id>, or run USE_CPU=1." >&2
    exit 2
  fi
}

if [[ "${RUN_PRETRAIN}" == "1" ]]; then
  check_gpu_memory
  run_training_stage "${PRETRAIN_CHECKPOINT}" \
    "${COMMON_ARGS[@]}" \
    --max_missing_prob 0 \
    --double_missing_prob 0 \
    --name "${PRETRAIN_CHECKPOINT}" \
    "$@"
fi

for max_missing_prob in ${MAX_MISSING_VALUES}; do
  double_missing_prob="${DOUBLE_MISSING_PROB}"
  tag="max$(tag_value "${max_missing_prob}")_double$(tag_value "${double_missing_prob}")"
  finetune_checkpoint="${OUTPUT_DIR}/mosi_ablation_${tag}_finetune.pt"
  eval_csv="${OUTPUT_DIR}/mosi_ablation_${tag}_eval.csv"

  if [[ "${RUN_FINETUNE}" == "1" ]]; then
    check_gpu_memory
    run_training_stage "${finetune_checkpoint}" \
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
      --dataset mosi \
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
    csv_path = output_dir / f"mosi_ablation_{tag}_eval.csv"
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
    "metrics.corr",
    "metrics.f1",
    "metrics.mae",
    "metrics.mult_acc_5",
    "metrics.mult_acc_7",
]
ordered_fields += sorted(fieldnames - set(ordered_fields))

with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=ordered_fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved ablation summary to {summary_path}")
PY
fi
