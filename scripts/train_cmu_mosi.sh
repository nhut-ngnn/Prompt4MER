#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/minhnhutngnn/data/cmu-mosi}"
MOSI_FILE="${MOSI_FILE:-}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-./checkpoints/mosi_4mser_concat_pretrain.pt}"
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT:-./checkpoints/mosi_4mser_concat_finetune.pt}"
STAGE="${STAGE:-all}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU_ID="${GPU_ID:-0}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-2048}"
USE_CPU="${USE_CPU:-0}"

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
)

if [[ "${USE_CPU}" == "1" ]]; then
  COMMON_ARGS+=(--no_cuda)
else
  COMMON_ARGS+=(--gpu_id "${GPU_ID}")
fi

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

if [[ "${STAGE}" != "pretrain" && "${STAGE}" != "finetune" && "${STAGE}" != "all" ]]; then
  echo "Invalid STAGE='${STAGE}'. Use STAGE=pretrain, STAGE=finetune, or STAGE=all." >&2
  exit 2
fi

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

if [[ "${STAGE}" == "pretrain" || "${STAGE}" == "all" ]]; then
  check_gpu_memory
  run_training_stage "${PRETRAIN_CHECKPOINT}" \
    "${COMMON_ARGS[@]}" \
    --max_missing_prob 0 \
    --double_missing_prob 0 \
    --name "${PRETRAIN_CHECKPOINT}" \
    "$@"
fi

if [[ "${STAGE}" == "finetune" || "${STAGE}" == "all" ]]; then
  check_gpu_memory
  run_training_stage "${FINETUNE_CHECKPOINT}" \
    "${COMMON_ARGS[@]}" \
    --pretrained_model "${PRETRAIN_CHECKPOINT}" \
    --name "${FINETUNE_CHECKPOINT}" \
    "$@"
fi
