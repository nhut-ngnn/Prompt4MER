#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-feature/}"
IEMOCAP_DATA_PATH="${IEMOCAP_DATA_PATH:-${DATA_PATH}}"
MSP_IMPROV_DATA_PATH="${MSP_IMPROV_DATA_PATH:-${DATA_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/module_ablation}"
EVAL_MODALITIES="${EVAL_MODALITIES:-t,a,v,at,av,tv}"
SEED="${SEED:-32}"
NUM_SEEDS="${NUM_SEEDS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --linear_layer_output 512,256
  --optim AdamW
  --lr 5e-4
  --weight_decay 1e-4
  --when 5
  --scheduler_factor 0.5
  --seed "${SEED}"
  --num_seeds "${NUM_SEEDS}"
)

DATASETS=("iemocap" "msp-improv")
SETTINGS=(
  "Prompt4MSER|full|0.5|0.25"
  "W/o P_bank|no_prompt_bank|0.5|0.25"
  "W/o P_missing|no_missing_prompt|0.5|0.25"
  "W/o P_modality|no_modality_prompt|0.5|0.25"
  "W/o modality missing|full|0.0|0.0"
)

dataset_data_path() {
  case "$1" in
    iemocap) echo "${IEMOCAP_DATA_PATH}" ;;
    msp-improv) echo "${MSP_IMPROV_DATA_PATH}" ;;
    *) echo "${DATA_PATH}" ;;
  esac
}

dataset_missing_prob() {
  case "$1" in
    iemocap) echo "0.5" ;;
    msp-improv) echo "0.6" ;;
    *) echo "0.5" ;;
  esac
}

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//'
}

for dataset in "${DATASETS[@]}"; do
  data_path="$(dataset_data_path "${dataset}")"
  dataset_prob="$(dataset_missing_prob "${dataset}")"
  dataset_tag="$(safe_name "${dataset}")"

  for setting in "${SETTINGS[@]}"; do
    IFS="|" read -r label module_ablation missing_prob double_missing_prob <<< "${setting}"
    if [[ "${label}" != "W/o modality missing" ]]; then
      missing_prob="${dataset_prob}"
    fi

    setting_tag="$(safe_name "${label}")"
    checkpoint="${OUTPUT_DIR}/${dataset_tag}_${setting_tag}.pt"
    eval_csv="${OUTPUT_DIR}/${dataset_tag}_${setting_tag}_eval.csv"

    if [[ "${RUN_TRAIN}" == "1" ]]; then
      "${PYTHON}" main.py \
        "${COMMON_ARGS[@]}" \
        --dataset "${dataset}" \
        --data_path "${data_path}" \
        --module_ablation "${module_ablation}" \
        --max_missing_prob "${missing_prob}" \
        --double_missing_prob "${double_missing_prob}" \
        --name "${checkpoint}" \
        "$@"
    fi

    if [[ "${RUN_EVAL}" == "1" ]]; then
      "${PYTHON}" main.py \
        --eval_only \
        --dataset "${dataset}" \
        --data_path "${data_path}" \
        --checkpoint "${checkpoint}" \
        --linear_layer_output 512,256 \
        --module_ablation "${module_ablation}" \
        --eval_split test \
        --eval_modalities "${EVAL_MODALITIES}" \
        --eval_csv "${eval_csv}" \
        --seed "${SEED}" \
        --num_seeds "${NUM_SEEDS}" \
        "$@"
    fi
  done
done

"${PYTHON}" - "${OUTPUT_DIR}" "${EVAL_MODALITIES}" "${SETTINGS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
modalities = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
settings = [item.split("|", 1)[0] for item in sys.argv[3:]]

datasets = [
    ("iemocap", "iemocap", "IEMOCAP"),
    ("msp_improv", "msp-improv", "MSP-IMPROV"),
]


def safe_name(value: str) -> str:
    out = []
    previous_sep = False
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
            previous_sep = False
        elif not previous_sep:
            out.append("_")
            previous_sep = True
    return "".join(out).strip("_")


def read_average(csv_path: Path):
    if not csv_path.exists():
        return "", ""

    acc_values = []
    f1_values = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("row_type") != "mean":
                continue
            if row.get("modality") not in modalities:
                continue
            try:
                acc_values.append(float(row["metrics.acc"]))
                f1_values.append(float(row["metrics.f1"]))
            except (KeyError, TypeError, ValueError):
                continue

    if not acc_values or not f1_values:
        return "", ""
    return (
        100.0 * sum(acc_values) / len(acc_values),
        100.0 * sum(f1_values) / len(f1_values),
    )


rows = []
for setting in settings:
    row = {"Setting": setting}
    setting_tag = safe_name(setting)
    for dataset_tag, _dataset_name, label in datasets:
        acc, f1 = read_average(output_dir / f"{dataset_tag}_{setting_tag}_eval.csv")
        row[f"{label} ACC (%)"] = f"{acc:.2f}" if acc != "" else ""
        row[f"{label} F1 (%)"] = f"{f1:.2f}" if f1 != "" else ""
    rows.append(row)

summary_path = output_dir / "module_ablation_summary.csv"
fieldnames = [
    "Setting",
    "IEMOCAP ACC (%)",
    "IEMOCAP F1 (%)",
    "MSP-IMPROV ACC (%)",
    "MSP-IMPROV F1 (%)",
]
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved module ablation summary to {summary_path}")
PY
