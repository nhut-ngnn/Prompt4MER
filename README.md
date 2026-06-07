# Prompt4MER: Prompt-Guided Missing Modality Completion for Multimodal Emotion Recognition

> Official code repository for the manuscript 
  <b>"Prompt-Guided Missing Modality Completion for Multimodal Emotion Recognition"</b>, submitted to <a href="https://www.ieice.org/cs/icm/apnoms/2026/index.html">The 26th Asia-Pacific Network Operations and Management Symposium</a>.

> Please press ⭐ button and/or cite papers if you feel helpful.

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-21.04.2026-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>

<div align="center">

[**Introduction**](#introduciton) •
[**Repository Layout**](#repository-layout) •
[**Setup**](#setup) •
[**Data Preparation**](#data-preparation) •
[**Training**](#training) •
[**Evaluation**](#evaluation) •
[**Notes**](#notes)

</div>

## Introduciton

Prompt4MER is a prompt-guided model for multimodal emotion recognition under missing-modality conditions. The model uses audio, text, and visual features, then improves robustness with:

- **LMMP**: learnable modality prompts for observed inputs and missing prompts for unavailable modalities.
- **Text-guided cross-modal fusion**: text acts as the semantic guide to attend over audio and visual representations.
- **Two-stage training**: full-modality pretraining first, then missing-modality fine-tuning with random modality masks.

This codebase currently supports the two main datasets used in the study: **IEMOCAP** and **MSP-IMPROV**. Both are treated as 4-class emotion recognition datasets: `angry`, `happy`, `sad`, `neutral`.

## Repository Layout

```text
main.py                         # train / evaluate Prompt4MER
cli.py                          # helper CLI entry
src/architecture/               # Prompt4MER architecture
src/data/                       # feature dataset loaders
src/data_processing/            # metadata preprocessing
src/feature_extract/            # BERT / WavLM / CLIP feature extraction
scripts/                        # MSP-IMPROV helper scripts
metadata/                       # preprocessed split files
feature/                        # extracted feature .pkl files
checkpoints/                    # saved models and eval CSV files
```

## Setup

Run commands from the repository root.

```bash
cd /home/minhnhutngnn/Prompt4MER
pip install -r requirements.txt
```

If your environment uses `python` instead of `python3`, replace `python3` in the commands below.

## Data Preparation

Prompt4MER trains from extracted feature files in `feature/`. The expected files are:

```text
feature/IEMOCAP_BERT_LARGE_WavLM_CLIP_train.pkl
feature/IEMOCAP_BERT_LARGE_WavLM_CLIP_val.pkl
feature/IEMOCAP_BERT_LARGE_WavLM_CLIP_test.pkl
feature/MSP_IMPROV_BERT_LARGE_WavLM_CLIP_train.pkl
feature/MSP_IMPROV_BERT_LARGE_WavLM_CLIP_val.pkl
feature/MSP_IMPROV_BERT_LARGE_WavLM_CLIP_test.pkl
```

Each sample contains text, audio, visual embeddings and a 4-class label.

### MSP-IMPROV

```bash
DATA_ROOT=/path/to/MSP-IMPROV scripts/preprocess_msp_improv.sh
WAV_BASE=/path/to/MSP-IMPROV VIDEO_BASE=/path/to/MSP-IMPROV scripts/extract_msp_improv_features.sh
```

### IEMOCAP

```bash
python3 src/data_processing/preprocess.py \
  --dataset iemocap \
  --data_root /path/to/IEMOCAP \
  --output_root metadata

python3 src/feature_extract/extract_feature.py \
  --dataset iemocap \
  --wav_base /path/to/IEMOCAP \
  --video_base /path/to/IEMOCAP
```

## Training

Stage 1 pretrains with all modalities available:

```bash
python3 main.py \
  --dataset iemocap \
  --data_path feature/ \
  --linear_layer_output 512,256 \
  --optim AdamW \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --when 5 \
  --scheduler_factor 0.5 \
  --max_missing_prob 0 \
  --double_missing_prob 0 \
  --num_seeds 5 \
  --name ./checkpoints/iemocap_4mser_concat_pretrain.pt
```

Stage 2 fine-tunes with random missing-modality masks:

```bash
python3 main.py \
  --pretrained_model ./checkpoints/iemocap_4mser_concat_pretrain.pt \
  --dataset iemocap \
  --data_path feature/ \
  --linear_layer_output 512,256 \
  --optim AdamW \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --when 5 \
  --scheduler_factor 0.5 \
  --num_seeds 5 \
  --name ./checkpoints/iemocap_4mser_concat_finetune.pt
```

For MSP-IMPROV, use `--dataset msp-improv` and MSP checkpoint names. You can also run:

```bash
scripts/train_msp_improv.sh
```

## Evaluation

```bash
python3 main.py \
  --eval_only \
  --dataset iemocap \
  --data_path feature/ \
  --checkpoint ./checkpoints/iemocap_4mser_concat_finetune.pt \
  --linear_layer_output 512,256 \
  --eval_split test \
  --eval_modalities atv,t,a,v,at,av,tv
```

Modality cases:

- `atv`: audio + text + visual
- `t`, `a`, `v`: single observed modality
- `at`, `av`, `tv`: two observed modalities

With `--num_seeds 5`, training and evaluation use seeds `32, 33, 34, 35, 36` by default. Eval-only writes a CSV next to the checkpoint unless `--eval_csv` is provided.

## Notes

- Use `--max_missing_prob 0 --double_missing_prob 0` for full-modality pretraining.
- During fine-tuning, `--max_missing_prob` controls how often samples are masked and `--double_missing_prob` controls how often two modalities are dropped.
- The current repository is intentionally focused on IEMOCAP and MSP-IMPROV only.
