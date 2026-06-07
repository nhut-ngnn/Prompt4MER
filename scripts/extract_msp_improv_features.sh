#!/usr/bin/env bash
set -euo pipefail

WAV_BASE="${WAV_BASE:-/}"
VIDEO_BASE="${VIDEO_BASE:-/}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-8}"

python src/feature_extract/extract_feature.py \
  --dataset msp-improv \
  --wav_base "${WAV_BASE}" \
  --video_base "${VIDEO_BASE}" \
  --max_video_frames "${MAX_VIDEO_FRAMES}" \
  "$@"
