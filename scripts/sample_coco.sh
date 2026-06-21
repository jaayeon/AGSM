#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-sd3}"
DATASET="${DATASET:-coco}"
DATADIR="${DATADIR:-/path/to/datasets}"

case "${MODEL}" in
  sd3)
    DEFAULT_CHECKPOINT_DIR="checkpoints/sd3"
    DEFAULT_CFG_SCALE="4.0"
    DEFAULT_IMG_SIZE="1024"
    DEFAULT_NFE="28"
    DEFAULT_UNCON2NEG="false"
    ;;
  sd1.5)
    DEFAULT_CHECKPOINT_DIR="checkpoints/sd1.5"
    DEFAULT_CFG_SCALE="7.0"
    DEFAULT_IMG_SIZE="512"
    DEFAULT_NFE="30"
    DEFAULT_UNCON2NEG="false"
    ;;
  sdxl)
    DEFAULT_CHECKPOINT_DIR="checkpoints/sdxl"
    DEFAULT_CFG_SCALE="7.0"
    DEFAULT_IMG_SIZE="1024"
    DEFAULT_NFE="30"
    DEFAULT_UNCON2NEG="true"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}" >&2
    exit 1
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
EPOCH="${EPOCH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/samples/${MODEL}}"

CFG_SCALE="${CFG_SCALE:-${DEFAULT_CFG_SCALE}}"
NFE="${NFE:-${DEFAULT_NFE}}"
IMG_SIZE="${IMG_SIZE:-${DEFAULT_IMG_SIZE}}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_SOFT_TOKENS="${N_SOFT_TOKENS:-4}"
N_SOFT_LAYERS="${N_SOFT_LAYERS:-5}"
UNCON2NEG="${UNCON2NEG:-${DEFAULT_UNCON2NEG}}"

EXTRA_ARGS=()
if [[ "${UNCON2NEG}" == "true" ]]; then
  EXTRA_ARGS+=(--uncon2neg)
fi
if [[ -n "${EPOCH}" ]]; then
  EXTRA_ARGS+=(--load_ep "${EPOCH}")
fi

python sample.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --datadir "${DATADIR}" \
  --load_dir "${CHECKPOINT_DIR}" \
  --save_dir "${OUTPUT_DIR}" \
  --cfg_scale "${CFG_SCALE}" \
  --NFE "${NFE}" \
  --img_size "${IMG_SIZE}" \
  --num "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --n_soft_tokens "${N_SOFT_TOKENS}" \
  --n_soft_layers "${N_SOFT_LAYERS}" \
  --use_soft_t \
  --use_soft_tokens \
  "${EXTRA_ARGS[@]}"
