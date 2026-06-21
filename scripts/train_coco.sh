#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-sd3}"
DATASET="${DATASET:-coco}"
DATADIR="${DATADIR:-/path/to/datasets}"
LOGDIR="${LOGDIR:-./outputs/train}"
USE_PRECOMPUTED_ENCODINGS="${USE_PRECOMPUTED_ENCODINGS:-false}"
ENCODING_DIR="${ENCODING_DIR:-${DATADIR}/encodings_${MODEL}}"

case "${MODEL}" in
  sd3)
    DEFAULT_IMG_SIZE="512"
    DEFAULT_BATCH_SIZE="4"
    DEFAULT_NROWS="4"
    DEFAULT_EPOCHS="10"
    DEFAULT_NUM_ITER="1000"
    DEFAULT_NEG_SCALE="0.1"
    ;;
  sd1.5)
    DEFAULT_IMG_SIZE="512"
    DEFAULT_BATCH_SIZE="4"
    DEFAULT_NROWS="4"
    DEFAULT_EPOCHS="20"
    DEFAULT_NUM_ITER="1000"
    DEFAULT_NEG_SCALE="1.0"
    ;;
  sdxl)
    DEFAULT_IMG_SIZE="512"
    DEFAULT_BATCH_SIZE="4"
    DEFAULT_NROWS="2"
    DEFAULT_EPOCHS="1"
    DEFAULT_NUM_ITER="100"
    DEFAULT_NEG_SCALE="0.1"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}" >&2
    exit 1
    ;;
esac

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
IMG_SIZE="${IMG_SIZE:-${DEFAULT_IMG_SIZE}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NROWS="${NROWS:-${DEFAULT_NROWS}}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
NUM_ITER="${NUM_ITER:-${DEFAULT_NUM_ITER}}"
NUM_EVAL="${NUM_EVAL:-50}"

N_SOFT_TOKENS="${N_SOFT_TOKENS:-4}"
N_SOFT_LAYERS="${N_SOFT_LAYERS:-5}"
SCALE="${SCALE:-1.0}"
NEG_SCALE="${NEG_SCALE:-${DEFAULT_NEG_SCALE}}"
WANDB="${WANDB:-false}"
NOTE="${NOTE:-agsm}"

EXTRA_ARGS=()
if [[ "${USE_PRECOMPUTED_ENCODINGS}" == "true" ]]; then
  EXTRA_ARGS+=(--use_precomputed_encodings --encoding_dir "${ENCODING_DIR}")
fi

torchrun --nproc-per-node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --datadir "${DATADIR}" \
  --logdir "${LOGDIR}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --nrows "${NROWS}" \
  --epochs "${EPOCHS}" \
  --num_iter "${NUM_ITER}" \
  --num_eval "${NUM_EVAL}" \
  --n_soft_tokens "${N_SOFT_TOKENS}" \
  --n_soft_layers "${N_SOFT_LAYERS}" \
  --use_soft_t \
  --use_soft_tokens \
  --scale "${SCALE}" \
  --neg_scale "${NEG_SCALE}" \
  --wandb "${WANDB}" \
  --note "${NOTE}" \
  "${EXTRA_ARGS[@]}"
