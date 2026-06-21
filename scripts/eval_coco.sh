#!/usr/bin/env bash
set -euo pipefail

DATADIR="${DATADIR:-/path/to/datasets}"
MODEL="${MODEL:-sd3}"
DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
BENCHMARKS="${BENCHMARKS:-ImageReward-v1.0,CLIP,PickScore}"

case "${MODEL}" in
  sd3)
    DEFAULT_CHECKPOINT_DIR="checkpoints/sd3"
    DEFAULT_CFG_SCALE="4.0"
    DEFAULT_UNCON2NEG="false"
    ;;
  sd1.5)
    DEFAULT_CHECKPOINT_DIR="checkpoints/sd1.5"
    DEFAULT_CFG_SCALE="7.0"
    DEFAULT_UNCON2NEG="false"
    ;;
  sdxl)
    DEFAULT_CHECKPOINT_DIR="checkpoints/sdxl"
    DEFAULT_CFG_SCALE="7.0"
    DEFAULT_UNCON2NEG="true"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}" >&2
    exit 1
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
RESULT_ROOT="${RESULT_ROOT:-./outputs/samples/${MODEL}}"
EPOCH="${EPOCH:-}"
CFG_SCALE="${CFG_SCALE:-${DEFAULT_CFG_SCALE}}"
UNCON2NEG="${UNCON2NEG:-${DEFAULT_UNCON2NEG}}"

SUFFIX=""
if [[ "${UNCON2NEG}" == "true" ]]; then
  SUFFIX="-uncon2neg"
fi
EP_PART=""
if [[ -n "${EPOCH}" ]]; then
  EP_PART="-ep${EPOCH}"
fi
LOAD_NAME="${LOAD_NAME:-${DATASET}-cfg${CFG_SCALE}-softTrue-softtTrue${EP_PART}-num${NUM_SAMPLES}${SUFFIX}}"

python eval.py \
  --load_dir "${RESULT_ROOT}" \
  --load_name "${LOAD_NAME}" \
  --datadir "${DATADIR}" \
  --dataset "${DATASET}" \
  --benchmark "${BENCHMARKS}" \
  --num "${NUM_SAMPLES}"
