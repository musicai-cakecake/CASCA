#!/bin/bash
# ============================================================
# XMIDI-GetMusic validation NLL / PPL / BPD (region-aware NELBO upper bound).
#
# Calls the single-GPU canonical fine-grained NELBO evaluator on all 6817
# validation samples. Reported NLL/PPL/BPD are computed only over **target content
# region** (rows 4..15, cols 16..511, excluding fully-<UNK> rows /
# individual <UNK> positions).
#
# Usage:
#   bash scripts/ppl/eval_xmidi_getmusic_bd3lm.sh <CKPT_PATH> [BLOCK_SIZE] [BSZ] [MODEL]
#
# Defaults:
#   CKPT_PATH:  required
#   BLOCK_SIZE: 16
#   BSZ:        2
#   MODEL:      xmidi_getmusic_small
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f ./setup_env.sh ]; then
  source ./setup_env.sh
fi

CKPT=${1:?"usage: $0 <CKPT_PATH> [BLOCK_SIZE] [BSZ] [MODEL]"}
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
BLOCK_SIZE=${2:-16}
BSZ=${3:-2}
MODEL=${4:-xmidi_getmusic_small}
SEED=${SEED:-1}
WEIGHT_SOURCE=${WEIGHT_SOURCE:-ema}
if [[ "${WEIGHT_SOURCE}" != "ema" && "${WEIGHT_SOURCE}" != "raw" ]]; then
  echo "WEIGHT_SOURCE must be ema or raw" >&2
  exit 1
fi

# Canonical validation uses exactly one GPU so the 6817-sample split is not
# padded/duplicated by a distributed sampler.
export CUDA_VISIBLE_DEVICES=${EVAL_DEVICE:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

N_GPUS=1

PY=${PYBIN:-$(command -v python)}
if [ ! -x "${PY}" ]; then
  PY=$(command -v python)
fi

RUN_NAME=ppl-xmidi-getmusic-${MODEL}-bs${BLOCK_SIZE}
OUT_DIR=${OUT_DIR:-${BD3LM_ROOT:-$(pwd)}/outputs/${RUN_NAME}}
NUM_T_SAMPLES=${NUM_T_SAMPLES:-4}
mkdir -p logs "${OUT_DIR}"

echo "[ppl_xmidi_getmusic] ckpt       = ${CKPT}"
echo "[ppl_xmidi_getmusic] block_size = ${BLOCK_SIZE}"
echo "[ppl_xmidi_getmusic] num GPUs   = ${N_GPUS}"
echo "[ppl_xmidi_getmusic] eval bsz   = ${BSZ}"
echo "[ppl_xmidi_getmusic] model      = ${MODEL}"
echo "[ppl_xmidi_getmusic] weights    = ${WEIGHT_SOURCE}"
echo "[ppl_xmidi_getmusic] seed       = ${SEED}"
echo "[ppl_xmidi_getmusic] t samples  = ${NUM_T_SAMPLES}"
echo "[ppl_xmidi_getmusic] out dir    = ${OUT_DIR}"

${PY} -u scripts/eval/evaluate_xmidi_getmusic.py \
    --ckpt "${CKPT}" \
    --model "${MODEL}" \
    --block_size "${BLOCK_SIZE}" \
    --bsz "${BSZ}" \
    --num_t_samples "${NUM_T_SAMPLES}" \
    --max_batches 0 \
    --expect_valid_size 6817 \
    --weight_source "${WEIGHT_SOURCE}" \
    --seed "${SEED}" \
    --out_json "${OUT_DIR}/nelbo_summary.json" \
    --out_csv "${OUT_DIR}/nelbo_summary.csv" \
    2>&1 | tee "logs/${RUN_NAME}.log"
