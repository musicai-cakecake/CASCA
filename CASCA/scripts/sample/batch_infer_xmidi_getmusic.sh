#!/bin/bash
# ============================================================
# Batch source-conditioned inference on the FULL XMIDI-GetMusic valid set.
#
# Wrapper around scripts/eval/batch_infer_xmidi_getmusic.py. Writes a
# canonical JSONL v3 plus manifest. Each record includes exact 16x512 token
# grids, valid_index/sample_seed, generation-mask RLE, eight structural/runtime
# checks, diffusion_nfes and total backbone_forward_calls.
#
# Adaptive masking (R&D §"局部 attention 算法"): the underlying
# _xmidi_semi_ar_sampler now registers `row_unk_mask` on the DIT
# backbone for the duration of each batch, so fully-<UNK> rows are
# dropped from attention both as queries and as keys-- matching the
# train-time mask. Source rows / prompt cols / <UNK> tokens stay
# fixed throughout the diffusion chain.
#
# Usage:
#   bash scripts/sample/batch_infer_xmidi_getmusic.sh <CKPT> [BLOCK_SIZE] [BSZ] [T] [MODEL] [OUT]
#
# Defaults:
#   CKPT:        required
#   BLOCK_SIZE:  16
#   BSZ:         1   (first_hitting requires bsz=1)
#   T:           1000
#   MODEL:       xmidi_getmusic_small
#   OUT:         sample_logs/xmidi_getmusic_valid_full.jsonl
#
# Useful env vars:
#   START_IDX=N       resume / shard from this index (default 0)
#   MAX_SAMPLES=N     limit to first N samples (debug)
#   SAMPLE_DEVICE=k   use GPU k (default 0)
#   WEIGHT_SOURCE=ema select EMA (default) or raw checkpoint weights
#   RESUME=1          safely resume using the JSONL manifest
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f ./setup_env.sh ]; then
  source ./setup_env.sh
fi

CKPT=${1:?"usage: $0 <CKPT_PATH> [BLOCK_SIZE] [BSZ] [T] [MODEL] [OUT]"}
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
BLOCK_SIZE=${2:-16}
BSZ=${3:-1}
T_STEPS=${4:-1000}
MODEL=${5:-xmidi_getmusic_small}
OUT=${6:-sample_logs/xmidi_getmusic_valid_full.jsonl}
SEED=${SEED:-1}
WEIGHT_SOURCE=${WEIGHT_SOURCE:-ema}
if [[ "${WEIGHT_SOURCE}" != "ema" && "${WEIGHT_SOURCE}" != "raw" ]]; then
  echo "WEIGHT_SOURCE must be ema or raw" >&2
  exit 1
fi

START_IDX=${START_IDX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_DEVICE=${SAMPLE_DEVICE:-0}

# Fixed-batch single-GPU inference.
export CUDA_VISIBLE_DEVICES=${SAMPLE_DEVICE}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}

mkdir -p "$(dirname "$OUT")" logs

PY=${PYBIN:-$(command -v python)}
if [ ! -x "${PY}" ]; then
  PY=$(command -v python)
fi

echo "[batch_infer_xmidi_getmusic] ckpt        = ${CKPT}"
echo "[batch_infer_xmidi_getmusic] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "[batch_infer_xmidi_getmusic] python      = ${PY}"
echo "[batch_infer_xmidi_getmusic] block_size  = ${BLOCK_SIZE}  bsz=${BSZ}  T=${T_STEPS}"
echo "[batch_infer_xmidi_getmusic] model       = ${MODEL}"
echo "[batch_infer_xmidi_getmusic] weights     = ${WEIGHT_SOURCE}"
echo "[batch_infer_xmidi_getmusic] out         = ${OUT}"
echo "[batch_infer_xmidi_getmusic] start_idx   = ${START_IDX}"
[ -n "$MAX_SAMPLES" ] && echo "[batch_infer_xmidi_getmusic] max_samples = ${MAX_SAMPLES}"

EXTRA_ARGS=()
[ -n "$MAX_SAMPLES" ] && EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
[ "${RESUME:-0}" = "1" ] && EXTRA_ARGS+=(--resume)

# Default log name is keyed on (block_size, T). When the same shell is
# reused to run two ckpts (e.g. best.ckpt then last.ckpt), the second
# run would otherwise stomp on the first's log. Override with LOGNAME=...
# (or LOGNAME=$(basename "$OUT" .jsonl)) to keep them side by side.
LOGNAME=${LOGNAME:-batch_infer_xmidi_getmusic_bs${BLOCK_SIZE}_T${T_STEPS}}

${PY} -u scripts/eval/batch_infer_xmidi_getmusic.py \
    --ckpt       "${CKPT}" \
    --model      "${MODEL}" \
    --block_size ${BLOCK_SIZE} \
    --bsz        ${BSZ} \
    --T          ${T_STEPS} \
    --out        "${OUT}" \
    --start_idx  ${START_IDX} \
    --seed       ${SEED} \
    --weight_source "${WEIGHT_SOURCE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee logs/${LOGNAME}.log
