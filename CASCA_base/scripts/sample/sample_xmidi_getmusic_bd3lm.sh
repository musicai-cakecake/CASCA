#!/bin/bash
# ============================================================
# Source-conditioned XMIDI-GetMusic sampling.
#
# Pulls source rows + target prompts from the validation split (via the
# standard get_dataloaders pipeline) and generates up to 12 x 496 = 5952
# target content tokens for each sample using the BD3-LM block-diffusion
# semi-AR sampler. UNK rows are pre-filled with <UNK> and skipped.
#
# Usage:
#   bash scripts/sample/sample_xmidi_getmusic_bd3lm.sh <CKPT_PATH> [BLOCK_SIZE] [BSZ] [NUM_BATCHES] [T] [MODEL]
#
# Defaults:
#   CKPT_PATH:    required
#   BLOCK_SIZE:   16
#   BSZ:          1   (first_hitting requires bsz=1)
#   NUM_BATCHES:  1
#   T:            1000 diffusion steps per content block
#   MODEL:        xmidi_getmusic_small
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f ./setup_env.sh ]; then
  source ./setup_env.sh
fi

CKPT_PATH=${1:?"usage: $0 <CKPT_PATH> [BLOCK_SIZE] [BSZ] [NUM_BATCHES] [T] [MODEL]"}
CKPT_PATH=$(cd "$(dirname "$CKPT_PATH")" && pwd)/$(basename "$CKPT_PATH")
BLOCK_SIZE=${2:-16}
BSZ=${3:-1}
NUM_BATCHES=${4:-1}
T_STEPS=${5:-1000}
MODEL=${6:-xmidi_getmusic_small}
SEED=${SEED:-1}

# XMIDI-GetMusic sampling is single-GPU (first_hitting=true requires
# eval_batch_size=1 and Lightning's eval_global_batch_size must be
# divisible by num_gpus).
export CUDA_VISIBLE_DEVICES=${SAMPLE_DEVICE:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

LOGDIR=${BD3LM_ROOT:-$(pwd)}/sample_logs/xmidi_getmusic_bs${BLOCK_SIZE}_T${T_STEPS}
# `sampling.logdir` is actually a CSV *file* path (legacy misnomer);
# make sure its parent directory exists rather than the file itself.
mkdir -p "$(dirname "$LOGDIR")"
mkdir -p logs

PY=${PYBIN:-$(command -v python)}
if [ ! -x "${PY}" ]; then
  PY=$(command -v python)
fi

echo "[sample_xmidi_getmusic] ckpt        = ${CKPT_PATH}"
echo "[sample_xmidi_getmusic] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "[sample_xmidi_getmusic] python      = ${PY}"
echo "[sample_xmidi_getmusic] block_size  = ${BLOCK_SIZE}  bsz=${BSZ}  num_batches=${NUM_BATCHES}  T=${T_STEPS}"
echo "[sample_xmidi_getmusic] model       = ${MODEL}"

${PY} -u main.py \
    mode=sample_eval \
    data=xmidi_getmusic \
    model=${MODEL} \
    model.attn_backend=sdpa \
    algo=xmidi_getmusic_bd3lm \
    algo.T=${T_STEPS} \
    block_size=${BLOCK_SIZE} \
    loader.batch_size=${BSZ} \
    loader.eval_batch_size=${BSZ} \
    loader.global_batch_size=${BSZ} \
    loader.eval_global_batch_size=${BSZ} \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    eval.checkpoint_path=${CKPT_PATH} \
    eval.disable_ema=False \
    sampling.num_sample_batches=${NUM_BATCHES} \
    sampling.first_hitting=true \
    sampling.kv_cache=true \
    sampling.var_length=false \
    sampling.nucleus_p=0.95 \
    sampling.logdir=${LOGDIR} \
    seed=${SEED} \
    wandb=null \
    2>&1 | tee logs/sample_xmidi_getmusic_bs${BLOCK_SIZE}_T${T_STEPS}.log
