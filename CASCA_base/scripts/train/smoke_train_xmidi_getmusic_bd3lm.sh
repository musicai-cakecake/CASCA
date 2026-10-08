#!/bin/bash
# ============================================================
# [SMOKE] BD3-LM training on XMIDI-GetMusic.
#
# Validates the full pipeline (XMIDIGetmusicDataset/Tokenizer +
# region-aware diffusion + UNK-row attention + 2D RoPE on 16x512)
# end-to-end with a tiny model. Runs ~50 steps + 2-batch validation.
#
# Usage:
#   bash scripts/train/smoke_train_xmidi_getmusic_bd3lm.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODEL=xmidi_getmusic_small \
#     MAX_STEPS=1 VAL_CHECK=1 LIMIT_VAL=1 \
#     bash scripts/train/smoke_train_xmidi_getmusic_bd3lm.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [ -f ./setup_env.sh ]; then
  source ./setup_env.sh
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
[[ "${N_GPUS}" -lt 1 ]] && N_GPUS=1
BLOCK_SIZE=${BLOCK_SIZE:-16}
BSZ=${BSZ:-1}                    # per-GPU batch (8192 seqlen + 142421 vocab is heavy)
GLOBAL_BSZ=${GLOBAL_BSZ:-$((BSZ * N_GPUS))}
MAX_STEPS=${MAX_STEPS:-50}
VAL_CHECK=${VAL_CHECK:-20}
LIMIT_VAL=${LIMIT_VAL:-2}
LIMIT_TRAIN=${LIMIT_TRAIN:-1.0}
MODEL=${MODEL:-xmidi_getmusic_tiny}
LR=${LR:-3e-5}
SEED=${SEED:-1}

RUN_NAME=smoke-bd3lms-music-v2-bs${BLOCK_SIZE}-${MODEL}
HYDRA_DIR=${BD3LM_ROOT:-$(pwd)}/outputs/${RUN_NAME}
mkdir -p logs

echo "[smoke_xmidi_getmusic] run name = ${RUN_NAME}"
echo "[smoke_xmidi_getmusic] hydra dir = ${HYDRA_DIR}"
echo "[smoke_xmidi_getmusic] block_size=${BLOCK_SIZE} GPUs=${N_GPUS} per-GPU bsz=${BSZ} global_bsz=${GLOBAL_BSZ} max_steps=${MAX_STEPS}"

PY=${PYBIN:-$(command -v python)}
if [ ! -x "${PY}" ]; then
  PY=$(command -v python)
fi
echo "[smoke_xmidi_getmusic] python = ${PY}"

# ----- W&B logging (default off for smoke; flip to on with WANDB=on) -----
WANDB=${WANDB:-off}
WANDB_RUN_TAG=${WANDB_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}-${WANDB_RUN_TAG}}
WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN_NAME}-${WANDB_RUN_TAG}-s${SEED}}
if [ "${WANDB}" = "on" ]; then
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[smoke_xmidi_getmusic] ERROR: WANDB=on but WANDB_API_KEY is empty."
    echo "  Put your key in ${BD3LM_ROOT:-$(pwd)}/.wandb_env or prefix"
    echo "  the command with WANDB_API_KEY=<key>."
    exit 1
  fi
  WANDB_OVERRIDES="wandb.project=BD3-LMs \
    wandb.name=${WANDB_RUN_NAME} \
    wandb.id=${WANDB_RUN_ID} \
    wandb.group=smoke_xmidi_getmusic"
else
  WANDB_OVERRIDES="wandb=null"
fi

${PY} -u main.py \
    hydra.run.dir=${HYDRA_DIR} \
    mode=train \
    data=xmidi_getmusic \
    model=${MODEL} \
    model.attn_backend=flex \
    algo=xmidi_getmusic_bd3lm \
    block_size=${BLOCK_SIZE} \
    loader.global_batch_size=${GLOBAL_BSZ} \
    loader.eval_global_batch_size=${GLOBAL_BSZ} \
    loader.batch_size=${BSZ} \
    loader.eval_batch_size=${BSZ} \
    loader.num_workers=2 \
    trainer.devices=${N_GPUS} \
    trainer.num_nodes=1 \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=${VAL_CHECK} \
    trainer.log_every_n_steps=5 \
    trainer.limit_train_batches=${LIMIT_TRAIN} \
    trainer.limit_val_batches=${LIMIT_VAL} \
    trainer.num_sanity_val_steps=0 \
    trainer.precision=bf16-mixed \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${MAX_STEPS} \
    optim.lr=${LR} \
    training.resample=False \
    training.from_pretrained=null \
    training.ema=0.0 \
    seed=${SEED} \
    ${WANDB_OVERRIDES} \
    2>&1 | tee logs/${RUN_NAME}.log
