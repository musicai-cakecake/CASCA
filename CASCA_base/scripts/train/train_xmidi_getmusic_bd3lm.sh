#!/bin/bash
# ============================================================
# Single-stage BD3-LM training on XMIDI-GetMusic (R&D_Requirements_for_getmusic.md).
#
# Trains the weight-tied ``xmidi_getmusic_small`` DiT on the 16x512 grid
# (flat seqlen=8192, vocab=142421) with region-aware diffusion,
# per-sample inactive-row attention, axis-local 2D RoPE, and tied chunked
# output projection.
#
# Hardware target: 8 x 80GB GPUs (per setup_env.sh defaults).
# FlexAttention uses a compact BlockMask instead of materializing the
# B x 1 x 16384 x 16384 training mask. Per-GPU micro-batch defaults to 1.
#
# Usage:
#   bash scripts/train/train_xmidi_getmusic_bd3lm.sh
#   GLOBAL_BSZ=128 BSZ=2 bash scripts/train/train_xmidi_getmusic_bd3lm.sh
#   MODEL=xmidi_getmusic_medium bash scripts/train/train_xmidi_getmusic_bd3lm.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."
source ./setup_env.sh

# Force HuggingFace into offline mode: metrics.py/Diffusion init may try to
# fetch gpt2-large for generative-PPL bookkeeping. We never use that for
# XMIDI (symbolic music), but the constructor still runs and would hang
# for many minutes retrying network calls on offline nodes. The local
# hf_cache/ already contains gpt2-large from earlier runs.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# Allow opting into offline wandb on nodes without network to api.wandb.ai.
# Default is online; set WANDB_MODE=offline to log locally and `wandb sync`
# the run later from a node with internet.
export WANDB_MODE=${WANDB_MODE:-online}

BLOCK_SIZE=${BLOCK_SIZE:-16}
GLOBAL_BSZ=${GLOBAL_BSZ:-64}     # global eff. batch (8 GPUs * BSZ)
BSZ=${BSZ:-1}                    # per-GPU micro-batch
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_STEPS=${MAX_STEPS:-300000}
# V2 train: 133646 samples / GLOBAL_BSZ=64 ≈ 2088 steps/epoch.
# valid: 6817 / GLOBAL_BSZ=64 ≈ 107 validation batches.
# Lightning requires val_check_interval <= batches_per_epoch UNLESS
# check_val_every_n_epoch=null, so we set it to null.
VAL_CHECK=${VAL_CHECK:-1000}
CKPT_INTVL=${CKPT_INTVL:-2000}
MODEL=${MODEL:-xmidi_getmusic_small}
SEED=${SEED:-1}
# Peak LR (after warmup). XMIDI was unstable at 3e-4 for the 110M small
# model; the larger vocab embedding makes XMIDI-GetMusic even softer to
# instabilities. Default to the same conservative 3e-5 used in v1.
LR=${LR:-3e-5}
WARMUP_STEPS=${WARMUP_STEPS:-2500}

RUN_NAME=${RUN_NAME:-bd3lms-music-v2-${MODEL}-bs${BLOCK_SIZE}}
HYDRA_DIR=${HYDRA_DIR:-${BD3LM_ROOT}/outputs/${RUN_NAME}}

# Auto-detect the number of GPUs from CUDA_VISIBLE_DEVICES.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  N_GPUS=$(nvidia-smi -L | wc -l)
else
  N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
fi
[[ "${N_GPUS}" -lt 1 ]] && N_GPUS=1

echo "[train_xmidi_getmusic] run name      = ${RUN_NAME}"
echo "[train_xmidi_getmusic] hydra dir     = ${HYDRA_DIR}"
echo "[train_xmidi_getmusic] block_size    = ${BLOCK_SIZE}"
echo "[train_xmidi_getmusic] num GPUs      = ${N_GPUS}"
echo "[train_xmidi_getmusic] global bsz    = ${GLOBAL_BSZ}"
echo "[train_xmidi_getmusic] per-GPU bsz   = ${BSZ}"
echo "[train_xmidi_getmusic] max_steps     = ${MAX_STEPS}"
echo "[train_xmidi_getmusic] peak LR       = ${LR}  (warmup=${WARMUP_STEPS})"
echo "[train_xmidi_getmusic] model         = ${MODEL}"

mkdir -p logs

PY=${PYBIN:-$(command -v python)}
if [ ! -x "${PY}" ]; then
  PY=$(command -v python)
fi
echo "[train_xmidi_getmusic] python        = ${PY}"

# ----- W&B logging (default on; set WANDB=off to disable) -----
WANDB=${WANDB:-on}
WANDB_RUN_TAG=${WANDB_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}-${WANDB_RUN_TAG}}
WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN_NAME}-${WANDB_RUN_TAG}-s${SEED}}
if [ "${WANDB}" = "on" ]; then
  if [ "${WANDB_MODE}" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[train_xmidi_getmusic] ERROR: online W&B requested but WANDB_API_KEY is empty."
    echo "  Set WANDB_MODE=offline for local records, or provide the key via"
    echo "  the shell / ${BD3LM_ROOT}/.wandb_env."
    exit 1
  fi
  mkdir -p "${WANDB_DIR:-${HYDRA_DIR}/wandb}"
  export WANDB_DIR=${WANDB_DIR:-${HYDRA_DIR}/wandb}
  echo "[train_xmidi_getmusic] wandb         = on (mode=${WANDB_MODE} project=${WANDB_PROJECT:-BD3-LMs})"
  echo "[train_xmidi_getmusic] wandb dir     = ${WANDB_DIR}"
  echo "[train_xmidi_getmusic] wandb run id  = ${WANDB_RUN_ID}"
  echo "[train_xmidi_getmusic] wandb run name= ${WANDB_RUN_NAME}"
  WANDB_OVERRIDES="wandb.project=${WANDB_PROJECT:-BD3-LMs} \
    wandb.name=${WANDB_RUN_NAME} \
    wandb.id=${WANDB_RUN_ID} \
    wandb.group=xmidi_getmusic"
else
  echo "[train_xmidi_getmusic] wandb         = off"
  WANDB_OVERRIDES="wandb=null"
fi

RESUME_OVERRIDES="checkpointing.resume_from_ckpt=false"
if [ -n "${RESUME_CKPT:-}" ]; then
  if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[train_xmidi_getmusic] ERROR: RESUME_CKPT not found: ${RESUME_CKPT}"
    exit 1
  fi
  RESUME_OVERRIDES="checkpointing.resume_from_ckpt=true checkpointing.resume_ckpt_path=${RESUME_CKPT}"
  echo "[train_xmidi_getmusic] resume ckpt  = ${RESUME_CKPT}"
fi

${PY} -u main.py \
    hydra.run.dir=${HYDRA_DIR} \
    checkpointing.save_dir=${HYDRA_DIR} \
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
    loader.num_workers=${NUM_WORKERS} \
    trainer.devices=${N_GPUS} \
    trainer.num_nodes=1 \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=${VAL_CHECK} \
    +trainer.check_val_every_n_epoch=null \
    trainer.log_every_n_steps=100 \
    trainer.precision=bf16-mixed \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_INTVL} \
    optim.lr=${LR} \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    training.resample=False \
    training.ema=0.9999 \
    training.from_pretrained=null \
    seed=${SEED} \
    ${RESUME_OVERRIDES} \
    ${WANDB_OVERRIDES} \
    2>&1 | tee -a logs/${RUN_NAME}.log
