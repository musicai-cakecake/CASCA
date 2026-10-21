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

TMUX_SESSION=$(tmux display-message -p '#S' 2>/dev/null || true)
TMUX_WINDOW=$(tmux display-message -p '#W' 2>/dev/null || true)
if [ -n "${BD3LM_EXPECTED_TMUX}" ] && [ "${TMUX_SESSION}" != "${BD3LM_EXPECTED_TMUX}" ] && [ "${TMUX_WINDOW}" != "${BD3LM_EXPECTED_TMUX}" ]; then
  echo "[train_xmidi_getmusic] ERROR: run inside tmux ${BD3LM_EXPECTED_TMUX}; current=${TMUX_SESSION:-none}:${TMUX_WINDOW:-none}" >&2
  exit 1
fi

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
NUM_WORKERS=${NUM_WORKERS:-8}
ENABLE_PROGRESS_BAR=${ENABLE_PROGRESS_BAR:-true}
FROM_PRETRAINED=${FROM_PRETRAINED:-null}
if [[ "${FROM_PRETRAINED}" != "null" && ! -f "${FROM_PRETRAINED}" ]]; then
  echo "[train_xmidi_getmusic] ERROR: FROM_PRETRAINED not found: ${FROM_PRETRAINED}" >&2
  exit 1
fi
USE_FOCAL=$(printf '%s' "${USE_FOCAL:-false}" | tr '[:upper:]' '[:lower:]')
case "${USE_FOCAL}" in
  true|false) ;;
  *) echo "[train_xmidi_getmusic] ERROR: USE_FOCAL must be true or false, got ${USE_FOCAL}" >&2; exit 1 ;;
esac
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
FOCAL_ALPHA_PAD=${FOCAL_ALPHA_PAD:-0.8600010833864916}
FOCAL_ALPHA_EVENT=${FOCAL_ALPHA_EVENT:-1.2900016250797375}
FOCAL_PAD_TOKEN_ID=${FOCAL_PAD_TOKEN_ID:-605}
FOCAL_ALPHA_NORMALIZATION=${FOCAL_ALPHA_NORMALIZATION:-none}
ENABLE_PHASE2_TELEMETRY=$(printf '%s' "${ENABLE_PHASE2_TELEMETRY:-false}" | tr '[:upper:]' '[:lower:]')
case "${ENABLE_PHASE2_TELEMETRY}" in
  true|false) ;;
  *) echo "[train_xmidi_getmusic] ERROR: ENABLE_PHASE2_TELEMETRY must be true or false" >&2; exit 1 ;;
esac
LOSS_LABEL=subs
[[ "${USE_FOCAL}" == "true" ]] && LOSS_LABEL=focal-g${FOCAL_GAMMA}-ap${FOCAL_ALPHA_PAD}-ae${FOCAL_ALPHA_EVENT}

RUN_NAME=${RUN_NAME:-v2-${LOSS_LABEL}-${MODEL}-bs${BLOCK_SIZE}-scratch-s${SEED}}
HYDRA_DIR=${HYDRA_DIR:-${BD3LM_ROOT}/outputs/${RUN_NAME}}
HYDRA_DIR=$(realpath -m "${HYDRA_DIR}")
PHASE2_TELEMETRY_DIR=${PHASE2_TELEMETRY_DIR:-${HYDRA_DIR}/phase2_telemetry}
case "${HYDRA_DIR}" in
  "${BD3LM_ROOT}"/*) ;;
  *) echo "[train_xmidi_getmusic] ERROR: HYDRA_DIR must stay under ${BD3LM_ROOT}: ${HYDRA_DIR}" >&2; exit 1 ;;
esac

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
echo "[train_xmidi_getmusic] pretrained    = ${FROM_PRETRAINED}"
echo "[train_xmidi_getmusic] use_focal     = ${USE_FOCAL}"
echo "[train_xmidi_getmusic] focal params  = gamma=${FOCAL_GAMMA} alpha_pad=${FOCAL_ALPHA_PAD} alpha_event=${FOCAL_ALPHA_EVENT} pad_id=${FOCAL_PAD_TOKEN_ID} normalization=${FOCAL_ALPHA_NORMALIZATION}"
echo "[train_xmidi_getmusic] progress bar  = ${ENABLE_PROGRESS_BAR}"
echo "[train_xmidi_getmusic] phase2 telemetry = ${ENABLE_PHASE2_TELEMETRY}"

PHASE2_OVERRIDES=()
if [[ "${ENABLE_PHASE2_TELEMETRY}" == "true" ]]; then
  PHASE2_OVERRIDES+=(
    "+callbacks.phase2_performance._target_=xmidi_getmusic.phase2_performance.Phase2PerformanceCallback"
    "+callbacks.phase2_performance.output_dir=${PHASE2_TELEMETRY_DIR}"
  )
fi

mkdir -p logs

PY=${BD3LM_EXPECTED_PYTHON}
if [ ! -x "${PY}" ] || [ "$(readlink -f "${PY}")" != "$(readlink -f "$(command -v python)")" ]; then
  echo "[train_xmidi_getmusic] ERROR: required Python is unavailable or inactive: ${BD3LM_EXPECTED_PYTHON}" >&2
  exit 1
fi
echo "[train_xmidi_getmusic] python        = ${PY}"
echo "[train_xmidi_getmusic] tmux         = ${TMUX_SESSION}:${TMUX_WINDOW}"

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
  export WANDB_DIR=${WANDB_DIR:-${HYDRA_DIR}/wandb}
  WANDB_DIR=$(realpath -m "${WANDB_DIR}")
  case "${WANDB_DIR}" in
    "${BD3LM_ROOT}"/*) ;;
    *) echo "[train_xmidi_getmusic] ERROR: WANDB_DIR must stay under ${BD3LM_ROOT}: ${WANDB_DIR}" >&2; exit 1 ;;
  esac
  export WANDB_DIR
  mkdir -p "${WANDB_DIR}"
  echo "[train_xmidi_getmusic] wandb         = on (mode=${WANDB_MODE} project=${WANDB_PROJECT})"
  echo "[train_xmidi_getmusic] wandb dir     = ${WANDB_DIR}"
  echo "[train_xmidi_getmusic] wandb run id  = ${WANDB_RUN_ID}"
  echo "[train_xmidi_getmusic] wandb run name= ${WANDB_RUN_NAME}"
  WANDB_OVERRIDES="wandb.project=${WANDB_PROJECT} \
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
    --use_focal ${USE_FOCAL} \
    loss.gamma=${FOCAL_GAMMA} \
    loss.alpha_pad=${FOCAL_ALPHA_PAD} \
    loss.alpha_event=${FOCAL_ALPHA_EVENT} \
    loss.pad_token_id=${FOCAL_PAD_TOKEN_ID} \
    loss.alpha_normalization=${FOCAL_ALPHA_NORMALIZATION} \
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
    +trainer.enable_progress_bar=${ENABLE_PROGRESS_BAR} \
    trainer.log_every_n_steps=100 \
    trainer.precision=bf16-mixed \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=${CKPT_INTVL} \
    "${PHASE2_OVERRIDES[@]}" \
    optim.lr=${LR} \
    lr_scheduler.num_warmup_steps=${WARMUP_STEPS} \
    training.resample=False \
    training.ema=0.9999 \
    training.from_pretrained=${FROM_PRETRAINED} \
    seed=${SEED} \
    ${RESUME_OVERRIDES} \
    ${WANDB_OVERRIDES} \
    "$@" \
    2>&1 | tee -a logs/${RUN_NAME}.log
