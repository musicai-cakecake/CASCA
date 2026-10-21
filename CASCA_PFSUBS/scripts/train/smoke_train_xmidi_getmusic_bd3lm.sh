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
source ./setup_env.sh
TMUX_SESSION=$(tmux display-message -p '#S' 2>/dev/null || true)
TMUX_WINDOW=$(tmux display-message -p '#W' 2>/dev/null || true)
if [ "${TMUX_SESSION}" != "${BD3LM_EXPECTED_TMUX}" ] && [ "${TMUX_WINDOW}" != "${BD3LM_EXPECTED_TMUX}" ]; then
  echo "[smoke_xmidi_getmusic] ERROR: run inside tmux ${BD3LM_EXPECTED_TMUX}; current=${TMUX_SESSION:-none}:${TMUX_WINDOW:-none}" >&2
  exit 1
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
USE_FOCAL=$(printf '%s' "${USE_FOCAL:-false}" | tr '[:upper:]' '[:lower:]')
case "${USE_FOCAL}" in
  true|false) ;;
  *) echo "[smoke_xmidi_getmusic] ERROR: USE_FOCAL must be true or false, got ${USE_FOCAL}" >&2; exit 1 ;;
esac
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
FOCAL_ALPHA_PAD=${FOCAL_ALPHA_PAD:-0.8600010833864916}
FOCAL_ALPHA_EVENT=${FOCAL_ALPHA_EVENT:-1.2900016250797375}
FOCAL_PAD_TOKEN_ID=${FOCAL_PAD_TOKEN_ID:-605}
FOCAL_ALPHA_NORMALIZATION=${FOCAL_ALPHA_NORMALIZATION:-none}
LOSS_LABEL=subs
[[ "${USE_FOCAL}" == "true" ]] && LOSS_LABEL=focal-g${FOCAL_GAMMA}-ap${FOCAL_ALPHA_PAD}-ae${FOCAL_ALPHA_EVENT}

RUN_NAME=${RUN_NAME:-phase2-smoke-${LOSS_LABEL}-${MODEL}-bs${BLOCK_SIZE}-scratch-s${SEED}}
HYDRA_DIR=$(realpath -m "${BD3LM_ROOT}/outputs/${RUN_NAME}")
case "${HYDRA_DIR}" in
  "${BD3LM_ROOT}"/*) ;;
  *) echo "[smoke_xmidi_getmusic] ERROR: HYDRA_DIR must stay under ${BD3LM_ROOT}: ${HYDRA_DIR}" >&2; exit 1 ;;
esac
mkdir -p logs

echo "[smoke_xmidi_getmusic] run name = ${RUN_NAME}"
echo "[smoke_xmidi_getmusic] hydra dir = ${HYDRA_DIR}"
echo "[smoke_xmidi_getmusic] block_size=${BLOCK_SIZE} GPUs=${N_GPUS} per-GPU bsz=${BSZ} global_bsz=${GLOBAL_BSZ} max_steps=${MAX_STEPS}"
echo "[smoke_xmidi_getmusic] use_focal=${USE_FOCAL} gamma=${FOCAL_GAMMA} alpha_pad=${FOCAL_ALPHA_PAD} alpha_event=${FOCAL_ALPHA_EVENT}"

PY=${BD3LM_EXPECTED_PYTHON}
if [ ! -x "${PY}" ] || [ "$(readlink -f "${PY}")" != "$(readlink -f "$(command -v python)")" ]; then
  echo "[smoke_xmidi_getmusic] ERROR: required Python is unavailable or inactive: ${BD3LM_EXPECTED_PYTHON}" >&2
  exit 1
fi
echo "[smoke_xmidi_getmusic] python = ${PY}"
echo "[smoke_xmidi_getmusic] tmux = ${TMUX_SESSION}:${TMUX_WINDOW}"

# ----- W&B logging (required; offline mode is allowed) -----
WANDB=${WANDB:-on}
WANDB_RUN_TAG=${WANDB_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}-${WANDB_RUN_TAG}}
WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN_NAME}-${WANDB_RUN_TAG}-s${SEED}}
if [ "${WANDB}" = "on" ]; then
  export WANDB_MODE=${WANDB_MODE:-online}
  if [ "${WANDB_MODE}" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[smoke_xmidi_getmusic] ERROR: online W&B requested but WANDB_API_KEY is empty."
    echo "  Set WANDB_MODE=offline or provide the key via ${BD3LM_ROOT}/.wandb_env."
    exit 1
  fi
  export WANDB_DIR=${HYDRA_DIR}/wandb
  mkdir -p "${WANDB_DIR}"
  WANDB_OVERRIDES="wandb.project=${WANDB_PROJECT} \
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
    +callbacks.phase2_performance._target_=xmidi_getmusic.phase2_performance.Phase2PerformanceCallback \
    +callbacks.phase2_performance.output_dir=${HYDRA_DIR}/phase2_telemetry \
    optim.lr=${LR} \
    training.resample=False \
    training.from_pretrained=null \
    training.ema=0.0 \
    seed=${SEED} \
    ${WANDB_OVERRIDES} \
    2>&1 | tee logs/${RUN_NAME}.log
