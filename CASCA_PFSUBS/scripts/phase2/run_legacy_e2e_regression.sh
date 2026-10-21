#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/setup_env.sh"
V2_ROOT=$(cd "$ROOT/../CASCA_base" && pwd)
PY=${PYBIN:-$(command -v python)}
OUT=${ROOT}/outputs/phase2-legacy-e2e-regression-v2
: "${BD3LM_DATA_ROOT:?Set BD3LM_DATA_ROOT to external XMIDI data}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_PROJECT=CASCA_focal
mkdir -p "${OUT}/v2" "${OUT}/focal" "${ROOT}/logs/phase2"

COMMON_OVERRIDES=(
  mode=train
  data=xmidi_getmusic
  model=xmidi_getmusic_tiny
  model.attn_backend=flex
  algo=xmidi_getmusic_bd3lm
  block_size=16
  loader.global_batch_size=1
  loader.eval_global_batch_size=1
  loader.batch_size=1
  loader.eval_batch_size=1
  loader.num_workers=0
  trainer.devices=1
  trainer.num_nodes=1
  trainer.max_steps=1
  trainer.val_check_interval=1
  trainer.log_every_n_steps=1
  trainer.limit_train_batches=1
  trainer.limit_val_batches=0
  trainer.num_sanity_val_steps=0
  trainer.precision=bf16-mixed
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1
  optim.lr=3e-5
  training.resample=False
  training.from_pretrained=null
  training.ema=0.9999
  seed=1
  wandb.project=CASCA_focal
  wandb.group=phase2-legacy-e2e-regression
)

cd "${V2_ROOT}"
BD3LM_ROOT="${V2_ROOT}" "${PY}" -u main.py \
  hydra.run.dir="${OUT}/v2" \
  checkpointing.save_dir="${OUT}/v2" \
  wandb.name=phase2-v2-subs-tiny-scratch-s1-reference \
  wandb.id=phase2-v2-subs-tiny-scratch-s1-reference \
  "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee "${ROOT}/logs/phase2/legacy_e2e_v2.log"

cd "${ROOT}"
export BD3LM_ROOT="${ROOT}"
export BD3LM_DATA_ROOT
"${PY}" -u main.py \
  --use_focal false \
  hydra.run.dir="${OUT}/focal" \
  checkpointing.save_dir="${OUT}/focal" \
  wandb.name=phase2-v2-subs-tiny-scratch-s1-integrated \
  wandb.id=phase2-v2-subs-tiny-scratch-s1-integrated \
  "${COMMON_OVERRIDES[@]}" \
  2>&1 | tee "${ROOT}/logs/phase2/legacy_e2e_focal.log"
