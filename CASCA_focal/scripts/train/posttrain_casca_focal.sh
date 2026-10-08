#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${FROM_PRETRAINED:?Set FROM_PRETRAINED to an external CASCA_base Lightning checkpoint}"
: "${BD3LM_DATA_ROOT:?Set BD3LM_DATA_ROOT to the external XMIDI dataset}"
[[ -z "${RESUME_CKPT:-}" ]] || { echo 'This entry point warm-starts, not resumes; unset RESUME_CKPT' >&2; exit 1; }
export FROM_PRETRAINED=$(realpath -e "$FROM_PRETRAINED")
export USE_FOCAL=true
export FOCAL_GAMMA=${FOCAL_GAMMA:-0.25}
export FOCAL_ALPHA_PAD=${FOCAL_ALPHA_PAD:-0.26567013281085866}
export FOCAL_ALPHA_EVENT=${FOCAL_ALPHA_EVENT:-1.210166680846564}
export FOCAL_ALPHA_NORMALIZATION=none
export MAX_STEPS=${MAX_STEPS:-225}
export LR=${LR:-3e-6}
export WARMUP_STEPS=${WARMUP_STEPS:-100}
export GLOBAL_BSZ=${GLOBAL_BSZ:-64}
export BSZ=${BSZ:-1}
export NUM_WORKERS=${NUM_WORKERS:-2}
export CKPT_INTVL=${CKPT_INTVL:-$MAX_STEPS}
export ENABLE_PROGRESS_BAR=false
export RUN_NAME=${RUN_NAME:-CASCA_focal-posttrain-$(date +%Y%m%d-%H%M%S)}
exec bash "$ROOT/scripts/train/train_xmidi_getmusic_bd3lm.sh" \
  trainer.num_sanity_val_steps=0 trainer.limit_val_batches=0 trainer.log_every_n_steps=25 "$@"
