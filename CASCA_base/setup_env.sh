#!/usr/bin/env bash
export BD3LM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYBIN=${PYBIN:-$(command -v python)}
export HF_HOME=${HF_HOME:-${BD3LM_ROOT}/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export WANDB_PROJECT=${WANDB_PROJECT:-CASCA_base}
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ -n "${BD3LM_DATA_ROOT:-}" ]]; then
  export BD3LM_DATA_ROOT
fi
cd "${BD3LM_ROOT}"
echo "[CASCA_base] root=${BD3LM_ROOT} python=${PYBIN}"
echo "[CASCA_base] Set BD3LM_DATA_ROOT to the external XMIDI dataset; use CKPT/RESUME_CKPT for external weights."
