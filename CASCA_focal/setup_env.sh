#!/usr/bin/env bash
export BD3LM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYBIN=${PYBIN:-$(command -v python)}
export BD3LM_EXPECTED_CONDA_ENV=${BD3LM_EXPECTED_CONDA_ENV:-}
export BD3LM_EXPECTED_PYTHON=${BD3LM_EXPECTED_PYTHON:-${PYBIN}}
export BD3LM_EXPECTED_TMUX=${BD3LM_EXPECTED_TMUX:-}
export WANDB_PROJECT=${WANDB_PROJECT:-CASCA_focal}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HOME=${HF_HOME:-${BD3LM_ROOT}/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ -n "${BD3LM_EXPECTED_CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${BD3LM_EXPECTED_CONDA_ENV}" ]]; then
  echo "Expected Conda environment ${BD3LM_EXPECTED_CONDA_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -n "${BD3LM_DATA_ROOT:-}" ]]; then
  export BD3LM_DATA_ROOT
fi
cd "${BD3LM_ROOT}"
echo "[CASCA_focal] root=${BD3LM_ROOT} python=${PYBIN}"
echo "[CASCA_focal] Set BD3LM_DATA_ROOT to external XMIDI data; FROM_PRETRAINED selects a CASCA_base checkpoint."
