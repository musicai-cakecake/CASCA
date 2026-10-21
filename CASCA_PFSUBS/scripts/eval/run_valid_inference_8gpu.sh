#!/usr/bin/env bash
# Full Focal V2 valid-set inference, strict merge and evaluation pipeline.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"
source ./setup_env.sh

PY=${PYBIN:-$(command -v python)}
CKPT=${CKPT:?Set CKPT to an external CASCA_focal checkpoint}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs/CASCA_focal/inference_valid_best_ema}
MODEL=${MODEL:-xmidi_getmusic_small}
WEIGHT_SOURCE=${WEIGHT_SOURCE:-ema}
USE_FOCAL=$(printf '%s' "${USE_FOCAL:-true}" | tr '[:upper:]' '[:lower:]')
SEED=${SEED:-1}
T=${T:-1000}
TOP_P=${TOP_P:-0.95}
VALID_SIZE=${VALID_SIZE:-6817}
EMBED_DEVICE=${EMBED_DEVICE:-cuda}

case "${USE_FOCAL}" in
  true|1|yes|on) FOCAL_ARG=(--use_focal) ;;
  false|0|no|off) FOCAL_ARG=(--no-use_focal) ;;
  *) echo "[error] USE_FOCAL must be true or false, got ${USE_FOCAL}" >&2; exit 2 ;;
esac

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/shards" "${OUT_ROOT}/logs" "${OUT_ROOT}/evaluation"

# Contiguous, non-overlapping ranges covering 0..6816.
STARTS=(0 853 1706 2559 3412 4264 5115 5966)
COUNTS=(853 853 853 853 852 851 851 851)
# The final shard absorbs the remainder so the union is exactly 0..VALID_SIZE-1.
COUNTS[7]=$((VALID_SIZE - STARTS[7]))
PIDS=()

for gpu in $(seq 0 7); do
  shard=$(printf '%s/shards/shard_%02d.jsonl' "${OUT_ROOT}" "${gpu}")
  log=$(printf '%s/logs/shard_%02d.log' "${OUT_ROOT}" "${gpu}")
  echo "[launch] gpu=${gpu} start=${STARTS[$gpu]} count=${COUNTS[$gpu]} out=${shard}"
  CUDA_VISIBLE_DEVICES=${gpu} \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  "${PY}" -u scripts/eval/batch_infer_xmidi_getmusic.py \
    --ckpt "${CKPT}" \
    --model "${MODEL}" \
    --weight_source "${WEIGHT_SOURCE}" \
    --block_size 16 \
    --bsz 1 \
    --T "${T}" \
    --nucleus_p "${TOP_P}" \
    --seed "${SEED}" \
    --start_idx "${STARTS[$gpu]}" \
    --max_samples "${COUNTS[$gpu]}" \
    "${FOCAL_ARG[@]}" \
    --out "${shard}" \
    >"${log}" 2>&1 &
  PIDS+=("$!")
done

failed=0
for gpu in $(seq 0 7); do
  if ! wait "${PIDS[$gpu]}"; then
    echo "[error] shard ${gpu} failed; see ${OUT_ROOT}/logs/shard_$(printf '%02d' "${gpu}").log" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

MERGED=${OUT_ROOT}/valid_inference.jsonl
"${PY}" scripts/eval/merge_xmidi_getmusic_shards.py \
  --shards_dir "${OUT_ROOT}/shards" \
  --out "${MERGED}" \
  --expected_size "${VALID_SIZE}" \
  | tee "${OUT_ROOT}/logs/merge.log"

CUDA_VISIBLE_DEVICES=0 \
"${PY}" scripts/eval/objective_evaluation.py \
  --jsonl "${MERGED}" \
  --out_dir "${OUT_ROOT}/evaluation/objective" \
  --ckpt "${CKPT}" \
  --weight_source "${WEIGHT_SOURCE}" \
  --dict_path "${BD3LM_DATA_ROOT}/whole_dict.txt" \
  --device "${EMBED_DEVICE}" \
  --expect_valid_size "${VALID_SIZE}" \
  | tee "${OUT_ROOT}/logs/objective_evaluation.log"

"${PY}" scripts/eval/token_distribution_xmidi_getmusic.py \
  --jsonl "${MERGED}" \
  --out_dir "${OUT_ROOT}/evaluation/token_distribution" \
  | tee "${OUT_ROOT}/logs/token_distribution.log"

"${PY}" scripts/eval/pad_event_evaluation.py \
  --jsonl "${MERGED}" \
  --expect_valid_size "${VALID_SIZE}" \
  --out_json "${OUT_ROOT}/evaluation/pad_event_summary.json" \
  --out_csv "${OUT_ROOT}/evaluation/pad_event_summary.csv" \
  | tee "${OUT_ROOT}/logs/pad_event_evaluation.log"

CUDA_VISIBLE_DEVICES=0 \
"${PY}" scripts/eval/evaluate_xmidi_getmusic.py \
  --ckpt "${CKPT}" \
  --model "${MODEL}" \
  --block_size 16 \
  --bsz 1 \
  --num_t_samples 4 \
  --max_batches 0 \
  --expect_valid_size "${VALID_SIZE}" \
  --weight_source "${WEIGHT_SOURCE}" \
  "${FOCAL_ARG[@]}" \
  --seed "${SEED}" \
  --out_json "${OUT_ROOT}/evaluation/nelbo_summary.json" \
  --out_csv "${OUT_ROOT}/evaluation/nelbo_summary.csv" \
  | tee "${OUT_ROOT}/logs/nelbo_evaluation.log"

echo "[done] generated results: ${MERGED}"
echo "[done] evaluation results: ${OUT_ROOT}/evaluation"
