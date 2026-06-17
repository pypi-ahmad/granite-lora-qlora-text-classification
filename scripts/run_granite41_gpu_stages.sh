#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NB_PATH="${NB_PATH:-${PROJECT_ROOT}/granite41_lora_qlora_text_finetuning.ipynb}"
JUPYTER_NBCONVERT="${JUPYTER_NBCONVERT:-${PROJECT_ROOT}/.venv/bin/jupyter-nbconvert}"
GPU_GUARD_SCRIPT="${GPU_GUARD_SCRIPT:-${SCRIPT_DIR}/gpu_guard.sh}"

VRAM_FREE_TARGET_MIB="${VRAM_FREE_TARGET_MIB:-7600}"
GPU_GUARD_GRACE_SEC="${GPU_GUARD_GRACE_SEC:-8}"
GPU_GUARD_POLL_SEC="${GPU_GUARD_POLL_SEC:-2}"
GPU_GUARD_TIMEOUT_SEC="${GPU_GUARD_TIMEOUT_SEC:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/artifacts/runs/${RUN_ID}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ ! -f "${NB_PATH}" ]]; then
  echo "Notebook not found: ${NB_PATH}" >&2
  exit 1
fi

if [[ ! -x "${JUPYTER_NBCONVERT}" ]]; then
  echo "jupyter-nbconvert not executable: ${JUPYTER_NBCONVERT}" >&2
  exit 1
fi

if [[ ! -x "${GPU_GUARD_SCRIPT}" ]]; then
  echo "GPU guard script not executable: ${GPU_GUARD_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

lora_seq_ladder=(384 320 256 192 160 128 96 64)
qlora_seq_ladder=(384 320 256 192 160 128)

is_oom_failure() {
  local log_file="$1"
  rg -qi "cuda out of memory|out of memory" "${log_file}"
}

run_stage_once() {
  local stage="$1"
  local seq_len="$2"
  local attempt="$3"
  local output_name="${stage}_seq${seq_len}_attempt${attempt}.executed.ipynb"
  local output_path="${RUN_DIR}/${output_name}"
  local log_path="${RUN_DIR}/${stage}_seq${seq_len}_attempt${attempt}.log"

  echo ""
  echo "=== Stage ${stage} | attempt ${attempt} | MAX_SEQ_LENGTH=${seq_len} ==="
  echo "Output notebook: ${output_path}"
  echo "Log file: ${log_path}"

  if ((DRY_RUN == 1)); then
    echo "[dry-run] ${GPU_GUARD_SCRIPT} ${VRAM_FREE_TARGET_MIB} ${GPU_GUARD_GRACE_SEC} ${GPU_GUARD_POLL_SEC} ${GPU_GUARD_TIMEOUT_SEC}"
    echo "[dry-run] RUN_STAGE=${stage} FAIL_ON_STAGE_ERROR=true MAX_SEQ_LENGTH_OVERRIDE=${seq_len} ${JUPYTER_NBCONVERT} --to notebook --execute ${NB_PATH} --output ${output_name} --output-dir ${RUN_DIR}"
    return 0
  fi

  "${GPU_GUARD_SCRIPT}" "${VRAM_FREE_TARGET_MIB}" "${GPU_GUARD_GRACE_SEC}" "${GPU_GUARD_POLL_SEC}" "${GPU_GUARD_TIMEOUT_SEC}"

  set +e
  env \
    RUN_STAGE="${stage}" \
    FAIL_ON_STAGE_ERROR=true \
    MAX_SEQ_LENGTH_OVERRIDE="${seq_len}" \
    "${JUPYTER_NBCONVERT}" \
      --to notebook \
      --execute "${NB_PATH}" \
      --output "${output_name}" \
      --output-dir "${RUN_DIR}" \
      2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  if ((exit_code == 0)); then
    echo "Stage ${stage} attempt ${attempt} succeeded."
    return 0
  fi

  echo "Stage ${stage} attempt ${attempt} failed (exit ${exit_code})."
  return "${exit_code}"
}

run_stage_with_retry() {
  local stage="$1"
  shift
  local -a seq_ladder=("$@")
  local attempt=0

  for seq_len in "${seq_ladder[@]}"; do
    attempt=$((attempt + 1))

    if run_stage_once "${stage}" "${seq_len}" "${attempt}"; then
      return 0
    fi

    local log_path="${RUN_DIR}/${stage}_seq${seq_len}_attempt${attempt}.log"
    if ! is_oom_failure "${log_path}"; then
      echo "Stage ${stage} failed for a non-OOM reason. Aborting retries." >&2
      return 1
    fi

    echo "OOM detected for stage ${stage} at MAX_SEQ_LENGTH=${seq_len}; trying smaller sequence length."
  done

  echo "Stage ${stage} failed after exhausting sequence-length retries: ${seq_ladder[*]}" >&2
  return 1
}

run_stage_single() {
  local stage="$1"
  local seq_len="$2"
  run_stage_once "${stage}" "${seq_len}" 1
}

echo "Run ID: ${RUN_ID}"
echo "Run directory: ${RUN_DIR}"
echo "Notebook: ${NB_PATH}"
echo "GPU target free memory: ${VRAM_FREE_TARGET_MIB} MiB"
echo "Dry run: ${DRY_RUN}"

# Stage order is fixed for strict one-by-one execution.
run_stage_single "base" 384
run_stage_with_retry "lora" "${lora_seq_ladder[@]}"
run_stage_with_retry "qlora" "${qlora_seq_ladder[@]}"
run_stage_single "final" 384

echo "All stages completed."
