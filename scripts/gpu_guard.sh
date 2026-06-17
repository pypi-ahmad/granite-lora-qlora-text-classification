#!/usr/bin/env bash
set -euo pipefail

TARGET_FREE_MIB="${1:-7600}"
GRACE_SECONDS="${2:-8}"
POLL_SECONDS="${3:-2}"
TIMEOUT_SECONDS="${4:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; NVIDIA driver/tooling is required." >&2
  exit 1
fi

list_python_gpu_pids() {
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '
      NF >= 2 {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        name=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
        if (tolower(name) ~ /python/) {
          print $1
        }
      }
    ' \
    | sort -u
}

kill_python_gpu_pids() {
  local -a pids=()
  mapfile -t pids < <(list_python_gpu_pids)

  if ((${#pids[@]} == 0)); then
    echo "No active Python GPU processes detected."
    return
  fi

  echo "Terminating Python GPU PIDs: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep "${GRACE_SECONDS}"

  mapfile -t pids < <(list_python_gpu_pids)
  if ((${#pids[@]} > 0)); then
    echo "Force killing remaining Python GPU PIDs: ${pids[*]}"
    kill -9 "${pids[@]}" 2>/dev/null || true
  fi
}

read_gpu_free_mib() {
  local line total used free
  line="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n1)"
  total="$(echo "${line}" | awk -F',' '{gsub(/[[:space:]]+/, "", $1); print $1}')"
  used="$(echo "${line}" | awk -F',' '{gsub(/[[:space:]]+/, "", $2); print $2}')"
  free=$((total - used))
  echo "${free}"
}

wait_for_vram() {
  local start_ts now elapsed free_mib
  start_ts="$(date +%s)"
  while true; do
    free_mib="$(read_gpu_free_mib)"
    echo "GPU free memory: ${free_mib} MiB (target >= ${TARGET_FREE_MIB} MiB)"
    if ((free_mib >= TARGET_FREE_MIB)); then
      break
    fi

    if ((TIMEOUT_SECONDS > 0)); then
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      if ((elapsed >= TIMEOUT_SECONDS)); then
        echo "Timed out waiting for GPU free memory target after ${elapsed}s." >&2
        exit 1
      fi
    fi

    sleep "${POLL_SECONDS}"
  done
}

echo "GPU guard start: target_free=${TARGET_FREE_MIB}MiB grace=${GRACE_SECONDS}s poll=${POLL_SECONDS}s timeout=${TIMEOUT_SECONDS}s"
kill_python_gpu_pids
wait_for_vram
echo "GPU guard complete."
