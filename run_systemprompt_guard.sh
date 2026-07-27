#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
gpu_devices="0,1,2,3"
leave_mb=512
reserve_mb=""

arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --gpu-devices)
      index=$((index + 1))
      gpu_devices="${arguments[index]}"
      ;;
    --gpu-devices=*)
      gpu_devices="${arguments[index]#*=}"
      ;;
    --gpu-reserve-free-mb)
      index=$((index + 1))
      leave_mb="${arguments[index]}"
      ;;
    --gpu-reserve-free-mb=*)
      leave_mb="${arguments[index]#*=}"
      ;;
    --gpu-reserve-mb)
      index=$((index + 1))
      reserve_mb="${arguments[index]}"
      ;;
    --gpu-reserve-mb=*)
      reserve_mb="${arguments[index]#*=}"
      ;;
  esac
done

if [[ "$reserve_mb" == "0" ]]; then
  exec "$python_bin" "$project_dir/main.py" "${arguments[@]}"
fi

guard_dir="$(mktemp -d "${TMPDIR:-/tmp}/systemprompt-gpu-guard.XXXXXX")"
pids=()

cleanup() {
  touch "$guard_dir/release" 2>/dev/null || true
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  rm -rf "$guard_dir"
}
trap cleanup EXIT INT TERM

IFS=',' read -r -a gpu_ids <<< "$gpu_devices"
for gpu_id in "${gpu_ids[@]}"; do
  gpu_id="${gpu_id//[[:space:]]/}"
  guard_args=(--gpu-id "$gpu_id" --guard-dir "$guard_dir" --leave-mb "$leave_mb")
  if [[ -n "$reserve_mb" ]]; then
    guard_args+=(--reserve-mb "$reserve_mb")
  fi
  "$python_bin" "$project_dir/gpu_guard.py" "${guard_args[@]}" >"$guard_dir/guard.$gpu_id.log" 2>&1 &
  pids+=("$!")
done

deadline=$((SECONDS + 60))
while true; do
  ready_count="$(find "$guard_dir" -maxdepth 1 -name 'ready.*' | wc -l)"
  if [[ "$ready_count" -eq "${#gpu_ids[@]}" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    cat "$guard_dir"/*.log 2>/dev/null || true
    echo "Timed out waiting for GPU guards." >&2
    exit 1
  fi
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      cat "$guard_dir"/*.log 2>/dev/null || true
      echo "A GPU guard exited before becoming ready." >&2
      exit 1
    fi
  done
  sleep 0.05
done

SYSTEMPROMPT_GPU_GUARD_DIR="$guard_dir" "$python_bin" "$project_dir/main.py" "${arguments[@]}"
