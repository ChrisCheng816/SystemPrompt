#!/usr/bin/env bash
# Dispatch a run to the interpreter its model requires.
#
# Two environments coexist because Qwen3.8-27B needs vllm>=0.19 and
# transformers>=5, while every previously evaluated model was measured under
# vllm 0.10.2 and transformers 4.57.6.  Running an old model under the new
# stack would not fail; it would silently produce numbers that do not match
# the published ones, so the mapping is enforced here rather than remembered.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Models that require the newer stack. Extend as needed.
new_stack_models=("Qwen/Qwen3.8-27B")

model=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --model-name) model="${arguments[index + 1]}" ;;
    --model-name=*) model="${arguments[index]#*=}" ;;
  esac
done

if [[ -z "$model" ]]; then
  echo "run.sh: --model-name is required so the interpreter can be chosen." >&2
  exit 2
fi

venv="$project_dir/.venv"
for candidate in "${new_stack_models[@]}"; do
  if [[ "$model" == "$candidate" ]]; then
    venv="$project_dir/.venv-qwen38"
  fi
done

if [[ ! -x "$venv/bin/python" ]]; then
  echo "run.sh: missing interpreter at $venv/bin/python" >&2
  exit 2
fi

echo "run.sh: $model -> $(basename "$venv")" >&2
exec "$venv/bin/python" "$project_dir/main.py" "${arguments[@]}"
