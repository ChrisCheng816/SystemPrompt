#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

sudo -v
sudo docker build -t mceval-sandbox:latest -f docker/mceval-sandbox.Dockerfile docker

WORKERS_PER_RUN="${WORKERS_PER_RUN:-4}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments_results_mceval}"
EVAL_RUNS="${EVAL_RUNS:-pass@1_t0 pass@1_t1 pass@5_t1}"
INCLUDE_MODEL="${INCLUDE_MODEL:-}"
EXCLUDE_MODEL="${EXCLUDE_MODEL:-}"
if [[ -n "$INCLUDE_MODEL" && -n "$EXCLUDE_MODEL" ]]; then
  echo "INCLUDE_MODEL and EXCLUDE_MODEL are mutually exclusive" >&2
  exit 2
fi
MODEL_FILTER_ARGS=()
if [[ -n "$INCLUDE_MODEL" ]]; then
  MODEL_FILTER_ARGS=(--include-model "$INCLUDE_MODEL")
elif [[ -n "$EXCLUDE_MODEL" ]]; then
  MODEL_FILTER_ARGS=(--exclude-model "$EXCLUDE_MODEL")
fi

run_one() {
  local eval_run="$1"
  local eval_dir="$RESULTS_ROOT/$eval_run/evaluation"
  mkdir -p "$eval_dir"
  {
    echo "[McEval] starting $eval_run with $WORKERS_PER_RUN worker(s)"
    env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/mceval_clean.py \
      --experiment-root "$RESULTS_ROOT/$eval_run" \
      "${MODEL_FILTER_ARGS[@]}"
    env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/evaluate_mceval.py \
      --experiment-root "$RESULTS_ROOT/$eval_run" \
      --sandbox docker \
      --docker-prefix sudo docker \
      --workers "$WORKERS_PER_RUN" \
      "${MODEL_FILTER_ARGS[@]}"
    env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/mceval_statistics.py \
      --experiment-root "$RESULTS_ROOT/$eval_run"
    echo "[McEval] completed $eval_run"
  } 2>&1 | sed -u "s/^/[McEval $eval_run] /" | tee "$eval_dir/run.log"
}

PIDS=()
for EVAL_RUN in $EVAL_RUNS; do
  run_one "$EVAL_RUN" &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  printf '%s\n' '[McEval] one or more runs failed; inspect their evaluation/run.log files' >&2
  exit 1
fi

printf '%s\n' '[McEval] requested runs completed'
