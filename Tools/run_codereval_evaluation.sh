#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

sudo -v

WORKERS_PER_RUN="${WORKERS_PER_RUN:-3}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments_results_codereval}"
INCLUDE_MODEL="${INCLUDE_MODEL:-}"
EXCLUDE_MODEL="${EXCLUDE_MODEL:-}"
EVAL_LANGUAGES="${EVAL_LANGUAGES:-}"
if [[ -n "$INCLUDE_MODEL" && -n "$EXCLUDE_MODEL" ]]; then
  echo "INCLUDE_MODEL and EXCLUDE_MODEL are mutually exclusive" >&2
  exit 2
fi
MODEL_FILTER_ARGS=()
LANGUAGE_ARGS=()
if [[ -n "$INCLUDE_MODEL" ]]; then
  MODEL_FILTER_ARGS=(--include-model "$INCLUDE_MODEL")
elif [[ -n "$EXCLUDE_MODEL" ]]; then
  MODEL_FILTER_ARGS=(--exclude-model "$EXCLUDE_MODEL")
fi
for EVAL_LANGUAGE in $EVAL_LANGUAGES; do
  if [[ "$EVAL_LANGUAGE" != "java" && "$EVAL_LANGUAGE" != "python" ]]; then
    echo "EVAL_LANGUAGES accepts only java and/or python" >&2
    exit 2
  fi
  LANGUAGE_ARGS+=(--language "$EVAL_LANGUAGE")
done
EVAL_RUNS="${EVAL_RUNS:-pass@1_t0 pass@1_t1 pass@5_t1}"

for EVAL_RUN in $EVAL_RUNS; do
  EVAL_DIR="$RESULTS_ROOT/$EVAL_RUN/evaluation"
  mkdir -p "$EVAL_DIR"
  {
    echo "[CoderEval] starting $EVAL_RUN with $WORKERS_PER_RUN worker(s)"
    env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/codereval_clean.py \
      --experiment-root "$RESULTS_ROOT/$EVAL_RUN" \
      "${MODEL_FILTER_ARGS[@]}" \
      "${LANGUAGE_ARGS[@]}"
    env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/evaluate_codereval.py \
      --experiment-root "$RESULTS_ROOT/$EVAL_RUN" \
      --docker-prefix sudo docker \
      --workers "$WORKERS_PER_RUN" \
      "${MODEL_FILTER_ARGS[@]}" \
      "${LANGUAGE_ARGS[@]}"
    env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/codereval_statistics.py \
      --experiment-root "$RESULTS_ROOT/$EVAL_RUN"
    echo "[CoderEval] completed $EVAL_RUN"
  } 2>&1 | sed -u "s/^/[CoderEval $EVAL_RUN] /" | tee "$EVAL_DIR/run.log"
done

printf '%s\n' '[CoderEval] requested runs completed'
