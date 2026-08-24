#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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
  RUN_LOG="$EVAL_DIR/run.log"
  if [[ "$INCLUDE_MODEL" == "qwen3.8-27b-original" ]]; then
    RUN_LOG="$EVAL_DIR/run_qwen38.log"
  fi
  mkdir -p "$EVAL_DIR"
  {
    echo "[CoderEval] starting $EVAL_RUN with $WORKERS_PER_RUN worker(s)"
    qwen_model="qwen3.8-27b-original"
    qwen_root="$RESULTS_ROOT/$EVAL_RUN/predictions/Qwen3.8_27b"
    run_qwen_cleaner=false
    generic_clean_args=("${MODEL_FILTER_ARGS[@]}")

    if [[ "$INCLUDE_MODEL" == "$qwen_model" ]]; then
      run_qwen_cleaner=true
      generic_clean_args=()
    elif [[ -z "$INCLUDE_MODEL" && "$EXCLUDE_MODEL" != "$qwen_model" ]]; then
      # Preserve Qwen-specific outputs even during an all-model evaluation.
      generic_clean_args=(--exclude-model "$qwen_model")
      run_qwen_cleaner=true
    fi

    if [[ ${#generic_clean_args[@]} -gt 0 || "$INCLUDE_MODEL" != "$qwen_model" ]]; then
      env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/codereval_clean.py \
        --experiment-root "$RESULTS_ROOT/$EVAL_RUN" \
        "${generic_clean_args[@]}" \
        "${LANGUAGE_ARGS[@]}"
    fi

    if [[ "$run_qwen_cleaner" == true && -d "$qwen_root" ]]; then
      if [[ "$EVAL_RUN" != "pass@1_t0" && "$EVAL_RUN" != "pass@1_t1" && "$EVAL_RUN" != "pass@5_t1" ]]; then
        echo "[CoderEval] Qwen3.8 only supports pass@1_t0, pass@1_t1, or pass@5_t1, got $EVAL_RUN" >&2
        exit 2
      fi
      if [[ ! -x .venv-qwen38/bin/python ]]; then
        echo "[CoderEval] missing Qwen3.8 interpreter: .venv-qwen38/bin/python" >&2
        exit 2
      fi
      env PYTHONDONTWRITEBYTECODE=1 .venv-qwen38/bin/python Tools/qwen38_clean.py \
        --dataset codereval \
        --pass-dir "$EVAL_RUN" \
        --experiment-root "$qwen_root" \
        --write \
        --require-complete
    fi

    generic_eval_args=("${MODEL_FILTER_ARGS[@]}")
    run_generic_evaluator=true
    if [[ "$INCLUDE_MODEL" == "$qwen_model" ]]; then
      run_generic_evaluator=false
    elif [[ -z "$INCLUDE_MODEL" && "$EXCLUDE_MODEL" != "$qwen_model" ]]; then
      generic_eval_args=(--exclude-model "$qwen_model")
    fi

    if [[ "$run_generic_evaluator" == true ]]; then
      sudo -v
      env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/evaluate_codereval.py \
        --experiment-root "$RESULTS_ROOT/$EVAL_RUN" \
        --docker-prefix sudo docker \
        --workers "$WORKERS_PER_RUN" \
        "${generic_eval_args[@]}" \
        "${LANGUAGE_ARGS[@]}"
    fi

    if [[ "$run_qwen_cleaner" == true ]]; then
      if [[ ! -d "$qwen_root" ]]; then
        if [[ "$INCLUDE_MODEL" == "$qwen_model" ]]; then
          echo "[CoderEval] no Qwen3.8 predictions found for $EVAL_RUN" >&2
          exit 2
        fi
      else
        env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/evaluate_qwen38_codereval.py \
          --experiment-root "$RESULTS_ROOT/$EVAL_RUN" \
          --docker-prefix docker \
          --workers "$WORKERS_PER_RUN"
      fi
    fi

    if [[ "$run_generic_evaluator" == true ]]; then
      env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Tools/codereval_statistics.py \
        --experiment-root "$RESULTS_ROOT/$EVAL_RUN"
    fi
    echo "[CoderEval] completed $EVAL_RUN"
  } 2>&1 | sed -u "s/^/[CoderEval $EVAL_RUN] /" | tee "$RUN_LOG"
done

printf '%s\n' '[CoderEval] requested runs completed'
