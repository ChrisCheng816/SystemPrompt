"""Materialize candidates skipped by CodeREval's official Python executor."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from evaluation_common import (
        evaluation_root,
        merge_selected_failure_records,
        parse_run_name,
        read_jsonl,
        summarize_failure_records,
        write_jsonl,
        write_pass_fail_csvs,
        write_summary_csv,
    )
except ModuleNotFoundError:
    from Tools.evaluation_common import (
        evaluation_root,
        merge_selected_failure_records,
        parse_run_name,
        read_jsonl,
        summarize_failure_records,
        write_jsonl,
        write_pass_fail_csvs,
        write_summary_csv,
    )


def _load_by_id(path: Path) -> dict[str, dict]:
    return {str(row["_id"]): row for row in read_jsonl(path)}


def _placeholder(run_info, task_id: str, candidate_index: int, raw_code: str, cleaned_code: str) -> dict:
    return {
        "dataset": "codereval",
        "run_name": run_info.run_name,
        "model": run_info.model,
        "language": run_info.language,
        "method": run_info.method,
        "shot": run_info.shot,
        "prompt_index": run_info.prompt_index,
        "task_id": task_id,
        "candidate_index": candidate_index,
        "is_pass": False,
        "stage": "infrastructure",
        "failure_type": "EVALUATOR_NO_RESULT",
        "signature_match": None,
        "return_code": None,
        "timeout": None,
        "stdout": "",
        "stderr": (
            "Official CodeREval PythonExec.py caught an exception while evaluating "
            "this candidate and continued without writing a verdict."
        ),
        "raw_code": raw_code,
        "cleaned_code": cleaned_code,
        "codereval_error": "OFFICIAL_EXECUTOR_EXCEPTION_NO_VERDICT",
    }


def backfill_run_records(run_dir: Path, detail_path: Path) -> tuple[list[dict], int]:
    """Return records in source-candidate order, adding explicit entries for omitted verdicts."""
    run_info = parse_run_name(run_dir.name)
    cleaned_by_id = _load_by_id(run_dir / "predictions_cleaned.jsonl")
    raw_path = run_dir / "predictions.jsonl"
    raw_by_id = _load_by_id(raw_path) if raw_path.is_file() else cleaned_by_id
    observed_by_id: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(detail_path):
        observed_by_id[str(record["task_id"])].append(record)

    reconstructed: list[dict] = []
    added = 0
    for task_id, cleaned_row in cleaned_by_id.items():
        raw_candidates = raw_by_id.get(task_id, cleaned_row).get("generate_results", [])
        observed = observed_by_id[task_id]
        consumed = [False] * len(observed)
        for candidate_index, cleaned_code in enumerate(cleaned_row.get("generate_results", [])):
            matching_index = next(
                (
                    index for index, record in enumerate(observed)
                    if not consumed[index] and record.get("cleaned_code", "") == cleaned_code
                ),
                None,
            )
            raw_code = raw_candidates[candidate_index] if candidate_index < len(raw_candidates) else cleaned_code
            if matching_index is None:
                reconstructed.append(
                    _placeholder(run_info, task_id, candidate_index, raw_code, cleaned_code)
                )
                added += 1
                continue
            consumed[matching_index] = True
            record = dict(observed[matching_index])
            record["candidate_index"] = candidate_index
            record["raw_code"] = raw_code
            record["cleaned_code"] = cleaned_code
            reconstructed.append(record)
    return reconstructed, added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--language", choices=("python", "java"), action="append", dest="languages")
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    languages = set(args.languages or ("python", "java"))
    eval_dir = evaluation_root(experiment_root)
    updated_records: list[dict] = []
    selected_runs: list[str] = []
    total_added = 0

    for language in sorted(languages):
        detail_dir = eval_dir / f"failure_details_{language}"
        for cleaned_path in sorted(experiment_root.glob(f"predictions/*/*_{language}_*/predictions_cleaned.jsonl")):
            detail_path = detail_dir / f"{cleaned_path.parent.name}.jsonl"
            if not detail_path.is_file():
                continue
            records, added = backfill_run_records(cleaned_path.parent, detail_path)
            write_jsonl(detail_path, records)
            updated_records.extend(records)
            selected_runs.append(cleaned_path.parent.name)
            total_added += added

    if not selected_runs:
        raise SystemExit("No evaluated cleaned prediction files found for the selected language(s).")

    merged = merge_selected_failure_records(
        eval_dir / "failure_modes_by_instance.jsonl", updated_records, selected_runs
    )
    write_jsonl(eval_dir / "failure_modes_by_instance.jsonl", merged)
    write_pass_fail_csvs(eval_dir, merged)
    write_summary_csv(eval_dir / "failure_modes_summary.csv", summarize_failure_records(merged))
    print(f"Materialized {total_added} EVALUATOR_NO_RESULT record(s) across {len(selected_runs)} run(s).")


if __name__ == "__main__":
    main()
