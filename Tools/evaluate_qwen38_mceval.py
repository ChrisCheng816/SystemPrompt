"""Evaluate only cleaned Qwen3.8 McEval predictions without touching other models."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    import evaluate_mceval
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
    from Tools import evaluate_mceval


QWEN38_MODELS = frozenset({
    "qwen3.8-27b-original",
    "qwen3.8-27b-paraphrase-a",
    "qwen3.8-27b-paraphrase-b",
})
MODEL_DIRECTORY = "Qwen3.8_27b"


def cleaned_prediction_files(experiment_root: Path) -> list[Path]:
    root = experiment_root.resolve() / "predictions" / MODEL_DIRECTORY
    if not root.is_dir():
        raise FileNotFoundError(f"Missing exact Qwen3.8 prediction root: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("predictions_cleaned.jsonl")):
        info = parse_run_name(path.parent.name)
        if info.model not in QWEN38_MODELS:
            raise ValueError(f"Refusing non-Qwen3.8 prediction file: {path}")
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No cleaned Qwen3.8 prediction files under {root}")
    return files


def evaluate_cleaned_file(path_text: str, task_by_id: dict[str, dict], limit: int | None) -> tuple[str, list[dict]]:
    """Evaluate Qwen-cleaned code directly; do not invoke the generic cleaner."""
    path = Path(path_text)
    info = parse_run_name(path.parent.name)
    if info.model not in QWEN38_MODELS:
        raise ValueError(f"Refusing non-Qwen3.8 prediction file: {path}")

    records: list[dict] = []
    for row_index, prediction in enumerate(read_jsonl(path)):
        if limit is not None and row_index >= limit:
            break
        task_id = str(prediction.get("_id", prediction.get("task_id")))
        task = task_by_id[task_id]
        language = task_id.split("/", 1)[0].lower()
        candidates = prediction.get("generate_results", [])
        if not isinstance(candidates, list):
            raise ValueError(f"Invalid candidate list in {path}: {task_id}")
        for candidate_index, code in enumerate(candidates):
            result, failure_type, signature_match = evaluate_mceval.evaluate_candidate(task, str(code), language)
            records.append(
                {
                    "dataset": "mceval",
                    "run_name": info.run_name,
                    "model": info.model,
                    "language": info.language,
                    "method": info.method,
                    "shot": info.shot,
                    "prompt_index": info.prompt_index,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "is_pass": result["is_pass"],
                    "stage": result["stage"],
                    "failure_type": failure_type,
                    "signature_match": signature_match,
                    "return_code": result.get("return_code"),
                    "timeout": result.get("timeout", False),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "raw_code": code,
                    "cleaned_code": code,
                }
            )
    return path_text, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sandbox-image", default="mceval-sandbox:latest")
    parser.add_argument("--docker-prefix", nargs="*", default=["docker"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks per run for smoke tests.")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    experiment_root = args.experiment_root.resolve()
    sources = cleaned_prediction_files(experiment_root)
    run_infos = [parse_run_name(path.parent.name) for path in sources]
    languages = {info.language for info in run_infos}
    evaluate_mceval.validate_runtime_tools(languages, "docker")
    task_by_id = evaluate_mceval.load_dataset_by_task(languages)

    if not args.skip_preflight:
        evaluate_mceval.initialize_worker("docker", args.docker_prefix, args.sandbox_image)
        try:
            preflight = evaluate_mceval.run_canonical_preflight(task_by_id, languages, args.limit)
        finally:
            evaluate_mceval.close_active_sandbox()
        failed = [row for row in preflight if not row["is_pass"]]
        if failed:
            raise RuntimeError(f"Canonical McEval preflight failed for {len(failed)} task(s).")

    completed: list[tuple[Path, list[dict]]] = []
    worker_count = min(args.workers, len(sources))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=evaluate_mceval.initialize_worker,
        initargs=("docker", args.docker_prefix, args.sandbox_image),
    ) as executor:
        futures = {
            executor.submit(evaluate_cleaned_file, str(path), task_by_id, args.limit): path
            for path in sources
        }
        for future in as_completed(futures):
            path_text, records = future.result()
            completed.append((Path(path_text), records))
            print(f"[Qwen3.8 McEval] evaluated {len(records)} candidate(s): {path_text}", flush=True)

    evaluation_dir = evaluation_root(experiment_root)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for path, records in sorted(completed, key=lambda item: str(item[0])):
        info = parse_run_name(path.parent.name)
        write_jsonl(evaluation_dir / f"failure_details_{info.language}" / f"{info.run_name}.jsonl", records)
        all_records.extend(records)

    selected_runs = [info.run_name for info in run_infos]
    merged = merge_selected_failure_records(
        evaluation_dir / "failure_modes_by_instance.jsonl", all_records, selected_runs
    )
    write_jsonl(evaluation_dir / "failure_modes_by_instance.jsonl", merged)
    write_pass_fail_csvs(evaluation_dir, all_records)
    write_summary_csv(evaluation_dir / "failure_modes_summary.csv", summarize_failure_records(merged))
    print(
        f"[Qwen3.8 McEval] evaluated {len(sources)} run(s) with {worker_count} worker(s), "
        f"collected {len(all_records)} verdicts"
    )


if __name__ == "__main__":
    main()
