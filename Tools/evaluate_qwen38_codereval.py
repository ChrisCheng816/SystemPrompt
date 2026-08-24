"""Evaluate only cleaned Qwen3.8 CodeEval predictions in an isolated workspace.

This script is deliberately separate from evaluate_codereval.py. It accepts
only the exact Qwen3.8 prediction directory and copies only cleaned
prediction files into a disposable Docker workspace. Derived evaluation files
follow the existing pass@*/evaluation layout and merge only Qwen records.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from evaluation_common import (
        evaluation_root,
        merge_selected_failure_records,
        parse_run_name,
        summarize_failure_records,
        write_jsonl,
        write_pass_fail_csvs,
        write_summary_csv,
    )
    import evaluate_codereval
except ModuleNotFoundError:
    from Tools.evaluation_common import (
        evaluation_root,
        merge_selected_failure_records,
        parse_run_name,
        summarize_failure_records,
        write_jsonl,
        write_pass_fail_csvs,
        write_summary_csv,
    )
    from Tools import evaluate_codereval


MODEL_NAME = "qwen3.8-27b-original"
MODEL_DIRECTORY = "Qwen3.8_27b"


def qwen_prediction_root(experiment_root: Path) -> Path:
    root = experiment_root.resolve() / "predictions" / MODEL_DIRECTORY
    if root.name != MODEL_DIRECTORY or not root.is_dir():
        raise FileNotFoundError(f"Missing exact Qwen3.8 prediction root: {root}")
    return root


def cleaned_prediction_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("predictions_cleaned.jsonl")):
        run_info = parse_run_name(path.parent.name)
        if run_info.model != MODEL_NAME:
            raise ValueError(f"Refusing non-Qwen3.8 prediction file: {path}")
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No cleaned Qwen3.8 prediction files under {root}")
    return files


def mirror_cleaned_predictions(root: Path, workspace: Path, sources: list[Path]) -> None:
    target_root = workspace / "predictions"
    for source in sources:
        target = target_root / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def collect_records(output_root: Path, experiment_root: Path) -> list[dict]:
    records = evaluate_codereval.collect_out_files(output_root, experiment_root)
    if any(record.get("model") != MODEL_NAME for record in records):
        raise ValueError("Refusing non-Qwen3.8 evaluator output.")
    return records



def run_qwen_docker_eval(
    experiment_root: Path,
    qwen_root: Path,
    docker_args: list[str],
    image: str,
    prediction_files: list[Path],
    worker_index: int,
) -> Path:
    """Evaluate one Qwen3.8 file partition in an isolated Docker workspace."""
    evaluation_dir = evaluation_root(experiment_root)
    workspace = Path(
        tempfile.mkdtemp(prefix=f"codereval-qwen38-w{worker_index}-", dir=evaluation_dir)
    )
    mirror_cleaned_predictions(qwen_root, workspace, prediction_files)
    runner_path = workspace / "run_all_jsonl.py"
    runner_path.write_text(
        evaluate_codereval.build_runner(evaluate_codereval.pass_at_from_experiment_root(experiment_root)),
        encoding="utf-8",
    )
    command = evaluate_codereval.build_docker_command(docker_args, image, workspace, runner_path)
    print(
        f"[Qwen3.8 CodeEval] worker {worker_index}: evaluating {len(prediction_files)} run(s)",
        flush=True,
    )
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        evaluate_codereval.restore_workspace_ownership(docker_args, image, workspace)
        print(
            f"[Qwen3.8 CodeEval] worker {worker_index}: failed; workspace preserved: {workspace}",
            flush=True,
        )
        raise
    evaluate_codereval.restore_workspace_ownership(docker_args, image, workspace)
    return workspace


def partition_qwen_prediction_files(files: list[Path], workers: int) -> list[list[Path]]:
    worker_count = min(workers, len(files))
    return [files[index::worker_count] for index in range(worker_count)] if worker_count else []


def run_qwen_docker_evals(
    experiment_root: Path,
    qwen_root: Path,
    docker_args: list[str],
    image: str,
    workers: int,
    prediction_files: list[Path],
) -> list[Path]:
    """Run independent Qwen3.8 prediction partitions concurrently."""
    groups = partition_qwen_prediction_files(prediction_files, workers)
    if not groups:
        return []
    if len(groups) == 1:
        return [
            run_qwen_docker_eval(
                experiment_root, qwen_root, docker_args, image, groups[0], worker_index=1
            )
        ]

    workspaces: list[Path] = []
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {
            executor.submit(
                run_qwen_docker_eval,
                experiment_root,
                qwen_root,
                docker_args,
                image,
                group,
                worker_index,
            ): worker_index
            for worker_index, group in enumerate(groups, start=1)
        }
        for future in as_completed(futures):
            worker_index = futures[future]
            workspaces.append(future.result())
            print(f"[Qwen3.8 CodeEval] worker {worker_index}: completed", flush=True)
    return workspaces

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--image", default="codereval:latest")
    parser.add_argument("--docker-prefix", nargs="*", default=["docker"])
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent Qwen3.8 Docker workspaces (capped at the number of runs).",
    )
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    experiment_root = args.experiment_root.resolve()
    qwen_root = qwen_prediction_root(experiment_root)
    sources = cleaned_prediction_files(qwen_root)
    evaluation_dir = evaluation_root(experiment_root)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    workspaces: list[Path] = []
    try:
        workspaces = run_qwen_docker_evals(
            experiment_root,
            qwen_root,
            args.docker_prefix,
            args.image,
            args.workers,
            sources,
        )
        records: list[dict] = []
        for workspace in workspaces:
            records.extend(collect_records(workspace / "predictions", experiment_root))
        if not records:
            raise RuntimeError("CodeEval returned no Qwen3.8 verdict records.")
        selected_runs = [parse_run_name(source.parent.name).run_name for source in sources]
        merged = merge_selected_failure_records(
            evaluation_dir / "failure_modes_by_instance.jsonl",
            records,
            selected_runs,
        )
        write_jsonl(evaluation_dir / "failure_modes_by_instance.jsonl", merged)
        # Per-model CSVs are independent files; update only Qwen's matrices.
        write_pass_fail_csvs(evaluation_dir, records)
        write_summary_csv(
            evaluation_dir / "failure_modes_summary.csv",
            summarize_failure_records(merged),
        )
        print(
            f"[Qwen3.8 CodeEval] evaluated {len(sources)} run(s) with "
            f"{min(args.workers, len(sources))} worker(s), collected {len(records)} verdicts"
        )
    except BaseException:
        print("[Qwen3.8 CodeEval] failed; completed worker workspaces are retained for diagnosis.", flush=True)
        raise
    finally:
        if not args.keep_workspace:
            for workspace in workspaces:
                try:
                    shutil.rmtree(workspace)
                except OSError:
                    print(f"[Qwen3.8 CodeEval] workspace retained: {workspace}", flush=True)


if __name__ == "__main__":
    main()
