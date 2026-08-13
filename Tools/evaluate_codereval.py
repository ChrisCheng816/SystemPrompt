"""Run or collect CoderEval Docker evaluation into pass@*/evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import tempfile
import re
from pathlib import Path

try:
    from evaluation_common import (
        classify_failure,
        evaluation_root,
        iter_prediction_files,
        merge_selected_failure_records,
        parse_run_name,
        read_jsonl,
        summarize_failure_records,
        write_jsonl,
        write_pass_fail_csvs,
        write_summary_csv,
    )
except ModuleNotFoundError:  # Supports `python Tools/...` and `import Tools...`.
    from Tools.evaluation_common import (
        classify_failure,
        evaluation_root,
        iter_prediction_files,
        merge_selected_failure_records,
        parse_run_name,
        read_jsonl,
        summarize_failure_records,
        write_jsonl,
        write_pass_fail_csvs,
        write_summary_csv,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]


RUNNER_TEMPLATE = r'''
import os
import subprocess

root_dir = "./predictions"
results_root = "./predictions/Results"

jobs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    jsonl_path1 = os.path.join(dirpath, "predictions.jsonl")
    jsonl_path2 = os.path.join(dirpath, "predictions_cleaned.jsonl")
    if os.path.isfile(jsonl_path2):
        jsonl_path = jsonl_path2
    elif os.path.isfile(jsonl_path1):
        jsonl_path = jsonl_path1
    else:
        continue

    rel_path = os.path.relpath(dirpath, root_dir)
    jobs.append((rel_path, jsonl_path))

print(f"[CoderEval] queued {len(jobs)} prediction file(s)", flush=True)
failures = []
for index, (rel_path, jsonl_path) in enumerate(sorted(jobs), start=1):
    result_dir = os.path.join(results_root, rel_path)
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"{rel_path}.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    language = "python" if "python" in jsonl_path else "java"
    print(f"[CoderEval {index}/{len(jobs)}] {language}: {rel_path}", flush=True)
    with open(output_file, "w", encoding="utf-8") as output:
        if language == "python":
            result = subprocess.run(["python", "PythonExec.py", jsonl_path, "__PASS_AT__"], stdout=output, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(["python", "JavaExec.py", f"../{jsonl_path}", "__PASS_AT__"], stdout=output, stderr=subprocess.STDOUT, cwd="./java")
    print(f"[CoderEval {index}/{len(jobs)}] exit={result.returncode}: {rel_path}", flush=True)
    if result.returncode != 0:
        failures.append((rel_path, output_file, result.returncode))

if failures:
    print(f"[CoderEval] {len(failures)} executor failure(s); preserving logs below.", flush=True)
    for rel_path, output_file, return_code in failures:
        print(f"[CoderEval] executor failure exit={return_code}: {rel_path}", flush=True)
        with open(output_file, encoding="utf-8", errors="replace") as failure_log:
            print(failure_log.read(), end="", flush=True)
    raise SystemExit(1)
'''


def pass_at_from_experiment_root(experiment_root: Path) -> int:
    match = re.fullmatch(r"pass@(\d+)_t[01]", experiment_root.name)
    if match is None:
        raise ValueError(f"Cannot infer pass@k from experiment root: {experiment_root}")
    pass_at = int(match.group(1))
    if pass_at not in (1, 5):
        raise ValueError(f"Unsupported pass@k in experiment root: {experiment_root}")
    return pass_at


def build_runner(pass_at: int) -> str:
    if pass_at not in (1, 5):
        raise ValueError(f"Unsupported pass@k: {pass_at}")
    return RUNNER_TEMPLATE.replace("__PASS_AT__", str(pass_at))


def mirror_predictions(experiment_root: Path, workspace: Path, sources: list[Path] | None = None) -> None:
    workspace_predictions = workspace / "predictions"
    for source in sources or list(iter_prediction_files(experiment_root)):
        relative = source.relative_to(experiment_root / "predictions")
        target = workspace_predictions / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_docker_command(
    docker_args: list[str],
    image: str,
    workspace: Path,
    runner_path: Path,
) -> list[str]:
    """Build the official evaluator command using the image's default root user.

    PythonExec writes candidate files into root-owned repositories bundled inside the
    image, so overriding the container user causes a PermissionError.
    """
    return [
        *docker_args,
        "run",
        "--rm",
        "-v",
        f"{workspace / 'predictions'}:/home/travis/builds/predictions",
        "-v",
        f"{runner_path}:/home/travis/builds/run_all_jsonl.py:ro",
        "-w",
        "/home/travis/builds",
        image,
        "python",
        "run_all_jsonl.py",
    ]


def build_workspace_chown_command(
    docker_args: list[str],
    workspace: Path,
    uid: int,
    gid: int,
) -> list[str] | None:
    """Return a narrow ownership-restoration command when Docker uses sudo."""
    if docker_args and docker_args[0] == "sudo":
        return ["sudo", "chown", "-R", f"{uid}:{gid}", str(workspace)]
    return None


def restore_workspace_ownership(docker_args: list[str], workspace: Path) -> None:
    command = build_workspace_chown_command(docker_args, workspace, os.getuid(), os.getgid())
    if command is None:
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[CoderEval] could not restore workspace ownership: {workspace}", flush=True)


def run_docker_eval(
    experiment_root: Path,
    docker_args: list[str],
    image: str,
    prediction_files: list[Path] | None = None,
    worker_index: int | None = None,
) -> Path:
    temp_parent = evaluation_root(experiment_root)
    temp_parent.mkdir(parents=True, exist_ok=True)
    prefix = f"codereval-docker-w{worker_index}-" if worker_index is not None else "codereval-docker-"
    workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=temp_parent))
    mirror_predictions(experiment_root, workspace, prediction_files)
    runner_path = workspace / "run_all_jsonl.py"
    runner_path.write_text(build_runner(pass_at_from_experiment_root(experiment_root)), encoding="utf-8")
    command = build_docker_command(docker_args, image, workspace, runner_path)
    if worker_index is not None:
        print(f"[CoderEval] worker {worker_index}: evaluating {len(prediction_files or [])} prediction file(s)", flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        restore_workspace_ownership(docker_args, workspace)
        print(f"[CoderEval] worker {worker_index}: failed; workspace preserved: {workspace}", flush=True)
        raise
    restore_workspace_ownership(docker_args, workspace)
    return workspace


def partition_prediction_files(prediction_files: list[Path], workers: int) -> list[list[Path]]:
    """Distribute independent prediction files across long-lived Docker workers."""
    worker_count = min(workers, len(prediction_files))
    return [prediction_files[index::worker_count] for index in range(worker_count)] if worker_count else []


def run_docker_evals(
    experiment_root: Path,
    docker_args: list[str],
    image: str,
    workers: int,
    prediction_files: list[Path] | None = None,
) -> list[Path]:
    selected_files = prediction_files or list(iter_prediction_files(experiment_root))
    groups = partition_prediction_files(selected_files, workers)
    if not groups:
        return []
    if len(groups) == 1:
        return [run_docker_eval(experiment_root, docker_args, image, groups[0], worker_index=1)]

    workspaces: list[Path] = []
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {
            executor.submit(run_docker_eval, experiment_root, docker_args, image, group, index): index
            for index, group in enumerate(groups, start=1)
        }
        for future in as_completed(futures):
            worker_index = futures[future]
            workspace = future.result()
            workspaces.append(workspace)
            print(f"[CoderEval] worker {worker_index}: completed", flush=True)
    return workspaces


def collect_out_files(out_root: Path, experiment_root: Path) -> list[dict]:
    eval_dir = evaluation_root(experiment_root)
    all_records = []
    for out_path in sorted(out_root.rglob("*_out.jsonl")):
        run_dir = out_path.parent
        try:
            run_info = parse_run_name(run_dir.name)
        except ValueError:
            continue
        records = normalize_codereval_out(out_path, run_info)
        detail_path = eval_dir / f"failure_details_{run_info.language}" / f"{run_info.run_name}.jsonl"
        write_jsonl(detail_path, records)
        all_records.extend(records)
        print(f"Collected {len(records)} candidate(s): {out_path}")
    return all_records


def normalize_codereval_out(out_path: Path, run_info) -> list[dict]:
    normalized = []
    for record in read_jsonl(out_path):
        task_id = str(record.get("_id", ""))
        results = record.get("generate_results", [])
        if not results:
            generated_code = record.get("code", [""])
            if isinstance(generated_code, list):
                generated_code = generated_code[0] if generated_code else ""
            normalized.append(
                {
                    "dataset": "codereval",
                    "run_name": run_info.run_name,
                    "model": run_info.model,
                    "language": run_info.language,
                    "method": run_info.method,
                    "shot": run_info.shot,
                    "prompt_index": run_info.prompt_index,
                    "task_id": task_id,
                    "candidate_index": 0,
                    "is_pass": False,
                    "stage": "infrastructure",
                    "failure_type": "EVALUATOR_NO_RESULT",
                    "signature_match": None,
                    "return_code": None,
                    "timeout": False,
                    "stdout": "",
                    "stderr": "Official CodeREval evaluator returned no candidate verdict.",
                    "raw_code": generated_code,
                    "cleaned_code": generated_code,
                    "codereval_error": "",
                }
            )
            continue
        for candidate_index, result in enumerate(results):
            is_pass = bool(result.get("is_pass"))
            raw_message = str(result.get("errormessage", ""))
            detail = str(result.get("error_detail_message", ""))
            if is_pass:
                stage = "pass"
            elif "compile" in raw_message.lower():
                stage = "compile"
            elif "exec" in raw_message.lower():
                stage = "exec"
            elif result.get("return_code") not in (None, 0):
                stage = "runtime"
            else:
                stage = "unknown"
            failure_type = classify_failure(stage, raw_message + "\n" + detail, signature_match=True)
            normalized.append(
                {
                    "dataset": "codereval",
                    "run_name": run_info.run_name,
                    "model": run_info.model,
                    "language": run_info.language,
                    "method": run_info.method,
                    "shot": run_info.shot,
                    "prompt_index": run_info.prompt_index,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "is_pass": is_pass,
                    "stage": stage,
                    "failure_type": failure_type,
                    "signature_match": None,
                    "return_code": result.get("return_code"),
                    "timeout": "timeout" in (raw_message + detail).lower(),
                    "stdout": "",
                    "stderr": detail,
                    "raw_code": result.get("generate_code", ""),
                    "cleaned_code": result.get("generate_code", ""),
                    "codereval_error": raw_message,
                }
            )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=REPO_ROOT / "experiments_results_codereval" / "pass@1_t0")
    parser.add_argument("--image", default="codereval:latest")
    parser.add_argument("--docker-prefix", nargs="*", default=["docker"], help="Docker command prefix, e.g. --docker-prefix sudo docker")
    parser.add_argument("--workers", type=int, default=1, help="Host-side Docker workers; each uses an independent copied workspace.")
    parser.add_argument("--exclude-model", action="append", default=[], help="Exact parsed model name to skip; may be repeated (e.g. gpt-20b).")
    parser.add_argument("--include-model", action="append", default=[], help="Exact parsed model name to evaluate; may be repeated (e.g. gpt-20b).")
    parser.add_argument("--include-run", action="append", default=[], help="Exact parsed run name to evaluate; may be repeated for diagnostic smoke runs.")
    parser.add_argument("--language", choices=("java", "python"), action="append", dest="languages")
    parser.add_argument("--collect-only", action="store_true", help="Collect existing *_out.jsonl files below the experiment predictions directory without running Docker.")
    parser.add_argument("--collect-workspace", type=Path, action="append", default=[], help="Collect *_out.jsonl files from a retained Docker workspace; may be repeated.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep temporary Docker workspace under evaluation/.")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.exclude_model and args.include_model:
        parser.error("--include-model and --exclude-model cannot be used together")

    experiment_root = args.experiment_root.resolve()
    eval_dir = evaluation_root(experiment_root)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.collect_only and args.collect_workspace:
        parser.error("--collect-only and --collect-workspace cannot be used together")

    workspaces: list[Path] = []
    if args.collect_workspace:
        out_roots = [(workspace.resolve() / "predictions") for workspace in args.collect_workspace]
    elif args.collect_only:
        out_roots = [experiment_root / "predictions"]
    else:
        prediction_files = list(iter_prediction_files(
            experiment_root,
            exclude_models=args.exclude_model,
            include_models=args.include_model,
        ))
        if args.languages:
            selected_languages = set(args.languages)
            prediction_files = [
                path for path in prediction_files
                if parse_run_name(path.parent.name).language in selected_languages
            ]
        if args.include_run:
            requested_runs = set(args.include_run)
            selected_by_name = {parse_run_name(path.parent.name).run_name: path for path in prediction_files}
            unknown_runs = requested_runs - set(selected_by_name)
            if unknown_runs:
                parser.error("Unknown --include-run value(s): " + ", ".join(sorted(unknown_runs)))
            prediction_files = [selected_by_name[run_name] for run_name in sorted(requested_runs)]
        print(f"[CoderEval] evaluating {len(prediction_files)} prediction file(s)", flush=True)
        workspaces = run_docker_evals(experiment_root, args.docker_prefix, args.image, args.workers, prediction_files)
        out_roots = [workspace / "predictions" for workspace in workspaces]

    all_records = []
    for out_root in out_roots:
        all_records.extend(collect_out_files(out_root, experiment_root))
    selected_run_names = [parse_run_name(path.parent.name).run_name for path in prediction_files] if not (args.collect_only or args.collect_workspace) else []
    merged_records = merge_selected_failure_records(
        eval_dir / "failure_modes_by_instance.jsonl", all_records, selected_run_names
    ) if selected_run_names else all_records
    write_jsonl(eval_dir / "failure_modes_by_instance.jsonl", merged_records)
    write_pass_fail_csvs(eval_dir, merged_records)
    write_summary_csv(eval_dir / "failure_modes_summary.csv", summarize_failure_records(merged_records))

    if not args.keep_workspace:
        for workspace in workspaces:
            try:
                shutil.rmtree(workspace)
            except PermissionError as error:
                print(f"[CoderEval] kept root-owned workspace for manual cleanup: {workspace} ({error})", flush=True)


if __name__ == "__main__":
    main()
