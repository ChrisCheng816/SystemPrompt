"""Evaluate McEval predictions and write pass/fail plus failure details."""

from __future__ import annotations

import argparse
import ast
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
from mceval import load_mceval
from mceval_clean import clean_candidate, model_family_from_path
from mceval_sandbox import DockerSandbox


REPO_ROOT = Path(__file__).resolve().parents[1]
JAVA_CLASS_NAME = "Solution"
TIMEOUT_SECONDS = 10
ACTIVE_SANDBOX: DockerSandbox | None = None


def load_dataset_by_task(languages: set[str]) -> dict[str, dict]:
    metadata = {}
    for language in sorted(languages):
        dataset = load_mceval(language)["test"]
        for record in dataset:
            metadata[str(record["task_id"])] = dict(record)
    return metadata


def prompt_prefix(prompt: str, language: str) -> str:
    if language == "python":
        lines = []
        for line in prompt.splitlines():
            if line.lstrip().startswith(("def ", "async def ")):
                break
            if line.strip():
                lines.append(line)
        defaults = ["from typing import *", "import math", "import re", "import itertools", "import collections"]
        return "\n".join(defaults + lines) + "\n\n"

    class_index = prompt.find("class Solution")
    prefix = prompt[:class_index] if class_index >= 0 else ""
    return prefix.strip() + "\n\n"


def java_class_preamble(prompt: str, signature: str) -> str:
    """Recreate the task's imports and class fields before its target method."""
    class_index = prompt.find("class Solution")
    if class_index < 0:
        return prompt.strip() + f"\n\nclass {JAVA_CLASS_NAME} {{"
    class_open = prompt.find("{", class_index)
    if class_open < 0:
        return prompt[:class_index].strip() + f"\n\nclass {JAVA_CLASS_NAME} {{"
    signature_index = prompt.find(signature, class_open + 1)
    if signature_index < 0:
        return prompt[:class_index].strip() + f"\n\nclass {JAVA_CLASS_NAME} {{"
    imports = prompt[:class_index].strip()
    members = prompt[class_open + 1:signature_index].strip()
    return f"{imports}\n\nclass {JAVA_CLASS_NAME} {{\n{members}".rstrip()


def has_python_signature(code: str, entry_point: str) -> bool:
    return re.search(rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code) is not None


def has_java_signature(code: str, entry_point: str) -> bool:
    return re.search(rf"\b{re.escape(entry_point)}\s*\(", code) is not None


def indent_block(code: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(line if not line.strip() else prefix + line.lstrip() for line in code.splitlines())


def build_python_source(task: dict, code: str) -> tuple[str, bool]:
    entry_point = task["entry_point"]
    signature_match = has_python_signature(code, entry_point)
    if signature_match:
        method_source = code.strip()
    else:
        method_source = f"{task['signature'].rstrip()}\n{indent_block(code)}"
    return prompt_prefix(task["prompt"], "python") + method_source + "\n\n" + task["test"], signature_match


def strip_java_wrapper(code: str) -> str:
    code = re.sub(r"(?m)^\s*import\s+[^;]+;\s*", "", code).strip()
    match = re.search(r"class\s+\w+\s*\{(?P<body>.*)\}\s*$", code, flags=re.DOTALL)
    if match:
        return match.group("body").strip()
    return code


def build_java_source(task: dict, code: str) -> tuple[str, bool]:
    code = strip_java_wrapper(code)
    entry_point = task["entry_point"]
    signature_match = has_java_signature(code, entry_point)
    if signature_match:
        method_source = code.strip()
    else:
        method_source = f"{task['signature'].rstrip()} {code.strip()}"
    source = f"{java_class_preamble(task['prompt'], task['signature'])}\n{method_source}\n{task['test']}\n"
    return source, signature_match


def run_python_source(source: str) -> dict:
    if ACTIVE_SANDBOX is not None:
        return ACTIVE_SANDBOX.execute(source, "python")
    try:
        ast.parse(source)
    except SyntaxError as error:
        return {
            "is_pass": False,
            "stage": "syntax",
            "stdout": "",
            "stderr": f"{error.__class__.__name__}: {error}",
            "return_code": None,
            "timeout": False,
        }

    with tempfile.TemporaryDirectory(prefix="mceval-py-") as temp_dir:
        path = Path(temp_dir) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        return run_subprocess([sys.executable, str(path)], timeout=TIMEOUT_SECONDS, cwd=Path(temp_dir))


def run_java_source(source: str) -> dict:
    if ACTIVE_SANDBOX is not None:
        return ACTIVE_SANDBOX.execute(source, "java")
    with tempfile.TemporaryDirectory(prefix="mceval-java-") as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / f"{JAVA_CLASS_NAME}.java"
        source_path.write_text(source, encoding="utf-8")
        compile_result = run_subprocess(["javac", str(source_path)], timeout=TIMEOUT_SECONDS, cwd=temp_path)
        if not compile_result["is_pass"]:
            compile_result["stage"] = "compile"
            return compile_result
        run_result = run_subprocess(["java", "-ea", "-cp", str(temp_path), JAVA_CLASS_NAME], timeout=TIMEOUT_SECONDS, cwd=temp_path)
        run_result["stage"] = "pass" if run_result["is_pass"] else "runtime"
        return run_result


def run_subprocess(command: list[str], timeout: int, cwd: Path | None = None) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd)
    except FileNotFoundError as error:
        return {
            "is_pass": False,
            "stage": "infrastructure",
            "stdout": "",
            "stderr": f"missing executable: {error.filename}",
            "return_code": None,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "is_pass": False,
            "stage": "timeout",
            "stdout": error.stdout or "",
            "stderr": error.stderr or "timeout",
            "return_code": None,
            "timeout": True,
        }
    return {
        "is_pass": result.returncode == 0,
        "stage": "pass" if result.returncode == 0 else "runtime",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode,
        "timeout": False,
    }


def validate_runtime_tools(languages: set[str], sandbox: str) -> None:
    missing = []
    if sandbox == "none" and "java" in languages:
        for command in ("javac", "java"):
            if shutil.which(command) is None:
                missing.append(command)
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            f"Missing required executable(s) for McEval Java evaluation: {missing_text}. "
            "Run Java McEval on a node with a JDK, or evaluate Java inside a Java-enabled container."
        )


def evaluate_candidate(task: dict, code: str, language: str) -> tuple[dict, str, bool]:
    if not code.strip():
        result = {
            "is_pass": False,
            "stage": "empty",
            "stdout": "",
            "stderr": "empty generation",
            "return_code": None,
            "timeout": False,
        }
        return result, classify_failure("empty", "empty generation"), False

    if language == "python":
        source, signature_match = build_python_source(task, code)
        result = run_python_source(source)
    else:
        source, signature_match = build_java_source(task, code)
        result = run_java_source(source)
    if not signature_match and not result["is_pass"]:
        result["stage"] = "signature"
    failure_type = classify_failure(result["stage"], result.get("stderr", "") + result.get("stdout", ""), signature_match)
    return result, failure_type, signature_match


def evaluate_file(prediction_path: Path, task_by_id: dict[str, dict], limit: int | None = None) -> list[dict]:
    run_info = parse_run_name(prediction_path.parent.name)
    model_family = model_family_from_path(prediction_path)
    records = []
    for row_index, record in enumerate(read_jsonl(prediction_path)):
        if limit is not None and row_index >= limit:
            break
        task_id = str(record.get("_id", record.get("task_id")))
        task = task_by_id[task_id]
        language = task_id.split("/", 1)[0].lower()
        candidates = record.get("generate_results", [])
        for candidate_index, raw_code in enumerate(candidates):
            cleaned_code = clean_candidate(
                str(raw_code),
                language,
                task.get("entry_point"),
                task.get("signature"),
                model_family,
            )
            result, failure_type, signature_match = evaluate_candidate(task, cleaned_code, language)
            records.append(
                {
                    "dataset": "mceval",
                    "run_name": run_info.run_name,
                    "model": run_info.model,
                    "language": run_info.language,
                    "method": run_info.method,
                    "shot": run_info.shot,
                    "prompt_index": run_info.prompt_index,
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
                    "raw_code": raw_code,
                    "cleaned_code": cleaned_code,
                }
            )
    return records


def initialize_worker(sandbox: str, docker_prefix: list[str], sandbox_image: str) -> None:
    """Create one reusable sandbox for this worker process, if requested."""
    global ACTIVE_SANDBOX
    if sandbox == "docker":
        ACTIVE_SANDBOX = DockerSandbox(docker_prefix, sandbox_image, TIMEOUT_SECONDS)
        atexit.register(close_active_sandbox)


def close_active_sandbox() -> None:
    global ACTIVE_SANDBOX
    if ACTIVE_SANDBOX is not None:
        ACTIVE_SANDBOX.close()
        ACTIVE_SANDBOX = None


def evaluate_file_worker(
    prediction_path_text: str,
    task_by_id: dict[str, dict],
    limit: int | None,
) -> tuple[str, list[dict]]:
    prediction_path = Path(prediction_path_text)
    return prediction_path_text, evaluate_file(prediction_path, task_by_id, limit)


def evaluate_prediction_files(
    prediction_files: list[Path],
    task_by_id: dict[str, dict],
    limit: int | None,
    workers: int,
    sandbox: str,
    docker_prefix: list[str],
    sandbox_image: str,
) -> list[tuple[Path, list[dict]]]:
    """Evaluate files independently; only the parent writes shared output files."""
    if workers == 1:
        return [(path, evaluate_file(path, task_by_id, limit)) for path in prediction_files]

    completed: list[tuple[Path, list[dict]]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(sandbox, docker_prefix, sandbox_image),
    ) as executor:
        futures = {
            executor.submit(evaluate_file_worker, str(path), task_by_id, limit): path
            for path in prediction_files
        }
        for future in as_completed(futures):
            path_text, records = future.result()
            completed.append((Path(path_text), records))
    return completed


def run_canonical_preflight(task_by_id: dict[str, dict], languages: set[str], limit: int | None = None) -> list[dict]:
    records = []
    for task_id, task in sorted(task_by_id.items()):
        language = task_id.split("/", 1)[0].lower()
        if language not in languages:
            continue
        if limit is not None and len(records) >= limit:
            break
        code = f"{task['signature']} {task['canonical_solution']}" if language == "java" else f"{task['signature']}\n{task['canonical_solution']}"
        result, failure_type, signature_match = evaluate_candidate(task, code, language)
        records.append(
            {
                "dataset": "mceval",
                "language": language,
                "task_id": task_id,
                "is_pass": result["is_pass"],
                "stage": result["stage"],
                "failure_type": failure_type,
                "signature_match": signature_match,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=REPO_ROOT / "experiments_results_mceval" / "pass@1_t0")
    parser.add_argument("--language", choices=("java", "python"), action="append", dest="languages")
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks per prediction file for smoke tests.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip canonical-solution harness check.")
    parser.add_argument("--sandbox", choices=("none", "docker"), default="none", help="Run candidates locally or in the Docker sandbox.")
    parser.add_argument("--sandbox-image", default="mceval-sandbox:latest")
    parser.add_argument("--workers", type=int, default=1, help="Independent prediction-file workers (one sandbox per worker).")
    parser.add_argument("--exclude-model", action="append", default=[], help="Exact parsed model name to skip; may be repeated (e.g. gpt-20b).")
    parser.add_argument("--include-model", action="append", default=[], help="Exact parsed model name to evaluate; may be repeated (e.g. gpt-20b).")
    parser.add_argument("--docker-prefix", nargs="*", default=["docker"], help="Docker command prefix, e.g. --docker-prefix sudo docker")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.exclude_model and args.include_model:
        parser.error("--include-model and --exclude-model cannot be used together")

    prediction_files = list(iter_prediction_files(
        args.experiment_root,
        exclude_models=args.exclude_model,
        include_models=args.include_model,
    ))
    languages = set(args.languages or [parse_run_name(path.parent.name).language for path in prediction_files])
    validate_runtime_tools(languages, args.sandbox)
    task_by_id = load_dataset_by_task(languages)
    eval_dir = evaluation_root(args.experiment_root)
    eval_dir.mkdir(parents=True, exist_ok=True)

    global ACTIVE_SANDBOX
    ACTIVE_SANDBOX = DockerSandbox(args.docker_prefix, args.sandbox_image, TIMEOUT_SECONDS) if args.sandbox == "docker" else None
    try:
      if not args.skip_preflight:
          preflight_records = run_canonical_preflight(task_by_id, languages, args.limit)
          write_jsonl(eval_dir / "mceval_canonical_preflight.jsonl", preflight_records)
          failed = [record for record in preflight_records if not record["is_pass"]]
          if failed:
              raise SystemExit(f"Canonical preflight failed for {len(failed)} task(s); see {eval_dir / 'mceval_canonical_preflight.jsonl'}")

      selected_files = [
          path for path in prediction_files
          if parse_run_name(path.parent.name).language in languages
      ]
      print(f"[McEval] evaluating {len(selected_files)} prediction file(s) with {args.workers} worker(s)")

      # The parent sandbox is only needed for the preflight. Worker processes create
      # their own sandbox, so no container or output file is shared between workers.
      if args.workers > 1:
          close_active_sandbox()

      all_records = []
      completed_files = evaluate_prediction_files(
          selected_files, task_by_id, args.limit, args.workers,
          args.sandbox, args.docker_prefix, args.sandbox_image,
      )
      for prediction_path, run_records in sorted(completed_files, key=lambda item: str(item[0])):
          run_info = parse_run_name(prediction_path.parent.name)
          detail_path = eval_dir / f"failure_details_{run_info.language}" / f"{run_info.run_name}.jsonl"
          write_jsonl(detail_path, run_records)
          all_records.extend(run_records)
          print(f"[McEval] evaluated {len(run_records)} candidate(s): {prediction_path}")

      selected_run_names = [parse_run_name(path.parent.name).run_name for path in selected_files]
      merged_records = merge_selected_failure_records(
          eval_dir / "failure_modes_by_instance.jsonl", all_records, selected_run_names
      )
      write_jsonl(eval_dir / "failure_modes_by_instance.jsonl", merged_records)
      write_pass_fail_csvs(eval_dir, merged_records)
      write_summary_csv(eval_dir / "failure_modes_summary.csv", summarize_failure_records(merged_records))
    finally:
      if ACTIVE_SANDBOX is not None:
          ACTIVE_SANDBOX.close()
          ACTIVE_SANDBOX = None


if __name__ == "__main__":
    main()
