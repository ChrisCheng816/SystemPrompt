"""Locally evaluate HumanEval-X Python/Java generations from SystemPrompt.

This executes model-generated code.  It uses a new temporary directory and a
per-candidate timeout, but it is not a security sandbox.  Run it only on a
machine you are willing to use for local code evaluation.

Examples:
    python Tools/evaluate_humaneval_x.py --predictions path/to/predictions_cleaned.jsonl
    python Tools/evaluate_humaneval_x.py --predictions path/to/predictions.jsonl --workers 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .humaneval_x import load_humaneval_x_test
else:  # Supports `python Tools/evaluate_humaneval_x.py`.
    from humaneval_x import load_humaneval_x_test


PYTHON_IMPORTS = """import math
import re
import sys
import copy
import datetime
import itertools
import collections
import heapq
import statistics
import functools
import hashlib
import numpy
import numpy as np
import string
from typing import *
from collections import *
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds allowed per candidate.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent candidate evaluations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results (default: the predictions file directory).",
    )
    parser.add_argument(
        "--example-test",
        action="store_true",
        help="Use the public example tests instead of the benchmark's hidden test field.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            task_id = record.get("_id", record.get("task_id"))
            candidates = record.get("generate_results")
            if not isinstance(task_id, str) or "/" not in task_id:
                raise ValueError(f"Line {line_number} has no valid HumanEval-X task id.")
            if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
                raise ValueError(f"Line {line_number} must contain a list of string generate_results.")
            for completion_id, candidate in enumerate(candidates):
                records.append(
                    {"task_id": task_id, "completion_id": completion_id, "generation": candidate}
                )
    if not records:
        raise ValueError("The predictions file contains no generated candidates.")
    return records


def load_problems(task_ids: set[str]) -> dict[str, dict[str, Any]]:
    languages = {task_id.split("/", maxsplit=1)[0].lower() for task_id in task_ids}
    unsupported = languages - {"python", "java"}
    if unsupported:
        raise ValueError(f"Only Python and Java are supported; found {sorted(unsupported)}.")

    problems = {}
    for language in sorted(languages):
        for row in load_humaneval_x_test(language):
            problems[row["task_id"]] = dict(row)
    missing = sorted(task_ids - problems.keys())
    if missing:
        raise ValueError(f"Prediction task IDs are not in HumanEval-X: {missing[:3]}")
    return problems


def normalize_python_completion(code: str) -> str:
    """Keep a function body and restore indentation removed by generic cleaning."""
    lines = code.replace("\r\n", "\n").split("\n")
    first_content = next((line for line in lines if line.strip()), "")
    if first_content and not first_content.startswith((" ", "\t")):
        # SystemPrompt asks for a raw method body; generic cleaning strips the
        # indentation from its first line. Restore it consistently for every
        # non-empty line in the continuation.
        return "\n".join("    " + line if line.strip() else line for line in lines).rstrip() + "\n"

    completion = []
    for line in lines:
        if line.strip() and not line.startswith((" ", "\t")):
            break
        completion.append(line)
    return "\n".join(completion).rstrip() + "\n"


def build_test_program(problem: dict[str, Any], generation: str, example_test: bool) -> tuple[str, str]:
    language = problem["task_id"].split("/", maxsplit=1)[0].lower()
    test = problem["example_test"] if example_test and problem.get("example_test") else problem["test"]
    if language == "python":
        return "python", PYTHON_IMPORTS + "\n" + problem["prompt"] + normalize_python_completion(generation) + "\n" + test
    if language == "java":
        return "java", problem["prompt"] + generation.strip() + "\n" + test
    raise ValueError(f"Unsupported language: {language}")


def compact_error(output: bytes) -> str:
    text = output.decode("utf-8", errors="replace").strip()
    return text[-2000:] if text else "no diagnostic output"


def run_candidate(
    candidate: dict[str, Any], problem: dict[str, Any], timeout: float, example_test: bool
) -> dict[str, Any]:
    language, program = build_test_program(problem, candidate["generation"], example_test)
    result = {**candidate, "passed": False, "result": "failed"}
    try:
        with tempfile.TemporaryDirectory(prefix="humaneval_x_") as tmp_name:
            tmp_dir = Path(tmp_name)
            if language == "python":
                source = tmp_dir / "main.py"
                source.write_text(program, encoding="utf-8")
                command = [sys.executable, "-I", str(source)]
            else:
                source = tmp_dir / "Main.java"
                source.write_text(program, encoding="utf-8")
                compile_result = subprocess.run(
                    ["javac", source.name], cwd=tmp_dir, capture_output=True, timeout=timeout
                )
                if compile_result.returncode != 0:
                    result["result"] = "compile_error: " + compact_error(compile_result.stderr)
                    return result
                command = ["java", "-cp", str(tmp_dir), "Main"]

            execution = subprocess.run(command, cwd=tmp_dir, capture_output=True, timeout=timeout)
            if execution.returncode == 0:
                result.update({"passed": True, "result": "passed"})
            else:
                result["result"] = "runtime_error: " + compact_error(execution.stderr or execution.stdout)
    except subprocess.TimeoutExpired:
        result["result"] = f"timed_out_after_{timeout:g}s"
    except FileNotFoundError as error:
        executable = "javac/java" if language == "java" else sys.executable
        result["result"] = f"missing_runtime ({executable}): {error}"
    except Exception as error:
        result["result"] = f"evaluator_error: {type(error).__name__}: {error}"
    return result


def estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total < k:
        raise ValueError("k cannot exceed the number of generated candidates.")
    if total - correct < k:
        return 1.0
    return 1.0 - math.prod((total - correct - index) / (total - index) for index in range(k))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[result["task_id"]].append(result)
    totals = [len(items) for items in by_task.values()]
    correct = [sum(item["passed"] for item in items) for items in by_task.values()]
    metrics = {}
    for k in (1, 5, 10, 100):
        if totals and min(totals) >= k:
            metrics[f"pass@{k}"] = sum(
                estimate_pass_at_k(total, passed, k) for total, passed in zip(totals, correct, strict=True)
            ) / len(totals)
    return {
        "tasks": len(by_task),
        "candidates": len(results),
        "passed_candidates": sum(correct),
        "candidate_pass_rate": sum(correct) / len(results),
        "pass_at_k": metrics,
    }


def evaluate_predictions(
    predictions: Path,
    timeout: float = 10.0,
    workers: int = 1,
    output_dir: Path | None = None,
    example_test: bool = False,
) -> dict[str, Any]:
    """Evaluate one cleaned prediction file and return aggregate metrics."""
    if not predictions.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {predictions}")
    if timeout <= 0 or workers < 1:
        raise ValueError("timeout and workers must be positive.")

    candidates = read_predictions(predictions)
    problems = load_problems({candidate["task_id"] for candidate in candidates})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_candidate, candidate, problems[candidate["task_id"]], timeout, example_test)
            for candidate in candidates
        ]
        results = [future.result() for future in futures]

    output_dir = output_dir or predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "humaneval_x_results.jsonl"
    summary_path = output_dir / "humaneval_x_summary.json"
    with results_path.open("w", encoding="utf-8") as output_file:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Per-candidate results: {results_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_predictions(
        args.predictions,
        timeout=args.timeout,
        workers=args.workers,
        output_dir=args.output_dir,
        example_test=args.example_test,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
