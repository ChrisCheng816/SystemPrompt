"""Shared helpers for benchmark evaluation outputs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROMPT_COLUMNS = ("BASE", "STRUCT", "ROBUST", "REASONING", "EDGE")
METHODS = ("zero", "naive", "retrieval")
LANGUAGES = ("java", "python")


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    model: str
    language: str
    method: str
    shot: str
    prompt_index: int

    @property
    def csv_stem(self) -> str:
        return f"{self.model}_{self.language}_{self.method}_{self.shot}"

    @property
    def prompt_column(self) -> str:
        return PROMPT_COLUMNS[self.prompt_index]


def parse_run_name(run_name: str) -> RunInfo:
    """Parse names like qwen2.5-7b_python_zero_0-shot_3 from the right."""
    parts = run_name.rsplit("_", 4)
    if len(parts) != 5:
        raise ValueError(f"Cannot parse run directory name: {run_name}")
    model, language, method, shot, prompt_index_text = parts
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language in run name {run_name}: {language}")
    if method not in METHODS:
        raise ValueError(f"Unsupported method in run name {run_name}: {method}")
    try:
        prompt_index = int(prompt_index_text)
    except ValueError as error:
        raise ValueError(f"Invalid prompt index in run name {run_name}") from error
    if prompt_index < 0 or prompt_index >= len(PROMPT_COLUMNS):
        raise ValueError(f"Prompt index out of range in run name {run_name}: {prompt_index}")
    return RunInfo(run_name, model, language, method, shot, prompt_index)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def predictions_root(experiment_root: Path) -> Path:
    return experiment_root if experiment_root.name == "predictions" else experiment_root / "predictions"


def evaluation_root(experiment_root: Path) -> Path:
    if experiment_root.name == "predictions":
        return experiment_root.parent / "evaluation"
    return experiment_root / "evaluation"


def iter_prediction_files(
    experiment_root: Path,
    prefer_cleaned: bool = True,
    exclude_models: Iterable[str] = (),
    include_models: Iterable[str] = (),
) -> Iterable[Path]:
    root = predictions_root(experiment_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {root}")
    excluded = set(exclude_models)
    included = set(include_models)
    if excluded and included:
        raise ValueError("include_models and exclude_models are mutually exclusive")
    for run_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
        cleaned = run_dir / "predictions_cleaned.jsonl"
        raw = run_dir / "predictions.jsonl"
        if not (cleaned.is_file() or raw.is_file()):
            continue
        try:
            run_info = parse_run_name(run_dir.name)
        except ValueError as error:
            print(f"Skipped unsupported prediction run: {run_dir} ({error})")
            continue
        if run_info.model in excluded:
            continue
        if included and run_info.model not in included:
            continue
        if prefer_cleaned and cleaned.is_file():
            yield cleaned
        elif raw.is_file():
            yield raw


def failure_record_run_name(record: dict) -> str | None:
    """Return the prediction directory name represented by an evaluation record."""
    required = ("model", "language", "method", "shot", "prompt_index")
    if not all(key in record for key in required):
        return None
    return "{model}_{language}_{method}_{shot}_{prompt_index}".format(**record)


def merge_selected_failure_records(
    output_path: Path,
    new_records: Iterable[dict],
    selected_run_names: Iterable[str],
) -> list[dict]:
    """Replace prior verdicts only for selected runs, retaining all other runs."""
    selected = set(selected_run_names)
    existing = read_jsonl(output_path) if output_path.is_file() else []
    retained = [record for record in existing if failure_record_run_name(record) not in selected]
    return retained + list(new_records)


def classify_failure(stage: str, message: str = "", signature_match: bool = True) -> str:
    text = f"{stage}\n{message}".lower()
    if stage == "pass":
        return "PASS"
    if stage == "empty":
        return "EMPTY_GENERATION"
    if not signature_match:
        return "SIGNATURE_CONTRACT"
    if "timeout" in text or "timed out" in text:
        return "TIMEOUT"
    if stage in {"syntax", "compile"}:
        return "SYNTAX_COMPILE"
    if "nameerror" in text or "cannot find symbol" in text or "undefined" in text:
        return "PROJECT_CONTEXT"
    if "assertionerror" in text or "assert" in text or stage == "assertion":
        return "LOGIC_ASSERTION"
    if "exception" in text or "traceback" in text or stage == "runtime":
        return "RUNTIME_EXCEPTION"
    if stage == "exec":
        return "EXECUTION_FAILURE"
    return "UNKNOWN_FAILURE"


def summarize_failure_records(records: Iterable[dict]) -> list[dict]:
    counts = Counter()
    totals = Counter()
    for record in records:
        key = (
            record.get("dataset", ""),
            record.get("language", ""),
            record.get("model", ""),
            record.get("method", ""),
            record.get("shot", ""),
            str(record.get("prompt_index", "")),
            record.get("failure_type", ""),
        )
        counts[key] += 1
        totals[key[:-1]] += 1

    rows = []
    for key, count in sorted(counts.items()):
        dataset, language, model, method, shot, prompt_index, failure_type = key
        total = totals[key[:-1]]
        rows.append(
            {
                "dataset": dataset,
                "language": language,
                "model": model,
                "method": method,
                "shot": shot,
                "prompt_index": prompt_index,
                "failure_type": failure_type,
                "count": count,
                "rate": count / total if total else 0.0,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "language",
        "model",
        "method",
        "shot",
        "prompt_index",
        "failure_type",
        "count",
        "rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pass_fail_csvs(evaluation_dir: Path, records: Iterable[dict]) -> None:
    grouped: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        run = RunInfo(
            record["run_name"],
            record["model"],
            record["language"],
            record["method"],
            record["shot"],
            int(record["prompt_index"]),
        )
        task_id = str(record["task_id"])
        passed = int(bool(record["is_pass"]))
        current = grouped[(run.language, run.csv_stem)][task_id].get(run.prompt_column)
        grouped[(run.language, run.csv_stem)][task_id][run.prompt_column] = max(current or 0, passed)

    for (language, csv_stem), task_rows in grouped.items():
        csv_dir = evaluation_dir / f"csv_results_{language}"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{csv_stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(PROMPT_COLUMNS))
            writer.writeheader()
            for task_id in sorted(task_rows, key=_natural_key):
                row = {column: task_rows[task_id].get(column, 0) for column in PROMPT_COLUMNS}
                writer.writerow(row)


def _natural_key(text: str) -> list[object]:
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", text)]
