"""Rebuild CodeREval pass/fail matrices and McNemar reports from Docker verdicts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from evaluation_common import (
        LANGUAGES,
        PROMPT_COLUMNS,
        evaluation_root,
        read_jsonl,
        write_pass_fail_csvs,
    )
except ModuleNotFoundError:  # Supports `python Tools/...` and `import Tools...`.
    from Tools.evaluation_common import (
        LANGUAGES,
        PROMPT_COLUMNS,
        evaluation_root,
        read_jsonl,
        write_pass_fail_csvs,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_LENGTHS = (40, 185, 294, 480, 638)


def reset_csv_results(evaluation_dir: Path) -> None:
    """Remove only derived pass/fail CSVs before rebuilding them."""
    for language in LANGUAGES:
        csv_dir = evaluation_dir / f"csv_results_{language}"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in csv_dir.glob("*.csv"):
            csv_path.unlink()


def _read_matrix(path: Path) -> tuple[list[str], list[dict[str, int]]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        columns = reader.fieldnames or []
        return columns, [{column: int(row.get(column, 0) or 0) for column in columns} for row in reader]


def _mcnemar_rows(csv_dir: Path, language: str) -> list[dict]:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    rows: list[dict] = []
    global_id = 0
    for csv_path in sorted(csv_dir.glob("*.csv")):
        columns, matrix = _read_matrix(csv_path)
        if "BASE" not in columns:
            continue
        model, task, method, shot = csv_path.stem.rsplit("_", 3)
        comparisons = [column for column in PROMPT_COLUMNS if column in columns and column != "BASE"]
        raw_pvalues: list[float] = []
        row_indexes: list[int] = []
        for column in comparisons:
            both_pass = sum(row["BASE"] == 1 and row[column] == 1 for row in matrix)
            base_only = sum(row["BASE"] == 1 and row[column] == 0 for row in matrix)
            candidate_only = sum(row["BASE"] == 0 and row[column] == 1 for row in matrix)
            both_fail = sum(row["BASE"] == 0 and row[column] == 0 for row in matrix)
            result = mcnemar([[both_pass, base_only], [candidate_only, both_fail]], exact=True, correction=False)
            # Direct paired odds ratio: prompt-only wins / Base-only wins.
            # OR > 1 therefore favors the prompt variant.
            global_id += 1
            rows.append(
                {
                    "ID": global_id,
                    "Model": model,
                    "Task": task,
                    "Method": method,
                    "Shot": shot,
                    "Compare": f"BASE vs {column}",
                    "OR": round((candidate_only + 1) / (base_only + 1), 3),
                    "p_value": round(float(result.pvalue), 5),
                }
            )
            raw_pvalues.append(float(result.pvalue))
            row_indexes.append(len(rows) - 1)
        if raw_pvalues:
            holm_values = multipletests(raw_pvalues, alpha=0.05, method="holm")[1]
            fdr_values = multipletests(raw_pvalues, alpha=0.05, method="fdr_bh")[1]
            for index, row_index in enumerate(row_indexes):
                rows[row_index]["p_value_holm"] = round(float(holm_values[index]), 5)
                rows[row_index]["p_value_fdr_bh"] = round(float(fdr_values[index]), 5)
    return rows


def _write_mcnemar_report(path: Path, rows: list[dict]) -> None:
    fields = ["ID", "Model", "Task", "Method", "Shot", "Compare", "OR", "p_value", "p_value_holm", "p_value_fdr_bh"]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_legacy_results_summary(evaluation_dir: Path, experiment_root: Path, records: list[dict]) -> None:
    """Keep Results/output_detail.csv current without relying on legacy .txt logs."""
    result_dir = evaluation_dir / "Results"
    result_dir.mkdir(parents=True, exist_ok=True)
    per_prompt: dict[tuple[str, str, str, str, int], dict[str, int]] = defaultdict(dict)
    for record in records:
        key = (
            str(record["model"]),
            str(record["language"]),
            str(record["method"]),
            str(record["shot"]),
            int(record["prompt_index"]),
        )
        task_id = str(record["task_id"])
        per_prompt[key][task_id] = max(per_prompt[key].get(task_id, 0), int(bool(record["is_pass"])))

    metric = "pass@5" if experiment_root.name.startswith("pass@5") else "pass@1"
    grouped: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for (*run_key, prompt_index), task_results in per_prompt.items():
        grouped[tuple(run_key)][prompt_index] = 100 * sum(task_results.values()) / len(task_results) if task_results else 0.0

    detail_path = result_dir / "output_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["model", "task", "method", "shot", "pass@k", *[f"pass@k_{length}" for length in PROMPT_LENGTHS]])
        for run_key, by_prompt in sorted(grouped.items()):
            writer.writerow([*run_key, metric, *[f"{by_prompt[index]:.2f}" if index in by_prompt else "" for index in range(len(PROMPT_LENGTHS))]])

    labels_path = result_dir / "output.json"
    with labels_path.open("w", encoding="utf-8") as output_file:
        for prompt_index, prompt_length in enumerate(PROMPT_LENGTHS):
            json.dump({"system_prompt": str(prompt_index), "prompt_len": prompt_length}, output_file, ensure_ascii=False)
            output_file.write("\n")


def rebuild_statistics(
    experiment_root: Path,
    exclude_models: set[str] | None = None,
    dataset: str = "codereval",
) -> tuple[int, dict[str, int]]:
    evaluation_dir = evaluation_root(experiment_root)
    records_path = evaluation_dir / "failure_modes_by_instance.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"Docker verdict file does not exist: {records_path}")
    excluded = exclude_models or set()
    records = [
        record
        for record in read_jsonl(records_path)
        if record.get("dataset") == dataset and record.get("model") not in excluded
    ]
    reset_csv_results(evaluation_dir)
    write_pass_fail_csvs(evaluation_dir, records)
    _write_legacy_results_summary(evaluation_dir, experiment_root, records)
    report_counts = {}
    for language, filename in (("java", "mcnemar_results_holm.csv"), ("python", "mcnemar_results_holm_p.csv")):
        rows = _mcnemar_rows(evaluation_dir / f"csv_results_{language}", language)
        _write_mcnemar_report(evaluation_dir / filename, rows)
        report_counts[language] = len(rows)
    return len(records), report_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=REPO_ROOT / "experiments_results_codereval" / "pass@1_t0")
    parser.add_argument("--exclude-model", action="append", default=[], help="Exact parsed model name to omit; may be repeated (e.g. gpt-20b).")
    parser.add_argument("--dataset", choices=("codereval", "mceval"), default="codereval")
    args = parser.parse_args()
    record_count, report_counts = rebuild_statistics(args.experiment_root.resolve(), set(args.exclude_model), args.dataset)
    label = "CoderEval" if args.dataset == "codereval" else "McEval"
    print(f"[{label} statistics] rebuilt matrices from {record_count} verdict(s)")
    print(f"[{label} statistics] McNemar rows: java={report_counts['java']}, python={report_counts['python']}")


if __name__ == "__main__":
    main()
