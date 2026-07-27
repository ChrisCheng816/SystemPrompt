"""Clean CodeEval predictions in one experiments_results/pass@*_t* run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments_results" / "pass@1_t0"


def process_code(text: str) -> str:
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1].strip()

    matches = re.findall(r"```(?:java|python)\n(.*?)```", text, re.DOTALL)
    if matches:
        return "".join(matches).strip()

    if text.startswith("```"):
        text = text[3:]
    fence_index = text.find("```")
    if fence_index != -1:
        text = text[:fence_index]
    return text.strip()


def predictions_root(experiment_root: Path) -> Path:
    return experiment_root if experiment_root.name == "predictions" else experiment_root / "predictions"


def clean_predictions(root: Path) -> tuple[int, int]:
    files_processed = 0
    candidates_processed = 0
    for input_path in root.rglob("predictions.jsonl"):
        records = []
        with input_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                candidates = record.get("generate_results", [])
                record["generate_results"] = [process_code(code) for code in candidates]
                candidates_processed += len(candidates)
                records.append(record)

        output_path = input_path.with_name("predictions_cleaned.jsonl")
        with output_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        files_processed += 1
        print(f"Cleaned {input_path}")
    return files_processed, candidates_processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Experiment directory or its predictions directory (default: experiments_results/pass@1_t0).",
    )
    args = parser.parse_args()
    root = predictions_root(args.experiment_root).resolve()
    if not root.is_dir():
        parser.error(f"Predictions directory does not exist: {root}")

    file_count, candidate_count = clean_predictions(root)
    print(f"Cleaned {candidate_count} candidate(s) in {file_count} prediction file(s).")


if __name__ == "__main__":
    main()
