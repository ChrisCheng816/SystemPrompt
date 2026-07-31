"""Export SystemPrompt predictions to the JSONL format used by HumanEval-X.

This script only prepares candidates for the official HumanEval-X evaluator;
it never executes model-generated code.  Use the benchmark's Docker-based
evaluator to run the untrusted candidates.

Example:
    python Tools/export_humaneval_x.py \
      --predictions experiments_results_human/pass@1_t0/predictions/<run>/predictions_cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="SystemPrompt predictions.jsonl or predictions_cleaned.jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: humaneval_x_generations.jsonl beside the input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {args.predictions}")

    output = args.output or args.predictions.with_name("humaneval_x_generations.jsonl")
    count = 0
    with args.predictions.open("r", encoding="utf-8") as input_file, output.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            task_id = record.get("_id")
            candidates = record.get("generate_results")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Line {line_number} has no valid _id/task_id.")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Line {line_number} has no generated candidates.")
            for candidate in candidates:
                if not isinstance(candidate, str):
                    raise ValueError(f"Line {line_number} contains a non-string candidate.")
                output_file.write(
                    json.dumps({"task_id": task_id, "generation": candidate}, ensure_ascii=False)
                    + "\n"
                )
                count += 1
    print(f"Exported {count} candidate(s) to {output}")


if __name__ == "__main__":
    main()
