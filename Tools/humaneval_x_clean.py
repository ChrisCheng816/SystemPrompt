"""Clean raw SystemPrompt generations for HumanEval-X Python and Java tasks.

This is intentionally separate from ``codereval_clean.py``.  In particular,
Python indentation is preserved because HumanEval-X completions are appended to
the function declaration contained in the benchmark prompt.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FENCED_CODE = re.compile(
    r"```\s*(?P<language>python|py|java)?\s*\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def remove_response_wrappers(text: str, language: str) -> str:
    """Remove chat/fence wrappers without stripping meaningful code indentation."""
    assistant_marker = re.search(r"assistantfinal", text, flags=re.IGNORECASE)
    if assistant_marker:
        text = text[assistant_marker.end():]

    blocks = list(FENCED_CODE.finditer(text))
    if blocks:
        aliases = {"python", "py"} if language == "python" else {"java"}
        matching_blocks = [
            block
            for block in blocks
            if block.group("language") and block.group("language").lower() in aliases
        ]
        text = (matching_blocks or blocks)[-1].group("code")

    # Only newlines are removed: leading spaces are part of a Python completion.
    return text.replace("\r\n", "\n").strip("\n")


def clean_prediction_file(input_path: Path) -> Path:
    """Create ``predictions_cleaned.jsonl`` for one HumanEval-X raw output."""
    records = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            task_id = record.get("_id", record.get("task_id"))
            if not isinstance(task_id, str) or "/" not in task_id:
                raise ValueError(f"Line {line_number} has no HumanEval-X task ID.")
            language = task_id.split("/", maxsplit=1)[0].lower()
            if language not in {"python", "java"}:
                raise ValueError(f"Line {line_number} has unsupported language: {language}")
            candidates = record.get("generate_results")
            if not isinstance(candidates, list) or not all(isinstance(code, str) for code in candidates):
                raise ValueError(f"Line {line_number} must contain string generate_results.")
            record["generate_results"] = [
                remove_response_wrappers(candidate, language) for candidate in candidates
            ]
            records.append(record)

    output_path = input_path.with_name("predictions_cleaned.jsonl")
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Raw HumanEval-X predictions.jsonl.")
    args = parser.parse_args()
    if not args.predictions.is_file():
        parser.error(f"Predictions file does not exist: {args.predictions}")
    print(f"Cleaned HumanEval-X predictions: {clean_prediction_file(args.predictions)}")


if __name__ == "__main__":
    main()
