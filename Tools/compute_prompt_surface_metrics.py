"""Compare the surface form of corresponding system prompts.

The analysis reports two punctuation-insensitive, word-level metrics:
* Jaccard lexical overlap: shared unique words divided by all unique words.
* Normalized Levenshtein distance: token edits divided by the longer prompt.

Use Jaccard overlap as the primary paper metric. It directly quantifies lexical
sharing; edit distance is retained as a robustness check because it also changes
when word order or prompt length changes.

Example:
    python Tools/compute_prompt_surface_metrics.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_SETS = ("original", "paraphrase_a", "paraphrase_b")
PAIRINGS = (
    ("original", "paraphrase_a"),
    ("original", "paraphrase_b"),
    ("paraphrase_a", "paraphrase_b"),
)
SET_LABELS = {
    "original": "Original",
    "paraphrase_a": "Imperative",
    "paraphrase_b": "Declarative",
}
PROMPT_LEVELS = ("Base", "Struct", "Robust", "Reason", "Edge")


def jaccard_overlap(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    """Return unique-token Jaccard overlap for two non-empty token sequences."""
    union = set(left_tokens) | set(right_tokens)
    if not union:
        raise ValueError("Cannot compute lexical overlap for two empty prompts.")
    return len(set(left_tokens) & set(right_tokens)) / len(union)


def levenshtein_distance(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> int:
    """Return the minimum number of word insertions, deletions, and substitutions."""
    if len(left_tokens) < len(right_tokens):
        left_tokens, right_tokens = right_tokens, left_tokens
    previous = list(range(len(right_tokens) + 1))
    for left_index, left_token in enumerate(left_tokens, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right_tokens, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    """Return token-level Levenshtein distance divided by the longer prompt."""
    denominator = max(len(left_tokens), len(right_tokens))
    if denominator == 0:
        raise ValueError("Cannot compute edit distance for two empty prompts.")
    return levenshtein_distance(left_tokens, right_tokens) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute lexical overlap and token-level edit distance for system prompts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Prompts" / "prompt_surface_metrics.csv",
        help="Per-prompt-pair CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "Prompts" / "prompt_surface_metrics_summary.csv",
        help="Pairwise-summary CSV path.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    from Prompts.prompt_sets import PROMPT_SETS
    from Tools.compute_prompt_entropy import lexical_tokens
    from Tools.compute_prompt_similarity import prompt_bundle_sha256

    missing = [name for name in EXPECTED_SETS if name not in PROMPT_SETS]
    if missing:
        raise ValueError(f"Missing prompt sets: {', '.join(missing)}")
    lengths = {name: len(PROMPT_SETS[name]) for name in EXPECTED_SETS}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != len(PROMPT_LEVELS):
        raise ValueError(f"Expected exactly five prompts in each set; found {lengths}.")

    bundle_hash = prompt_bundle_sha256(PROMPT_SETS)
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for left_name, right_name in PAIRINGS:
        comparison = f"{left_name}__{right_name}"
        for index, (left_text, right_text) in enumerate(
            zip(PROMPT_SETS[left_name], PROMPT_SETS[right_name], strict=True), start=1
        ):
            left_tokens = lexical_tokens(left_text)
            right_tokens = lexical_tokens(right_text)
            edit_distance = levenshtein_distance(left_tokens, right_tokens)
            row = {
                "comparison": comparison,
                "left_set": left_name,
                "left_label": SET_LABELS[left_name],
                "right_set": right_name,
                "right_label": SET_LABELS[right_name],
                "prompt_index": index,
                "prompt_level": PROMPT_LEVELS[index - 1],
                "left_token_count": len(left_tokens),
                "right_token_count": len(right_tokens),
                "lexical_jaccard_overlap": f"{jaccard_overlap(left_tokens, right_tokens):.8f}",
                "token_levenshtein_distance": edit_distance,
                "normalized_token_edit_distance": f"{normalized_edit_distance(left_tokens, right_tokens):.8f}",
                "prompt_bundle_sha256": bundle_hash,
            }
            rows.append(row)
            grouped[comparison].append(row)

    summary_rows: list[dict[str, object]] = []
    for comparison, comparison_rows in grouped.items():
        summary_rows.append(
            {
                "comparison": comparison,
                "left_label": comparison_rows[0]["left_label"],
                "right_label": comparison_rows[0]["right_label"],
                "prompt_pair_count": len(comparison_rows),
                "mean_lexical_jaccard_overlap": f"{sum(float(row['lexical_jaccard_overlap']) for row in comparison_rows) / len(comparison_rows):.8f}",
                "mean_normalized_token_edit_distance": f"{sum(float(row['normalized_token_edit_distance']) for row in comparison_rows) / len(comparison_rows):.8f}",
                "prompt_bundle_sha256": bundle_hash,
            }
        )

    write_rows(args.output, rows)
    write_rows(args.summary_output, summary_rows)
    print("Surface-form metrics (lower Jaccard / higher edit distance = more lexical change):")
    for row in summary_rows:
        print(
            f"  {row['left_label']} vs {row['right_label']}: "
            f"Jaccard={row['mean_lexical_jaccard_overlap']}, "
            f"normalized edit distance={row['mean_normalized_token_edit_distance']}"
        )
    print(f"Saved per-prompt-pair CSV: {args.output}")
    print(f"Saved summary CSV: {args.summary_output}")


if __name__ == "__main__":
    main()
