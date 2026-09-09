"""Compute lexical Shannon entropy for the three system-prompt sets.

The prompts use different punctuation and list formatting. This analysis therefore
computes entropy over lower-cased lexical tokens, excluding punctuation and layout
markers, so it measures wording diversity rather than formatting differences.

Example:
    python Tools/compute_prompt_entropy.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_SETS = ("original", "paraphrase_a", "paraphrase_b")
SET_LABELS = {
    "original": "Original",
    "paraphrase_a": "Imperative",
    "paraphrase_b": "Declarative",
}
PROMPT_LEVELS = ("Base", "Struct", "Robust", "Reason", "Edge")
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def lexical_tokens(text: str) -> list[str]:
    """Return a case-insensitive word-token sequence, excluding punctuation."""
    return WORD_PATTERN.findall(text.lower())


def shannon_entropy_bits(tokens: Sequence[str]) -> float:
    """Return Shannon entropy in bits for an empirical token distribution."""
    if not tokens:
        raise ValueError("Cannot calculate Shannon entropy for an empty prompt.")
    token_count = len(tokens)
    return -sum(
        (count / token_count) * math.log2(count / token_count)
        for count in Counter(tokens).values()
    )


def score_prompt(text: str) -> dict[str, object]:
    """Score one prompt and include the quantities needed for interpretation."""
    tokens = lexical_tokens(text)
    entropy = shannon_entropy_bits(tokens)
    vocabulary_size = len(set(tokens))
    normalized_entropy = entropy / math.log2(vocabulary_size) if vocabulary_size > 1 else 0.0
    return {
        "token_count": len(tokens),
        "vocabulary_size": vocabulary_size,
        "shannon_entropy_bits": entropy,
        "normalized_shannon_entropy": normalized_entropy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute word-level Shannon entropy for the three system-prompt sets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Prompts" / "prompt_entropy.csv",
        help="Per-prompt CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "Prompts" / "prompt_entropy_summary.csv",
        help="Prompt-set summary CSV path.",
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
    for prompt_set in EXPECTED_SETS:
        for index, text in enumerate(PROMPT_SETS[prompt_set], start=1):
            metrics = score_prompt(text)
            row = {
                "prompt_set": prompt_set,
                "set_label": SET_LABELS[prompt_set],
                "prompt_index": index,
                "prompt_level": PROMPT_LEVELS[index - 1],
                **{
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in metrics.items()
                },
                "prompt_bundle_sha256": bundle_hash,
            }
            rows.append(row)
            grouped[prompt_set].append(metrics)

    summary_rows: list[dict[str, object]] = []
    for prompt_set in EXPECTED_SETS:
        prompt_metrics = grouped[prompt_set]
        summary_rows.append(
            {
                "prompt_set": prompt_set,
                "set_label": SET_LABELS[prompt_set],
                "prompt_count": len(prompt_metrics),
                "mean_token_count": f"{sum(row['token_count'] for row in prompt_metrics) / len(prompt_metrics):.8f}",
                "mean_vocabulary_size": f"{sum(row['vocabulary_size'] for row in prompt_metrics) / len(prompt_metrics):.8f}",
                "mean_shannon_entropy_bits": f"{sum(row['shannon_entropy_bits'] for row in prompt_metrics) / len(prompt_metrics):.8f}",
                "mean_normalized_shannon_entropy": f"{sum(row['normalized_shannon_entropy'] for row in prompt_metrics) / len(prompt_metrics):.8f}",
                "prompt_bundle_sha256": bundle_hash,
            }
        )

    write_rows(args.output, rows)
    write_rows(args.summary_output, summary_rows)
    print("Per-prompt Shannon entropy (bits):")
    for row in rows:
        print(
            f"  {row['set_label']} | {row['prompt_level']}: "
            f"H={row['shannon_entropy_bits']}, H/log2(V)={row['normalized_shannon_entropy']}, "
            f"N={row['token_count']}, V={row['vocabulary_size']}"
        )
    print(f"Saved per-prompt CSV: {args.output}")
    print(f"Saved summary CSV: {args.summary_output}")


if __name__ == "__main__":
    main()
