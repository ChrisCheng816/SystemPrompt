"""Compute cosine similarity for the three corresponding prompt sets.

Examples:
    python Tools/compute_prompt_similarity.py --device cpu --history-output Prompts/prompt_similarity_iterations.csv
    python Tools/compute_prompt_similarity.py --model Qwen/Qwen3-Embedding-0.6B --history-output Prompts/prompt_similarity_iterations.csv
    python Tools/compute_prompt_similarity.py --report-input Prompts/prompt_similarity_iterations.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_SETS = ("original", "paraphrase_a", "paraphrase_b")
PAIRINGS = (
    ("original", "paraphrase_a"),
    ("original", "paraphrase_b"),
    ("paraphrase_a", "paraphrase_b"),
)


def build_comparisons(
    prompt_sets: Mapping[str, Sequence[str]],
) -> list[tuple[str, str, int, str, str]]:
    """Pair corresponding prompt indices across all three prompt sets."""
    missing = [name for name in EXPECTED_SETS if name not in prompt_sets]
    if missing:
        raise ValueError(f"Missing prompt sets: {', '.join(missing)}")

    lengths = {name: len(prompt_sets[name]) for name in EXPECTED_SETS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Prompt sets must contain the same number of entries: {lengths}")

    comparisons = []
    for left_name, right_name in PAIRINGS:
        for index, (left_text, right_text) in enumerate(
            zip(prompt_sets[left_name], prompt_sets[right_name], strict=True), start=1
        ):
            comparisons.append((left_name, right_name, index, left_text, right_text))
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalized embedding cosine similarity for all corresponding prompt pairs."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=f"SentenceTransformer model name or path. Repeat for multiple models. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, e.g. cuda:0 or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number recorded in CSV output.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Score CSV. Defaults to a model-specific path so Qwen cannot overwrite MiniLM.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=None,
        help="Optional CSV to append this iteration to.",
    )
    parser.add_argument(
        "--report-input",
        action="append",
        type=Path,
        default=[],
        help="Build a model-selection summary from existing score/history CSVs; repeat as needed.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "Prompts" / "prompt_similarity_summary.csv",
        help="Summary CSV used with --report-input or a multi-model run.",
    )
    return parser.parse_args()


def default_output_path(models: Sequence[str]) -> Path:
    """Choose a non-colliding report path for the requested model set."""
    if len(models) != 1:
        filename = "prompt_similarity_models.csv"
    elif models[0] == DEFAULT_MODEL:
        filename = "prompt_similarity.csv"
    elif "qwen" in models[0].lower():
        filename = "prompt_similarity_qwen.csv"
    else:
        slug = "".join(character if character.isalnum() else "-" for character in models[0])
        filename = f"prompt_similarity_{slug.strip('-').lower()}.csv"
    return PROJECT_ROOT / "Prompts" / filename


def prompt_bundle_sha256(prompt_sets: Mapping[str, Sequence[str]]) -> str:
    """Return a stable content hash proving which exact prompt bundle was scored."""
    payload = {name: list(prompt_sets[name]) for name in EXPECTED_SETS}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_report_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as report_file:
            rows.extend(csv.DictReader(report_file))
    if not rows:
        raise ValueError("No score rows were found in --report-input files.")
    return rows


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Summarize each run and globally select one model on original-pair mean."""
    grouped: dict[tuple[int, str], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    hashes: dict[tuple[int, str], set[str]] = defaultdict(set)
    runs_with_missing_hashes: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["iteration"]), str(row["model"]))
        comparison = str(row["comparison"])
        prompt_index = int(row["prompt_index"])
        if prompt_index in grouped[key][comparison]:
            raise ValueError(
                f"Run {key[0]}/{key[1]} has duplicate index {prompt_index} for {comparison}."
            )
        grouped[key][comparison][prompt_index] = float(row["cosine_similarity"])
        content_hash = str(row.get("prompt_bundle_sha256", ""))
        if content_hash:
            hashes[key].add(content_hash)
        else:
            runs_with_missing_hashes.add(key)

    required = {f"{left}__{right}" for left, right in PAIRINGS}
    summaries: list[dict[str, object]] = []
    for (iteration, model_name), comparison_scores in sorted(grouped.items()):
        missing = required - comparison_scores.keys()
        if missing:
            raise ValueError(
                f"Run {iteration}/{model_name} is missing comparisons: {', '.join(sorted(missing))}"
            )
        expected_indices = set(range(1, 6))
        for comparison, indexed_scores in comparison_scores.items():
            if set(indexed_scores) != expected_indices:
                raise ValueError(
                    f"Run {iteration}/{model_name} must contain prompt indices 1-5 "
                    f"exactly once for {comparison}."
                )
        if len(hashes[(iteration, model_name)]) > 1:
            raise ValueError(f"Run {iteration}/{model_name} mixes multiple prompt bundles.")
        means = {
            comparison: sum(indexed_scores.values()) / len(indexed_scores)
            for comparison, indexed_scores in comparison_scores.items()
        }
        original_mean = (
            means["original__paraphrase_a"] + means["original__paraphrase_b"]
        ) / 2
        summaries.append(
            {
                "iteration": iteration,
                "model": model_name,
                "prompt_bundle_sha256": next(iter(hashes[(iteration, model_name)]), ""),
                "original_paraphrase_a_mean": f'{means["original__paraphrase_a"]:.8f}',
                "original_paraphrase_b_mean": f'{means["original__paraphrase_b"]:.8f}',
                "original_pairs_overall_mean": f"{original_mean:.8f}",
                "paraphrase_a_paraphrase_b_mean": f'{means["paraphrase_a__paraphrase_b"]:.8f}',
                "selected_final_model": False,
            }
        )

    latest_by_model: dict[str, dict[str, object]] = {}
    for summary in summaries:
        model_name = str(summary["model"])
        if model_name not in latest_by_model or int(summary["iteration"]) > int(
            latest_by_model[model_name]["iteration"]
        ):
            latest_by_model[model_name] = summary

    for summary in latest_by_model.values():
        key = (int(summary["iteration"]), str(summary["model"]))
        if key in runs_with_missing_hashes or not summary["prompt_bundle_sha256"]:
            raise ValueError(
                "Every row in each latest model report must include a prompt bundle hash."
            )
    candidate_hashes = {
        str(summary["prompt_bundle_sha256"])
        for summary in latest_by_model.values()
    }
    if len(candidate_hashes) > 1:
        raise ValueError("Latest model reports score different prompt bundles; selection is invalid.")
    winner = max(
        latest_by_model.values(), key=lambda row: float(row["original_pairs_overall_mean"])
    )
    winner["selected_final_model"] = True
    return summaries


def _write_rows(path: Path, rows: list[dict[str, object]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.iteration < 0:
        raise ValueError("--iteration cannot be negative.")

    if args.report_input:
        if args.models:
            raise ValueError("--report-input cannot be combined with --model.")
        summary_rows = summarize_rows(read_report_rows(args.report_input))
        _write_rows(args.summary_output, summary_rows)
        winner = next(row for row in summary_rows if row["selected_final_model"])
        print(
            f"Selected model: {winner['model']} "
            f"(global original-pair mean {winner['original_pairs_overall_mean']})"
        )
        print(f"Saved summary: {args.summary_output}")
        return

    import torch
    from sentence_transformers import SentenceTransformer

    from Prompts.prompt_sets import PROMPT_SETS

    comparisons = build_comparisons(PROMPT_SETS)
    models = args.models or [DEFAULT_MODEL]
    output_path = args.output or default_output_path(models)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    bundle_hash = prompt_bundle_sha256(PROMPT_SETS)

    for model_name in models:
        print(f"Loading similarity model: {model_name} ({device})")
        encoder = SentenceTransformer(model_name, device=device)
        embeddings = {
            set_name: encoder.encode(
                list(PROMPT_SETS[set_name]),
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for set_name in EXPECTED_SETS
        }

        for left_name, right_name, index, left_text, right_text in comparisons:
            score = float(
                (embeddings[left_name][index - 1] * embeddings[right_name][index - 1]).sum()
            )
            rows.append(
                {
                    "iteration": args.iteration,
                    "model": model_name,
                    "comparison": f"{left_name}__{right_name}",
                    "prompt_index": index,
                    "cosine_similarity": f"{score:.8f}",
                    "left_words": len(left_text.split()),
                    "right_words": len(right_text.split()),
                    "prompt_bundle_sha256": bundle_hash,
                }
            )

        del encoder

    _write_rows(output_path, rows)
    if args.history_output is not None:
        _write_rows(args.history_output, rows, append=True)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["comparison"]))].append(
            float(row["cosine_similarity"])
        )

    print("\nCorresponding prompt similarities:")
    for row in rows:
        print(
            f"  {row['model']} | {row['comparison']}[{row['prompt_index']}]: "
            f"{row['cosine_similarity']}"
        )
    print("\nMeans:")
    for (model_name, comparison), scores in grouped.items():
        print(f"  {model_name} | {comparison}: {sum(scores) / len(scores):.8f}")
    print(f"Saved CSV: {output_path}")
    if args.history_output is not None:
        print(f"Appended history: {args.history_output}")
    if len(models) > 1:
        summary_rows = summarize_rows(rows)
        _write_rows(args.summary_output, summary_rows)
        winner = next(row for row in summary_rows if row["selected_final_model"])
        print(
            f"Selected model: {winner['model']} "
            f"(global original-pair mean {winner['original_pairs_overall_mean']})"
        )
        print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
