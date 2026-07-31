"""Compute cosine similarity for the corresponding prompt pairs.

The default model and normalization method are deliberately the same as the
repository's retrieval pipeline in ``common_methods.get_retrieval_prompt``.

Examples:
    python Tools/compute_prompt_similarity.py
    python Tools/compute_prompt_similarity.py --device cuda:0
    python Tools/compute_prompt_similarity.py --output results/prompt_similarity.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*(\"\"\"|''')", re.MULTILINE
)


def extract_prompt_variables(path: Path, prefix: str) -> dict[int, str]:
    """Read ``<prefix><number> = '''...'''`` assignments from a prompt file.

    This intentionally does not import the file: prompt data can be inspected
    even when it contains an unfinished triple-quoted string during editing.
    """
    text = path.read_text(encoding="utf-8")
    prompts: dict[int, str] = {}

    for match in PROMPT_ASSIGNMENT.finditer(text):
        name, delimiter = match.groups()
        number_match = re.fullmatch(re.escape(prefix) + r"(\d+)", name)
        if number_match is None:
            continue

        content_start = match.end()
        content_end = text.find(delimiter, content_start)
        if content_end == -1:
            content_end = len(text)
            print(
                f"Warning: {path.name}:{name} has no closing {delimiter}; "
                "using the remaining file content.",
                file=sys.stderr,
            )
        prompts[int(number_match.group(1))] = text[content_start:content_end].strip()

    if not prompts:
        raise ValueError(f"No variables matching {prefix}<number> were found in {path}.")
    return prompts


def pair_prompts(
    source: dict[int, str], target: dict[int, str]
) -> Iterable[tuple[int, str, str]]:
    source_ids, target_ids = set(source), set(target)
    if source_ids != target_ids:
        missing_in_source = sorted(target_ids - source_ids)
        missing_in_target = sorted(source_ids - target_ids)
        raise ValueError(
            "Prompt indices do not correspond: "
            f"missing in source={missing_in_source}, missing in target={missing_in_target}."
        )
    for index in sorted(source_ids):
        yield index, source[index], target[index]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compute normalized embedding cosine similarity for corresponding prompts."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "Prompts" / "gen_prompts.py",
        help="File containing gen_prompt_1, gen_prompt_2, ...",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=project_root / "Prompts" / "prompts_sim.py",
        help="File containing shadow_prompt_1, shadow_prompt_2, ...",
    )
    parser.add_argument(
        "--source-prefix", default="gen_prompt_", help="Variable-name prefix in --source."
    )
    parser.add_argument(
        "--target-prefix", default="shadow_prompt_", help="Variable-name prefix in --target."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or path.")
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, e.g. cuda:0 or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Prompts" / "prompt_similarity.csv",
        help="CSV path for per-pair results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not args.source.is_file() or not args.target.is_file():
        raise FileNotFoundError("Both --source and --target must be existing files.")

    source = extract_prompt_variables(args.source, args.source_prefix)
    target = extract_prompt_variables(args.target, args.target_prefix)
    pairs = list(pair_prompts(source, target))

    # Import lazily so `--help` works without the experiment environment.
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading retriever: {args.model} ({device})")
    retriever = SentenceTransformer(args.model, device=device)

    source_embeddings = retriever.encode(
        [source_text for _, source_text, _ in pairs],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(pairs) > args.batch_size,
    )
    target_embeddings = retriever.encode(
        [target_text for _, _, target_text in pairs],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(pairs) > args.batch_size,
    )

    # Dot product of L2-normalized embeddings equals cosine similarity.
    scores = (source_embeddings * target_embeddings).sum(axis=1)
    rows = [
        {
            "prompt_index": index,
            "source_variable": f"{args.source_prefix}{index}",
            "target_variable": f"{args.target_prefix}{index}",
            "cosine_similarity": f"{float(score):.8f}",
            "source_characters": len(source_text),
            "target_characters": len(target_text),
        }
        for (index, source_text, target_text), score in zip(pairs, scores, strict=True)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nCorresponding prompt similarities:")
    for row in rows:
        print(f"  {row['source_variable']} <-> {row['target_variable']}: {row['cosine_similarity']}")
    mean_score = sum(float(row["cosine_similarity"]) for row in rows) / len(rows)
    print(f"Mean cosine similarity: {mean_score:.8f}")
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
