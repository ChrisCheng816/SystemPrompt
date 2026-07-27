"""Recompute CodeEval metrics for one experiments_results/pass@*_t* run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import evaluate
from codebleu import calc_codebleu
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments_results" / "pass@1_t0"
BLEU_METRIC = evaluate.load("bleu")


def predictions_root(experiment_root: Path) -> Path:
    return experiment_root if experiment_root.name == "predictions" else experiment_root / "predictions"


def find_prediction_output_pairs(root: Path) -> list[tuple[Path, Path]]:
    return [
        (path, path.with_name("output.json"))
        for path in root.rglob("predictions_cleaned.jsonl")
        if path.with_name("output.json").is_file()
    ]


def extract_value(text: str) -> tuple[float, float]:
    match = re.search(r"ngram match:\s*([0-9.]+),\s*weighted ngram match:\s*([0-9.]+)", text)
    if match is None:
        raise ValueError("CodeBLEU command did not produce ngram scores.")
    return float(match.group(1)), float(match.group(2))


def append_to_output(output_path: Path, bleu_smooth: dict, codebleu: dict) -> None:
    with output_path.open("r", encoding="utf-8") as output_file:
        data = json.load(output_file)
    data["BLEU_Smooth"] = bleu_smooth
    data["CodeBleu"] = codebleu
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def load_predictions(cleaned_path: Path) -> list[str]:
    predictions = []
    with cleaned_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            candidates = json.loads(line).get("generate_results", [])
            if not candidates:
                raise ValueError(f"No generated candidates in {cleaned_path}")
            predictions.append(candidates[0].strip())
    return predictions


def load_references(language: str, experiment_root: Path) -> tuple[list[str], Path]:
    dataset_name = "vitaleantonio/codereval-java" if language == "java" else "vitaleantonio/codereval-python"
    references = [output.strip() for output in load_dataset(dataset_name)["train"]["output"]]
    references_dir = experiment_root / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    reference_path = references_dir / f"{language}_references.txt"
    reference_path.write_text("\n".join(output.replace("\n", "") for output in references) + "\n", encoding="utf-8")
    return references, reference_path


def recompute_pair(cleaned_path: Path, output_path: Path, experiment_root: Path) -> None:
    with output_path.open("r", encoding="utf-8") as output_file:
        language = json.load(output_file)["language"]
    predictions = load_predictions(cleaned_path)
    references, reference_path = load_references(language, experiment_root)
    prediction_path = cleaned_path.with_name("predictions.txt")
    prediction_path.write_text("\n".join(text.replace("\n", "") for text in predictions) + "\n", encoding="utf-8")

    command = [
        sys.executable,
        str(REPO_ROOT / "Tools" / "calc_code_bleu.py"),
        "--refs",
        str(reference_path),
        "--hyp",
        str(prediction_path),
        "--lang",
        language,
        "--params",
        "0.25,0.25,0.25,0.25",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    ngram_match, weighted_ngram_match = extract_value(result.stdout)
    codebleu_score = calc_codebleu(references, predictions, language)
    codebleu = {
        "codebleu": ngram_match * 25 + weighted_ngram_match * 25
        + codebleu_score["syntax_match_score"] * 25 + codebleu_score["dataflow_match_score"] * 25,
        "ngram_match_score": ngram_match,
        "weighted_ngram_match_score": weighted_ngram_match,
        "syntax_match_score": codebleu_score["syntax_match_score"],
        "dataflow_match_score": codebleu_score["dataflow_match_score"],
    }
    append_to_output(output_path, BLEU_METRIC.compute(predictions=predictions, references=references, smooth=True), codebleu)


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

    pairs = find_prediction_output_pairs(root)
    for index, (cleaned_path, output_path) in enumerate(pairs, start=1):
        print(f"Processing {index}/{len(pairs)}: {cleaned_path}")
        recompute_pair(cleaned_path, output_path, root.parent)


if __name__ == "__main__":
    main()
