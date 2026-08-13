"""Rebuild McEval matrices, McNemar/Holm reports, and Results summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from codereval_statistics import rebuild_statistics
except ModuleNotFoundError:
    from Tools.codereval_statistics import rebuild_statistics


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=REPO_ROOT / "experiments_results_mceval" / "pass@1_t0",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Exact parsed model name to omit; may be repeated (e.g. gpt-20b).",
    )
    args = parser.parse_args()
    record_count, report_counts = rebuild_statistics(
        args.experiment_root.resolve(),
        set(args.exclude_model),
        dataset="mceval",
    )
    print(f"[McEval statistics] rebuilt matrices from {record_count} verdict(s)")
    print(f"[McEval statistics] McNemar rows: java={report_counts['java']}, python={report_counts['python']}")


if __name__ == "__main__":
    main()
