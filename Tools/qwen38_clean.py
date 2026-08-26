"""Safely clean completed Qwen3.8-27B generations without touching other models.

This is deliberately separate from ``mceval_clean.py`` and
``codereval_clean.py``.  It reuses their proven extraction routines but adds a
strict Qwen3.8-only whitelist, complete-file validation, Qwen thinking-block
removal, post-clean structural checks, and an atomic per-file write.

Examples:
  # Inspect only; writes nothing.
  python Tools/qwen38_clean.py --dataset mceval --pass-dir pass@1_t1

  # Clean every complete Qwen3.8 run in the selected pass directory.
  python Tools/qwen38_clean.py --dataset mceval --pass-dir pass@1_t1 --write
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from evaluation_common import parse_run_name
    import codereval_clean
    import mceval_clean
except ModuleNotFoundError:  # Supports both `python Tools/...` and imports.
    from Tools.evaluation_common import parse_run_name
    from Tools import codereval_clean, mceval_clean


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN38_MODELS = frozenset({
    "qwen3.8-27b-original",
    "qwen3.8-27b-paraphrase-a",
    "qwen3.8-27b-paraphrase-b",
})
MODEL_DIRECTORY = "Qwen3.8_27b"
DATASETS = ("mceval", "codereval")
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
THINK_OPEN = re.compile(r"^\s*<think>.*?(?:</think>|$)", flags=re.IGNORECASE | re.DOTALL)
FENCE = re.compile(r"```")
STANDALONE_FENCE = re.compile(r"```[A-Za-z0-9_+#.\-]*")
ORPHAN_THINK_CLOSE = re.compile(r"[ \t]*</think>[ \t]*(?:\r?\n)?\Z", flags=re.IGNORECASE)
QWEN_TOKENS = re.compile(r"<\|[^>\n]*\|>")
LEADING_ANALYSIS = re.compile(r"^\s*(?:analysis|reasoning)\s*[:\n]", flags=re.IGNORECASE)
JAVA_TARGET = re.compile(
    r"(?m)^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected)\s+)?"
    r"(?:(?:static|final|synchronized|abstract|native|strictfp)\s+)*"
    r"(?:<[^;\n{}()]+>\s+)?"
    r"(?:(?:[A-Za-z_$][\w$]*\s*\.\s*)*[A-Za-z_$][\w$]*(?:\s*<[^;\n{}()]*>)?(?:\s*\[\s*\])*(?:\s*\.\.\.)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?:(?!```)[^();{}])*\)\s*"
    r"(?:throws\s+(?:(?!```)[^{])+)?\{"
)


@dataclass(frozen=True)
class ValidationResult:
    status: str
    records: list[dict[str, Any]] | None
    detail: str


def qwen_root(dataset: str, pass_dir: str, experiment_root: Path | None) -> Path:
    if experiment_root is not None:
        return experiment_root.resolve()
    return REPO_ROOT / f"experiments_results_{dataset}" / pass_dir / "predictions" / MODEL_DIRECTORY


def expected_task_count(dataset: str, language: str) -> int:
    if dataset == "mceval":
        return {"java": 53, "python": 50}[language]
    if dataset == "codereval":
        return 230
    raise ValueError(f"Unsupported dataset: {dataset}")


def expected_candidates(pass_dir: str) -> int:
    match = re.fullmatch(r"pass@(\d+)_t\d+(?:\.\d+)?", pass_dir)
    if match is None:
        raise ValueError(f"Invalid pass directory name: {pass_dir}")
    return int(match.group(1))


def read_complete_jsonl(path: Path, dataset: str, language: str, pass_dir: str) -> ValidationResult:
    """Read only fully formed, complete benchmark outputs; skip everything else."""
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    return ValidationResult("skipped_incomplete", None, f"blank line at {line_number}")
                row = json.loads(line)
                candidates = row.get("generate_results", row.get("raw_generation"))
                if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
                    return ValidationResult("skipped_invalid", None, f"invalid candidate list at line {line_number}")
                if len(candidates) != expected_candidates(pass_dir):
                    return ValidationResult(
                        "skipped_incomplete",
                        None,
                        f"line {line_number} has {len(candidates)} candidates; expected {expected_candidates(pass_dir)}",
                    )
                records.append(row)
    except json.JSONDecodeError as error:
        return ValidationResult("skipped_incomplete", None, f"invalid JSONL: {error.msg}")
    except OSError as error:
        return ValidationResult("skipped_io_error", None, str(error))

    expected = expected_task_count(dataset, language)
    if len(records) != expected:
        return ValidationResult("skipped_incomplete", None, f"{len(records)} records; expected {expected}")
    return ValidationResult("ready", records, f"{len(records)} complete records")


def strip_qwen38_reasoning(text: str) -> str:
    """Remove Qwen thinking blocks without touching content outside those blocks."""
    text = THINK_BLOCK.sub("", text)
    return THINK_OPEN.sub("", text).strip()


def _assigned_names(node: ast.AST) -> set[str]:
    return {
        candidate.id
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Store)
    }


def _node_bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names if alias.name != "*"}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return _assigned_names(node)
    return set()


def _target_function(tree: ast.Module, entry_point: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    return None


def _node_source(lines: list[str], node: ast.stmt) -> str:
    start = min(
        [node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", []))]
    )
    return "\n".join(lines[start - 1:getattr(node, "end_lineno", node.lineno)]).rstrip()


def _function_bound_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    bound = _assigned_names(node)
    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        bound.add(argument.arg)
    if node.args.vararg is not None:
        bound.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        bound.add(node.args.kwarg.arg)
    return bound


def _parse_raw_python_module(raw_text: str, entry_point: str) -> tuple[ast.Module, str] | None:
    candidates = [raw_text]
    fenced = codereval_clean.pick_target_fenced_code(raw_text, "python", entry_point)
    if fenced and fenced not in candidates:
        candidates.append(fenced)
    for candidate in candidates:
        try:
            return ast.parse(candidate), candidate
        except SyntaxError:
            continue
    return None


def restore_python_dependencies(raw_text: str, cleaned: str, entry_point: str | None) -> str:
    """Restore only raw top-level dependencies referenced by the cleaned target function.

    CodeEval replaces one function inside an existing project file.  Imports and
    helpers supplied by the model must therefore be placed inside the target
    function, where they are valid for both module functions and class methods.
    This runs only when both the raw and cleaned candidates parse exactly.
    """
    if not entry_point:
        return cleaned
    parsed_raw = _parse_raw_python_module(raw_text, entry_point)
    if parsed_raw is None:
        return cleaned
    raw_tree, raw_source = parsed_raw
    try:
        cleaned_tree = ast.parse(cleaned)
    except SyntaxError:
        return cleaned

    raw_target = _target_function(raw_tree, entry_point)
    cleaned_target = _target_function(cleaned_tree, entry_point)
    if raw_target is None or cleaned_target is None or not cleaned_target.body:
        return cleaned

    raw_nodes: dict[str, ast.stmt] = {}
    for node in raw_tree.body:
        if node is raw_target:
            continue
        for name in _node_bound_names(node):
            raw_nodes[name] = node

    available = _function_bound_names(cleaned_target)
    pending = {
        node.id
        for node in ast.walk(cleaned_target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - available
    selected: dict[int, ast.stmt] = {}
    while pending:
        name = pending.pop()
        node = raw_nodes.get(name)
        if node is None:
            continue
        node_id = id(node)
        if node_id in selected:
            continue
        selected[node_id] = node
        available.update(_node_bound_names(node))
        pending.update(
            candidate.id
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load)
        )
        pending.difference_update(available)

    if not selected:
        return cleaned

    lines = cleaned.splitlines()
    first_body = cleaned_target.body[0]
    if first_body.lineno <= cleaned_target.lineno:
        return cleaned
    header_line = lines[cleaned_target.lineno - 1]
    body_line = lines[first_body.lineno - 1]
    header_indent = len(header_line) - len(header_line.lstrip(" \t"))
    body_indent = len(body_line) - len(body_line.lstrip(" \t"))
    if body_indent <= header_indent:
        return cleaned

    support = "\n\n".join(
        _node_source(raw_source.splitlines(), node)
        for node in sorted(selected.values(), key=lambda item: item.lineno)
    ).strip()
    if not support:
        return cleaned
    if (
        isinstance(first_body, ast.Expr)
        and isinstance(getattr(first_body, "value", None), ast.Constant)
        and isinstance(first_body.value.value, str)
    ):
        insertion_index = getattr(first_body, "end_lineno", first_body.lineno)
    else:
        insertion_index = min(
            [first_body.lineno, *(decorator.lineno for decorator in getattr(first_body, "decorator_list", []))]
        ) - 1
    injected = textwrap.indent(support, body_line[:body_indent])
    return "\n".join([*lines[:insertion_index], injected, *lines[insertion_index:]]).strip()


def python_target_present(code: str, entry_point: str | None) -> bool:
    if not entry_point:
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return True
        if isinstance(node, ast.ClassDef):
            if any(isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == entry_point for member in node.body):
                return True
    return False


def java_target_present(code: str, entry_point: str | None) -> bool:
    if not entry_point:
        return True
    return any(match.group("name") == entry_point for match in JAVA_TARGET.finditer(code))


def _advance_triple_quote_state(line: str, quote: str | None) -> str | None:
    """Track Python/Java triple-quoted string state through one source line."""
    cursor = 0
    while cursor < len(line):
        if quote is None:
            positions = [
                (position, marker)
                for marker in ("'" * 3, '"' * 3)
                if (position := line.find(marker, cursor)) >= 0
            ]
            if not positions:
                break
            position, quote = min(positions, key=lambda item: item[0])
            cursor = position + len(quote)
        else:
            position = line.find(quote, cursor)
            if position < 0:
                break
            cursor = position + len(quote)
            quote = None
    return quote


def exterior_fence_index(code: str) -> int | None:
    """Return the first Markdown fence that is not inside a triple-quoted string."""
    quote: str | None = None
    offset = 0
    for line in code.splitlines(keepends=True):
        if quote is None and line.lstrip().startswith(chr(96) * 3):
            return offset
        quote = _advance_triple_quote_state(line, quote)
        offset += len(line)
    return None


def remove_exterior_presentation(code: str) -> str:
    """Drop an exterior fenced explanation while preserving literal fences in code strings."""
    index = exterior_fence_index(code)
    return code.strip() if index is None else code[:index].rstrip()


def remove_exterior_fence_lines(code: str) -> str:
    """Remove standalone Markdown fence lines only when they are outside source strings."""
    quote: str | None = None
    kept_lines: list[str] = []
    for line in code.splitlines(keepends=True):
        if quote is None and STANDALONE_FENCE.fullmatch(line.strip()):
            continue
        kept_lines.append(line)
        quote = _advance_triple_quote_state(line, quote)
    return "".join(kept_lines).strip()


def remove_exterior_orphan_think_close(code: str) -> str:
    """Remove standalone orphan think-close lines only when they are outside source strings."""
    quote: str | None = None
    kept_lines: list[str] = []
    for line in code.splitlines(keepends=True):
        if quote is None and ORPHAN_THINK_CLOSE.fullmatch(line):
            continue
        kept_lines.append(line)
        quote = _advance_triple_quote_state(line, quote)
    return "".join(kept_lines).strip()


def structural_issues(code: str, language: str, entry_point: str | None) -> list[str]:
    issues = []
    if exterior_fence_index(code) is not None:
        issues.append("residual_fence")
    if "<think" in code.lower() or "</think>" in code.lower():
        issues.append("residual_think_tag")
    if QWEN_TOKENS.search(code):
        issues.append("residual_qwen_token")
    if LEADING_ANALYSIS.search(code):
        issues.append("leading_analysis")
    if language == "python" and entry_point and not python_target_present(code, entry_point):
        issues.append("target_missing_or_unparsable")
    if language == "java" and entry_point and not java_target_present(code, entry_point):
        issues.append("target_method_missing")
    return issues


def clean_mceval_candidate(
    text: str,
    language: str,
    entry_point: str | None,
    signature: str | None,
) -> tuple[str, list[str]]:
    cleaned = mceval_clean.clean_candidate(
        strip_qwen38_reasoning(text), language, entry_point, signature, "qwen"
    )
    if language == "python":
        cleaned = remove_exterior_presentation(cleaned)
    cleaned = remove_exterior_orphan_think_close(cleaned)
    cleaned = remove_exterior_fence_lines(cleaned)
    if language == "python":
        cleaned = restore_python_dependencies(strip_qwen38_reasoning(text), cleaned, entry_point)
    return cleaned, structural_issues(cleaned, language, entry_point)


def clean_codereval_candidate(
    text: str,
    language: str,
    entry_point: str | None,
) -> tuple[str, list[str]]:
    cleaned = codereval_clean.clean_candidate_for_task(
        strip_qwen38_reasoning(text), language, "qwen", entry_point
    )
    if language == "python":
        cleaned = remove_exterior_presentation(cleaned)
    cleaned = remove_exterior_orphan_think_close(cleaned)
    cleaned = remove_exterior_fence_lines(cleaned)
    if language == "python":
        cleaned = restore_python_dependencies(strip_qwen38_reasoning(text), cleaned, entry_point)
    return cleaned, structural_issues(cleaned, language, entry_point)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, prefix=".qwen38-clean-", suffix=".jsonl"
    ) as output_file:
        temporary_path = Path(output_file.name)
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_manifest(root: Path, entry: dict[str, Any]) -> None:
    path = root / "cleaning_qwen38.jsonl"
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def clean_file(path: Path, dataset: str, pass_dir: str, write: bool) -> dict[str, Any]:
    run_info = parse_run_name(path.parent.name)
    if run_info.model not in QWEN38_MODELS:
        return {"run": path.parent.name, "status": "skipped_other_model"}
    validation = read_complete_jsonl(path, dataset, run_info.language, pass_dir)
    base = {
        "run": path.parent.name,
        "dataset": dataset,
        "language": run_info.language,
        "raw_file": str(path),
        "raw_sha256": sha256(path),
        "status": validation.status,
        "detail": validation.detail,
        "written": False,
    }
    if validation.records is None:
        return base

    issues: dict[str, int] = {}
    cleaned_records = []
    if dataset == "mceval":
        metadata_by_task = mceval_clean.load_mceval_metadata(path)
        for row in validation.records:
            task_id = row.get("_id", row.get("task_id"))
            metadata = metadata_by_task.get(str(task_id), {})
            language = mceval_clean.language_from_record(row)
            entry_point = row.get("entry_point", metadata.get("entry_point"))
            signature = row.get("signature", metadata.get("signature"))
            candidates = []
            for candidate in row.get("generate_results", row.get("raw_generation")):
                cleaned, candidate_issues = clean_mceval_candidate(candidate, language, entry_point, signature)
                candidates.append(cleaned)
                for issue in candidate_issues:
                    issues[issue] = issues.get(issue, 0) + 1
            cleaned_records.append({"_id": task_id, "generate_results": candidates})
    else:
        entry_points = codereval_clean.load_codereval_entry_points(run_info.language)
        for row in validation.records:
            task_id = row.get("_id", row.get("id"))
            candidates = []
            for candidate in row.get("generate_results", row.get("raw_generation")):
                cleaned, candidate_issues = clean_codereval_candidate(candidate, run_info.language, entry_points.get(str(task_id)))
                candidates.append(cleaned)
                for issue in candidate_issues:
                    issues[issue] = issues.get(issue, 0) + 1
            cleaned_records.append({"_id": task_id, "generate_results": candidates})

    # A cleanable presentation artifact must be eliminated.  Parser/signature
    # failures are retained and logged because they are genuine model outcomes.
    presentation_issues = {key: count for key, count in issues.items() if key.startswith("residual_") or key == "leading_analysis"}
    base.update({
        "records": len(cleaned_records),
        "candidates": sum(len(row["generate_results"]) for row in cleaned_records),
        "structural_issues": issues,
        "presentation_issues": presentation_issues,
    })
    if presentation_issues:
        base["status"] = "blocked_residual_presentation"
        return base

    if write:
        output_path = path.with_name("predictions_cleaned.jsonl")
        atomic_write_jsonl(output_path, cleaned_records)
        base.update({"status": "cleaned", "written": True, "cleaned_file": str(output_path)})
    else:
        base["status"] = "dry_run_ready"
    return base


def process_root(dataset: str, pass_dir: str, root: Path, write: bool) -> tuple[list[dict[str, Any]], int]:
    if root.name != MODEL_DIRECTORY:
        raise ValueError(f"For safety, --experiment-root must end in {MODEL_DIRECTORY}, got {root}")
    if not root.is_dir():
        raise FileNotFoundError(root)
    results = []
    skipped_other = 0
    for path in sorted(root.rglob("predictions.jsonl")):
        result = clean_file(path, dataset, pass_dir, write)
        if result["status"] == "skipped_other_model":
            skipped_other += 1
            continue
        results.append(result)
    if write:
        for result in results:
            append_manifest(root, result)
    return results, skipped_other


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--pass-dir", default="pass@1_t1", choices=("pass@1_t0", "pass@1_t1", "pass@5_t1"))
    parser.add_argument(
        "--experiment-root", type=Path,
        help=f"Exact Qwen3.8 predictions directory; must end in {MODEL_DIRECTORY}.",
    )
    parser.add_argument("--write", action="store_true", help="Write cleaned JSONL and the Qwen3.8-only manifest. Default is read-only dry-run.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every discovered Qwen3.8 file was ready and selected for this mode.",
    )
    args = parser.parse_args()

    root = qwen_root(args.dataset, args.pass_dir, args.experiment_root)
    results, skipped_other = process_root(args.dataset, args.pass_dir, root, args.write)
    if not results:
        raise SystemExit(f"No whitelisted Qwen3.8 prediction files found under {root}")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    summary = {
        "dataset": args.dataset,
        "pass_dir": args.pass_dir,
        "mode": "write" if args.write else "dry_run",
        "root": str(root),
        "runs_seen": len(results),
        "cleaned": sum(row["status"] == "cleaned" for row in results),
        "ready": sum(row["status"] == "dry_run_ready" for row in results),
        "incomplete": sum(str(row["status"]).startswith("skipped_incomplete") for row in results),
        "blocked": sum(row["status"] == "blocked_residual_presentation" for row in results),
        "skipped_other_models": skipped_other,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    expected_status = "cleaned" if args.write else "dry_run_ready"
    not_ready = [row for row in results if row["status"] != expected_status]
    if args.require_complete and not_ready:
        raise SystemExit(
            f"Refusing evaluation: {len(not_ready)} Qwen3.8 run(s) are not {expected_status}."
        )


if __name__ == "__main__":
    main()
