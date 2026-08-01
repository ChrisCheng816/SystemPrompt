"""Clean McEval raw generations in experiments_results_mceval runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments_results_mceval"
FENCED_CODE = re.compile(
    r"```\s*(?P<language>python|py|java)?\s*\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
JAVA_METHOD = re.compile(
    r"(?m)^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:public|private|protected)?\s*"
    r"(?:static\s+)?"
    r"(?:final\s+)?"
    r"(?:[\w<>\[\],.?]+\s+)+"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:throws\s+[^{]+)?\{"
)
TRAILING_MARKERS = (
    "Additional question:",
    "Answer:",
    "Explanation:",
    "Here is",
    "Here's",
    "This implementation",
    "Test the function",
    "Example usage",
    "Usage:",
)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_harmony_wrappers(text: str) -> str:
    final_markers = ("<|channel|>final<|message|>", "assistantfinal")
    lower_text = text.lower()
    for marker in final_markers:
        index = lower_text.rfind(marker.lower())
        if index != -1:
            return text[index + len(marker):].strip()
    text = re.sub(r"<\|(?:start|end|message|return|call)\|>", "", text)
    text = re.sub(r"<\|channel\|>\w+", "", text)
    return text.strip()


def pick_fenced_code(text: str, language: str, entry_point: str | None) -> str | None:
    blocks = list(FENCED_CODE.finditer(text))
    if not blocks:
        return None

    aliases = {"python", "py"} if language == "python" else {"java"}
    language_blocks = [
        block for block in blocks if (block.group("language") or "").lower() in aliases
    ]
    candidates = language_blocks or blocks
    if entry_point:
        for block in candidates:
            if re.search(rf"\b{re.escape(entry_point)}\s*\(", block.group("code")):
                return block.group("code")
    return candidates[0].group("code")


def trim_trailing_prose(text: str) -> str:
    for marker in TRAILING_MARKERS:
        match = re.search(rf"\n\s*{re.escape(marker)}", text)
        if match:
            text = text[:match.start()]
    return text.strip()


def slice_from_code_start(text: str, language: str, entry_point: str | None) -> str:
    patterns = []
    if language == "python":
        if entry_point:
            patterns.append(rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
        patterns.extend((r"(?m)^[ \t]*(?:from\s+\S+\s+import\s+.+|import\s+.+)$", r"(?m)^[ \t]*(?:async\s+)?def\s+"))
    else:
        if entry_point:
            patterns.append(rf"(?m)^[ \t]*(?:public|private|protected)?[^\n;{{}}]*\b{re.escape(entry_point)}\s*\(")
        patterns.append(r"(?m)^[ \t]*(?:public|private|protected)[^\n;{}]*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\{")

    starts = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            starts.append(match.start())
    return text[min(starts):] if starts else text


def extract_python_code(text: str, entry_point: str | None) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "python", entry_point))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text.strip()

    wanted_nodes = []
    found_entry = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            wanted_nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and entry_point == node.name:
                found_entry = True
            wanted_nodes.append(node)

    if entry_point and not found_entry:
        return text.strip()

    lines = text.splitlines()
    chunks = []
    for node in wanted_nodes:
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        chunks.append("\n".join(lines[start:end]).rstrip())
    return "\n\n".join(chunk for chunk in chunks if chunk).strip() or text.strip()


def matching_brace_index(text: str, open_index: int) -> int | None:
    if open_index < 0:
        return None

    depth = 0
    in_string = False
    string_quote = ""
    escaped = False
    in_line_comment = False
    in_block_comment = False

    for index in range(open_index, len(text)):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            continue
        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                in_string = False
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            continue
        if char in {"'", '"'}:
            in_string = True
            string_quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_java_methods(text: str, entry_point: str | None) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "java", entry_point))
    methods = []
    seen_spans = set()
    for match in JAVA_METHOD.finditer(text):
        open_index = text.find("{", match.end() - 1)
        close_index = matching_brace_index(text, open_index)
        if close_index is None:
            continue
        name = match.group("name")
        if name == "main":
            continue
        span = (match.start(), close_index + 1)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        methods.append((name, text[span[0]:span[1]].strip()))

    if entry_point and any(name == entry_point for name, _ in methods):
        ordered = []
        ordered.extend(code for name, code in methods if name == entry_point)
        ordered.extend(code for name, code in methods if name != entry_point)
        return "\n\n".join(ordered).strip()
    return "\n\n".join(code for _, code in methods).strip() or text.strip()


def clean_candidate(text: str, language: str, entry_point: str | None) -> str:
    text = normalize_text(strip_harmony_wrappers(text))
    fenced = pick_fenced_code(text, language, entry_point)
    if fenced is not None:
        text = fenced
    text = normalize_text(text)
    if language == "python":
        return extract_python_code(text, entry_point)
    if language == "java":
        return extract_java_methods(text, entry_point)
    return trim_trailing_prose(text)


def language_from_record(record: dict) -> str:
    task_id = str(record.get("task_id", ""))
    if "/" in task_id:
        language = task_id.split("/", maxsplit=1)[0].lower()
        if language in {"python", "java"}:
            return language
    raise ValueError(f"Unsupported or missing McEval task_id: {task_id!r}")


def search_root(experiment_root: Path) -> Path:
    if experiment_root.name == "predictions":
        return experiment_root
    predictions_dir = experiment_root / "predictions"
    if predictions_dir.is_dir():
        return predictions_dir
    return experiment_root


def clean_prediction_file(input_path: Path) -> tuple[Path, int]:
    records = []
    candidate_count = 0
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            language = language_from_record(record)
            entry_point = record.get("entry_point")
            candidates = record.get("raw_generation")
            if not isinstance(candidates, list) or not all(isinstance(code, str) for code in candidates):
                raise ValueError(f"Line {line_number} must contain a list of string raw_generation.")
            record["raw_generation"] = [
                clean_candidate(candidate, language, entry_point) for candidate in candidates
            ]
            candidate_count += len(candidates)
            records.append(record)

    output_path = input_path.with_name("predictions_cleaned.jsonl")
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path, candidate_count


def clean_predictions(root: Path) -> tuple[int, int]:
    files_processed = 0
    candidates_processed = 0
    for input_path in root.rglob("predictions.jsonl"):
        output_path, candidate_count = clean_prediction_file(input_path)
        files_processed += 1
        candidates_processed += candidate_count
        print(f"Cleaned {candidate_count} candidate(s): {output_path}")
    return files_processed, candidates_processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="McEval experiment directory or its predictions directory (default: experiments_results_mceval).",
    )
    args = parser.parse_args()
    root = search_root(args.experiment_root).resolve()
    if not root.is_dir():
        parser.error(f"Predictions directory does not exist: {root}")

    file_count, candidate_count = clean_predictions(root)
    print(f"Cleaned {candidate_count} candidate(s) in {file_count} prediction file(s).")


if __name__ == "__main__":
    main()
