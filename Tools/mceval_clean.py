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
    r"```\s*(?P<language>python|py|java)?[ \t]*\n?(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
JAVA_METHOD = re.compile(
    r"(?m)^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected)\s+)?"
    r"(?:(?:static|final|synchronized|abstract|native|strictfp)\s+)*"
    r"(?:<[^;\n{}()]+>\s+)?"
    r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;\n{}()]*>)?(?:\s*\[\])*(?:\s*\.\.\.)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:throws\s+[^{]+)?\{"
)
TRAILING_MARKERS = (
    "### It is your turn to generate",
    "### It is your turn now!",
    "### Test case",
    "Additional question:",
    "Answer:",
    "Explanation:",
    "Explanation of the code:",
    "<!---",
    "Here is",
    "Here's",
    "This function",
    "This implementation",
    "Please note",
    "Note that",
    "Test the function",
    "Example usage",
    "Usage:",
    "It can be seen",
    "analysisWe need",
    "analysis We need",
)
MODEL_FAMILY_MARKERS = {
    "qwen": (
        "The above code",
        "Please let me know",
    ),
    "codellama": (
        "[/INST]",
        "<s>",
        "</s>",
    ),
    "gpt_oss": (
        "<|return|>",
        "<|end|>",
        "<|call|>",
    ),
}
UNLABELED_ANALYSIS = re.compile(r"^\s*analysis(?:\b|[A-Z])", flags=re.IGNORECASE)
PROMPT_ECHO = re.compile(
    r"^\s*###\s*It is your turn to generate\b.*", flags=re.IGNORECASE | re.DOTALL
)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def model_family_from_path(input_path: Path) -> str:
    path_text = str(input_path).lower()
    if "qwen" in path_text:
        return "qwen"
    if "codellama" in path_text or "code_llama" in path_text:
        return "codellama"
    if "gpt-oss" in path_text or "gpt-20b" in path_text or "\\openai\\" in path_text or "/openai/" in path_text:
        return "gpt_oss"
    return "generic"


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


def looks_like_unlabeled_analysis(text: str) -> bool:
    return UNLABELED_ANALYSIS.search(text) is not None


def has_real_python_start(text: str, entry_point: str | None) -> bool:
    if entry_point and re.search(rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", text):
        return True
    return re.search(r"(?m)^[ \t]*(?:async\s+)?def\s+\w+\s*\(", text) is not None


def java_method_start_pattern(entry_point: str | None = None) -> str:
    name = re.escape(entry_point) if entry_point else r"[A-Za-z_]\w*"
    return (
        r"(?m)^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"
        r"(?:(?:public|private|protected)\s+)?"
        r"(?:(?:static|final|synchronized|abstract|native|strictfp)\s+)*"
        r"(?:<[^;\n{}()]+>\s+)?"
        r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;\n{}()]*>)?(?:\s*\[\])*(?:\s*\.\.\.)?)\s+"
        rf"{name}\s*\([^;{{}}]*\)\s*(?:throws\s+[^\{{]+)?\{{"
    )


def has_real_java_start(text: str, entry_point: str | None) -> bool:
    return re.search(java_method_start_pattern(entry_point), text) is not None


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


def trim_trailing_prose(text: str, model_family: str = "generic") -> str:
    markers = TRAILING_MARKERS + MODEL_FAMILY_MARKERS.get(model_family, ())
    for marker in markers:
        match = re.search(rf"(?:^|\n)\s*{re.escape(marker)}", text)
        if match:
            text = text[:match.start()]
    return text.strip()


def fallback_candidate(original_text: str) -> str:
    text = normalize_text(strip_harmony_wrappers(original_text))
    text = strip_inline_code_quotes(text)
    text = strip_bracket_tags(strip_leftover_fences(text))
    return normalize_text(text) or normalize_text(original_text)


def keep_nonempty(cleaned_text: str | None, original_text: str) -> str:
    cleaned_text = normalize_text(cleaned_text or "")
    if cleaned_text:
        return cleaned_text
    return fallback_candidate(original_text)


def strip_inline_code_quotes(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("`") and stripped.endswith("`") and not stripped.startswith("```"):
        text = stripped[1:-1]

    lines = []
    for line in text.splitlines():
        match = re.match(r"^([ \t]*)`([^`]+)`[ \t]*$", line)
        if match:
            lines.append(f"{match.group(1)}{match.group(2)}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def strip_bracket_tags(text: str) -> str:
    text = re.sub(r"(?im)^\s*\[(?:/?PYTHON|/?JAVA)\]\s*$", "", text)
    text = re.sub(r"(?ims)^\s*\[TESTS\].*", "", text)
    text = re.sub(r"(?i)\[/?(?:PYTHON|JAVA|TESTS)\]", "", text)
    return text.strip()


def strip_leftover_fences(text: str) -> str:
    text = re.sub(r"(?im)^\s*```\s*(?:python|py|java)?\s*$", "", text)
    text = text.replace("```", "")
    return text.strip()


def inline_signature(text: str, entry_point: str | None) -> str | None:
    if not entry_point:
        return None
    pattern = rf"`\s*(def\s+{re.escape(entry_point)}\s*\([^`]*?:)\s*`"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def restore_python_signature_if_needed(
    original_text: str,
    code: str,
    entry_point: str | None,
    signature: str | None,
) -> str:
    if not entry_point or re.search(rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code):
        return code

    header = inline_signature(original_text, entry_point) or signature
    if not header or not header.lstrip().startswith("def "):
        return code
    bare_signature = rf"(?m)^[ \t]*{re.escape(entry_point)}\s*\([^:\n]*\)\s*(?:->\s*[^:\n]+)?\s*:"
    if re.search(bare_signature, code):
        return re.sub(bare_signature, header.strip(), code, count=1)

    body_lines = code.splitlines()
    if not body_lines:
        return code
    indented_body = [
        line if not line.strip() or line.startswith((" ", "\t")) else f"    {line}"
        for line in body_lines
    ]
    return "\n".join([header.strip(), *indented_body])


def truncate_python_block(text: str, entry_point: str | None) -> str:
    if not entry_point:
        return text.strip()
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", line):
            start = index
            break
    if start is None:
        return text.strip()

    base_indent = len(lines[start]) - len(lines[start].lstrip(" \t"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= base_indent and not line.lstrip().startswith(("#", "@")):
            end = index
            break
    return "\n".join(lines[start:end]).rstrip()


def slice_from_code_start(text: str, language: str, entry_point: str | None) -> str:
    patterns = []
    if language == "python":
        if entry_point:
            patterns.append(rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
        patterns.extend((r"(?m)^[ \t]*(?:from\s+\S+\s+import\s+.+|import\s+.+)$", r"(?m)^[ \t]*(?:async\s+)?def\s+"))
    else:
        if entry_point:
            patterns.append(java_method_start_pattern(entry_point))
        patterns.append(java_method_start_pattern())

    starts = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            starts.append(match.start())
    return text[min(starts):] if starts else text


def extract_python_code(text: str, entry_point: str | None, model_family: str = "generic") -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "python", entry_point), model_family)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return truncate_python_block(text, entry_point)

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


def extract_java_methods(text: str, entry_point: str | None, model_family: str = "generic") -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "java", entry_point), model_family)
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
    return "\n\n".join(code for _, code in methods).strip()


def clean_common_candidate(
    text: str,
    language: str,
    entry_point: str | None,
    signature: str | None,
    model_family: str,
) -> str:
    original_text = text
    text = normalize_text(strip_harmony_wrappers(text))
    if looks_like_unlabeled_analysis(text):
        sliced = slice_from_code_start(text, language, entry_point)
        if sliced == text:
            return fallback_candidate(original_text)
        text = sliced
    if PROMPT_ECHO.match(text):
        text = slice_from_code_start(text, language, entry_point)
        if PROMPT_ECHO.match(text):
            return fallback_candidate(original_text)
    fenced = pick_fenced_code(text, language, entry_point)
    if fenced is not None:
        text = fenced
    text = normalize_text(strip_inline_code_quotes(text))
    if fenced is None and looks_like_unlabeled_analysis(text):
        if language == "python" and not has_real_python_start(text, entry_point):
            return fallback_candidate(original_text)
        if language == "java" and not has_real_java_start(text, entry_point):
            return fallback_candidate(original_text)
    text = strip_bracket_tags(strip_leftover_fences(text))
    if language == "python":
        if not has_real_python_start(text, entry_point) and has_real_java_start(text, None):
            return fallback_candidate(original_text)
        if not looks_like_unlabeled_analysis(text):
            text = restore_python_signature_if_needed(original_text, text, entry_point, signature)
        return keep_nonempty(extract_python_code(text, entry_point, model_family), original_text)
    if language == "java":
        if not has_real_java_start(text, entry_point) and has_real_python_start(text, None):
            return fallback_candidate(original_text)
        return keep_nonempty(extract_java_methods(text, entry_point, model_family), original_text)
    return keep_nonempty(trim_trailing_prose(text, model_family), original_text)


def clean_qwen_candidate(text: str, language: str, entry_point: str | None, signature: str | None) -> str:
    return clean_common_candidate(text, language, entry_point, signature, "qwen")


def clean_codellama_candidate(text: str, language: str, entry_point: str | None, signature: str | None) -> str:
    return clean_common_candidate(text, language, entry_point, signature, "codellama")


def clean_gpt_oss_candidate(text: str, language: str, entry_point: str | None, signature: str | None) -> str:
    return clean_common_candidate(text, language, entry_point, signature, "gpt_oss")


def clean_candidate(
    text: str,
    language: str,
    entry_point: str | None,
    signature: str | None,
    model_family: str = "generic",
) -> str:
    cleaners = {
        "qwen": clean_qwen_candidate,
        "codellama": clean_codellama_candidate,
        "gpt_oss": clean_gpt_oss_candidate,
    }
    cleaner = cleaners.get(model_family, clean_common_candidate)
    if cleaner is clean_common_candidate:
        return cleaner(text, language, entry_point, signature, "generic")
    return cleaner(text, language, entry_point, signature)


def language_from_record(record: dict) -> str:
    task_id = str(record.get("_id", record.get("task_id", "")))
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


def load_mceval_metadata(input_path: Path) -> dict[str, dict[str, str | None]]:
    full_path = input_path.with_name("predictions_mceval_full.jsonl")
    if not full_path.is_file():
        return {}

    metadata = {}
    with full_path.open("r", encoding="utf-8") as full_file:
        for line in full_file:
            record = json.loads(line)
            task_id = record.get("task_id", record.get("_id"))
            if task_id:
                metadata[str(task_id)] = {
                    "entry_point": record.get("entry_point"),
                    "signature": record.get("signature"),
                }
    return metadata


def clean_prediction_file(input_path: Path) -> tuple[Path, int]:
    records = []
    candidate_count = 0
    model_family = model_family_from_path(input_path)
    metadata_by_task_id = load_mceval_metadata(input_path)
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            language = language_from_record(record)
            task_id = record.get("_id", record.get("task_id"))
            task_metadata = metadata_by_task_id.get(str(task_id), {})
            entry_point = record.get("entry_point", task_metadata.get("entry_point"))
            signature = record.get("signature", task_metadata.get("signature"))
            candidates = record.get("generate_results", record.get("raw_generation"))
            if not isinstance(candidates, list) or not all(isinstance(code, str) for code in candidates):
                raise ValueError(f"Line {line_number} must contain a list of string generate_results.")
            record = {
                "_id": task_id,
                "generate_results": [
                    clean_candidate(candidate, language, entry_point, signature, model_family)
                    for candidate in candidates
                ],
            }
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
