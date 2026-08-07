"""Clean CodeEval predictions in one experiments_results_codereval/pass@*_t* run."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments_results_codereval" / "pass@1_t0"
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
    "### It is your turn",
    "### It is your turn to generate",
    "### It is your turn now!",
    "### It is your turn again!",
    "### Example",
    "### Test case",
    "Input:",
    "Output:",
    "Additional question:",
    "Answer:",
    "Explanation:",
    "Explanation of the code:",
    "Here is",
    "Here's",
    "This function",
    "This implementation",
    "Please note",
    "Note that",
    "Example usage",
    "Usage:",
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


def language_from_path(input_path: Path) -> str | None:
    path_text = str(input_path).lower()
    if "_python_" in path_text or "\\python_" in path_text or "/python_" in path_text:
        return "python"
    if "_java_" in path_text or "\\java_" in path_text or "/java_" in path_text:
        return "java"
    return None


def strip_harmony_wrappers(text: str) -> str:
    final_markers = ("<|channel|>final<|message|>", "assistantfinal")
    lower_text = text.lower()
    for marker in final_markers:
        index = lower_text.rfind(marker.lower())
        if index != -1:
            return text[index + len(marker):].strip()
    text = re.sub(r"<\|(?:start|message)\|>", "", text)
    text = re.sub(r"<\|channel\|>\w+", "", text)
    return text.strip()


def strip_inline_code_quotes(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("`") and not stripped.startswith("```"):
        text = stripped[1:-1] if stripped.endswith("`") else stripped[1:]

    lines = []
    for line in text.splitlines():
        match = re.match(r"^([ \t]*)`([^`]+)`[ \t]*$", line)
        lines.append(f"{match.group(1)}{match.group(2)}" if match else line)
    return "\n".join(lines).strip()


def strip_leftover_fences(text: str) -> str:
    text = re.sub(r"(?im)^\s*```\s*(?:python|py|java)?\s*$", "", text)
    return text.replace("```", "").strip()


def strip_bracket_tags(text: str) -> str:
    text = re.sub(r"(?im)^\s*\[(?:/?PYTHON|/?JAVA|/?INST)\]\s*$", "", text)
    text = re.sub(r"(?ims)^\s*\[TESTS\].*", "", text)
    text = re.sub(r"(?i)\[/?(?:PYTHON|JAVA|TESTS|INST)\]", "", text)
    text = re.sub(r"(?i)\[/INST\]", "", text)
    return text.strip()


def trim_trailing_prose(text: str, model_family: str) -> str:
    markers = TRAILING_MARKERS + MODEL_FAMILY_MARKERS.get(model_family, ())
    for marker in markers:
        match = re.search(rf"(?:^|\n)\s*{re.escape(marker)}", text)
        if match:
            text = text[:match.start()]
    return text.strip()


def fallback_candidate(original_text: str, language: str | None = None) -> str:
    raw_text = normalize_text(original_text)
    text = normalize_text(strip_harmony_wrappers(original_text))
    text = strip_bracket_tags(strip_leftover_fences(strip_inline_code_quotes(text)))
    text = normalize_text(text)
    if text:
        return text
    if raw_text:
        if language == "python":
            return "pass"
        if language == "java":
            return "/* empty generation */"
    return raw_text


def keep_nonempty(cleaned_text: str | None, original_text: str, language: str | None = None) -> str:
    cleaned_text = normalize_text(cleaned_text or "")
    if cleaned_text:
        return cleaned_text
    return fallback_candidate(original_text, language)


def pick_fenced_code(text: str, language: str | None) -> str | None:
    blocks = list(FENCED_CODE.finditer(text))
    if not blocks:
        return None
    if language:
        aliases = {"python", "py"} if language == "python" else {"java"}
        for block in blocks:
            if (block.group("language") or "").lower() in aliases:
                return block.group("code")
    return blocks[0].group("code")


def python_code_start(text: str) -> int | None:
    patterns = (
        r"(?m)^[ \t]*(?:from\s+\S+\s+import\s+.+|import\s+.+)$",
        r"(?m)^[ \t]*(?:async\s+)?def\s+\w+\s*\(",
        r"(?m)^[ \t]*class\s+\w+",
    )
    starts = [match.start() for pattern in patterns for match in [re.search(pattern, text)] if match]
    return min(starts) if starts else None


def java_code_start(text: str) -> int | None:
    starts = [match.start() for match in JAVA_METHOD.finditer(text)]
    class_match = re.search(r"(?m)^[ \t]*(?:public\s+)?class\s+\w+", text)
    if class_match:
        starts.append(class_match.start())
    return min(starts) if starts else None


def slice_from_code_start(text: str, language: str | None) -> str:
    if language == "python":
        start = python_code_start(text)
    elif language == "java":
        start = java_code_start(text)
    else:
        starts = [value for value in (python_code_start(text), java_code_start(text)) if value is not None]
        start = min(starts) if starts else None
    return text[start:] if start is not None else text


def extract_python_code(text: str, model_family: str) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "python"), model_family)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text.strip()

    lines = text.splitlines()
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            chunks.append("\n".join(lines[start:end]).rstrip())
    return "\n\n".join(chunk for chunk in chunks if chunk).strip() or text.strip()


def matching_brace_index(text: str, open_index: int) -> int | None:
    if open_index < 0:
        return None

    depth = 0
    in_string = False
    quote = ""
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
            elif char == quote:
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
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_java_code(text: str, model_family: str) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "java"), model_family)
    methods = []
    for match in JAVA_METHOD.finditer(text):
        if match.group("name") == "main":
            continue
        open_index = text.find("{", match.end() - 1)
        close_index = matching_brace_index(text, open_index)
        if close_index is not None:
            methods.append(text[match.start():close_index + 1].strip())
    if methods:
        return "\n\n".join(methods).strip()

    class_match = re.search(r"(?m)^[ \t]*(?:public\s+)?class\s+\w+.*?\{", text)
    if class_match:
        open_index = text.find("{", class_match.end() - 1)
        close_index = matching_brace_index(text, open_index)
        if close_index is not None:
            return text[class_match.start():close_index + 1].strip()
    return text.strip()


def clean_common_candidate(text: str, language: str | None, model_family: str) -> str:
    original_text = text
    text = normalize_text(strip_harmony_wrappers(text))
    if UNLABELED_ANALYSIS.search(text):
        sliced = slice_from_code_start(text, language)
        if sliced == text:
            return fallback_candidate(original_text, language)
        text = sliced

    fenced = pick_fenced_code(text, language)
    if fenced is not None:
        text = fenced
    text = strip_bracket_tags(strip_leftover_fences(strip_inline_code_quotes(normalize_text(text))))
    if language == "python":
        return keep_nonempty(extract_python_code(text, model_family), original_text, language)
    if language == "java":
        return keep_nonempty(extract_java_code(text, model_family), original_text, language)
    return keep_nonempty(trim_trailing_prose(slice_from_code_start(text, language), model_family), original_text, language)


def clean_qwen_candidate(text: str, language: str | None) -> str:
    return clean_common_candidate(text, language, "qwen")


def clean_codellama_candidate(text: str, language: str | None) -> str:
    return clean_common_candidate(text, language, "codellama")


def clean_gpt_oss_candidate(text: str, language: str | None) -> str:
    return clean_common_candidate(text, language, "gpt_oss")


def clean_candidate(text: str, language: str | None, model_family: str) -> str:
    cleaners = {
        "qwen": clean_qwen_candidate,
        "codellama": clean_codellama_candidate,
        "gpt_oss": clean_gpt_oss_candidate,
    }
    return cleaners.get(model_family, lambda value, lang: clean_common_candidate(value, lang, "generic"))(text, language)


def predictions_root(experiment_root: Path) -> Path:
    if experiment_root.name == "predictions":
        return experiment_root
    direct_predictions = experiment_root / "predictions"
    if direct_predictions.is_dir():
        return direct_predictions
    return experiment_root


def clean_prediction_file(input_path: Path) -> tuple[Path, int]:
    model_family = model_family_from_path(input_path)
    language = language_from_path(input_path)
    records = []
    candidates_processed = 0
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            candidates = record.get("generate_results", record.get("raw_generation", []))
            if not isinstance(candidates, list) or not all(isinstance(code, str) for code in candidates):
                raise ValueError(f"Line {line_number} must contain a list of string generate_results.")
            records.append(
                {
                    "_id": record.get("_id", record.get("id")),
                    "generate_results": [
                        clean_candidate(candidate, language, model_family)
                        for candidate in candidates
                    ],
                }
            )
            candidates_processed += len(candidates)

    output_path = input_path.with_name("predictions_cleaned.jsonl")
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path, candidates_processed


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
        help="Experiment directory or its predictions directory (default: experiments_results_codereval/pass@1_t0).",
    )
    args = parser.parse_args()
    root = predictions_root(args.experiment_root).resolve()
    if not root.is_dir():
        parser.error(f"Predictions directory does not exist: {root}")

    file_count, candidate_count = clean_predictions(root)
    print(f"Cleaned {candidate_count} candidate(s) in {file_count} prediction file(s).")


if __name__ == "__main__":
    main()
