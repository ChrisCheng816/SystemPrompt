"""Clean CodeEval predictions in one experiments_results_codereval/pass@*_t* run."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path

try:
    from evaluation_common import parse_run_name
except ModuleNotFoundError:  # Supports `python Tools/...` and `import Tools...`.
    from Tools.evaluation_common import parse_run_name


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments_results_codereval" / "pass@1_t0"
FENCED_CODE = re.compile(
    r"```[ \t]*(?:(?P<language>[A-Za-z0-9_+#.\\-]+)[ \t]*\r?\n)?(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
JAVA_METHOD = re.compile(
    r"(?m)^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected)\s+)?"
    r"(?:(?:static|final|synchronized|abstract|native|strictfp)\s+)*"
    r"(?:<[^;\n{}()]+>\s+)?"
    r"(?:(?:[A-Za-z_$][\w$]*\s*\.\s*)*[A-Za-z_$][\w$]*(?:\s*<[^;\n{}()]*>)?(?:\s*\[\s*\])*(?:\s*\.\.\.)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?:(?!```)[^();{}])*\)\s*"
    r"(?:throws\s+(?:(?!```)[^{])+)?\{"
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
    text = re.sub(r"<\|[^>\n]*\|>", "", text)
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
    text = re.sub(r"(?im)^\s*```\s*[A-Za-z0-9_+#.\-]*\s*$", "", text)
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
    matches = [
        match.start()
        for pattern in (r"(?m)^[ \t]*(?:async\s+)?def\s+\w+\s*\(", r"(?m)^[ \t]*class\s+\w+")
        for match in [re.search(pattern, text)]
        if match is not None
    ]
    return min(matches) if matches else None


def _python_node_source(lines: list[str], node: ast.AST) -> str:
    starts = [node.lineno]
    starts.extend(decorator.lineno for decorator in getattr(node, "decorator_list", []))
    start = min(starts) - 1
    end = getattr(node, "end_lineno", node.lineno)
    return textwrap.dedent("\n".join(lines[start:end])).rstrip()


def _first_python_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("test_"):
            return node
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and not method.name.startswith(("test_", "__")):
                    return method
    return None


def truncate_python_block(text: str) -> str:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if re.match(r"^[ \t]*(?:async\s+)?def\s+\w+\s*\(", line)), None)
    if start is None:
        return text.strip()
    base_indent = len(lines[start]) - len(lines[start].lstrip(" \t"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and len(line) - len(line.lstrip(" \t")) <= base_indent:
            end = index
            break
    return textwrap.dedent("\n".join(lines[start:end])).rstrip()


def java_code_start(text: str) -> int | None:
    match = next(iter(JAVA_METHOD.finditer(text)), None)
    return match.start() if match else None


def slice_from_code_start(
    text: str, language: str | None, entry_point: str | None = None,
) -> str:
    if language == "python" and entry_point:
        match = re.search(
            rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", text,
        )
        start = match.start() if match is not None else python_code_start(text)
    elif language == "java" and entry_point:
        match = next((candidate for candidate in JAVA_METHOD.finditer(text) if candidate.group("name") == entry_point), None)
        start = match.start() if match is not None else java_code_start(text)
    else:
        start = python_code_start(text) if language == "python" else java_code_start(text) if language == "java" else None
    return text[start:] if start is not None else text


def extract_python_code(text: str, model_family: str) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "python"), model_family)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return truncate_python_block(text)
    function = _first_python_function(tree)
    return _python_node_source(text.splitlines(), function) if function is not None else text.strip()

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
        return methods[0]

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


def extract_target_python_code(text: str, entry_point: str, model_family: str) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "python", entry_point), model_family)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        lines = text.splitlines()
        start = next(
            (index for index, line in enumerate(lines) if re.match(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", line)),
            None,
        )
        if start is None:
            return text.strip()
        base_indent = len(lines[start]) - len(lines[start].lstrip(" \t"))
        end = len(lines)
        for index in range(start + 1, len(lines)):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip(" \t")) <= base_indent:
                end = index
                break
        return textwrap.dedent("\n".join(lines[start:end])).rstrip()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return _python_node_source(text.splitlines(), node)
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == entry_point:
                    return _python_node_source(text.splitlines(), method)
    return text.strip()


def extract_target_java_code(text: str, entry_point: str, model_family: str) -> str:
    text = trim_trailing_prose(slice_from_code_start(text, "java", entry_point), model_family)
    for match in JAVA_METHOD.finditer(text):
        if match.group("name") != entry_point:
            continue
        open_index = text.find("{", match.end() - 1)
        close_index = matching_brace_index(text, open_index)
        if close_index is not None:
            return text[match.start():close_index + 1].strip()
    return text.strip()


def pick_target_fenced_code(text: str, language: str | None, entry_point: str) -> str | None:
    for block in FENCED_CODE.finditer(text):
        if re.search(rf"\b{re.escape(entry_point)}\s*\(", block.group("code")):
            return block.group("code")
    return pick_fenced_code(text, language)


def clean_candidate_for_task(
    text: str, language: str | None, model_family: str, entry_point: str | None,
) -> str:
    if not entry_point or language not in {"python", "java"}:
        return clean_candidate(text, language, model_family)

    original_text = text
    text = normalize_text(strip_harmony_wrappers(text))
    if UNLABELED_ANALYSIS.search(text):
        sliced = slice_from_code_start(text, language)
        if sliced == text:
            return fallback_candidate(original_text, language)
        text = sliced
    if language == "python":
        target_is_already_present = re.search(
            rf"(?m)^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", text,
        ) is not None
    else:
        target_is_already_present = any(
            match.group("name") == entry_point for match in JAVA_METHOD.finditer(text)
        )
    fenced = None if target_is_already_present else pick_target_fenced_code(text, language, entry_point)
    if fenced is not None:
        text = fenced
    text = strip_inline_code_quotes(normalize_text(text))
    text = strip_bracket_tags(text)
    if not target_is_already_present:
        text = strip_leftover_fences(text)
    if language == "python":
        cleaned = extract_target_python_code(text, entry_point, model_family)
    else:
        cleaned = extract_target_java_code(text, entry_point, model_family)
    return keep_nonempty(cleaned, original_text, language)


def entry_point_from_input(task_input: str, language: str) -> str | None:
    if language == "python":
        try:
            tree = ast.parse(task_input)
        except SyntaxError:
            return None
        function = _first_python_function(tree)
        return function.name if function is not None else None
    if language == "java":
        match = next(iter(JAVA_METHOD.finditer(task_input)), None)
        return match.group("name") if match is not None else None
    return None


def load_legacy_entry_points(language: str) -> dict[str, str]:
    """Read a locally exported task table from the frozen CoderEval image, if present."""
    legacy_path = REPO_ROOT / f"codereval_legacy_{language}_tasks.json"
    if not legacy_path.is_file():
        return {}
    try:
        payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        records = payload.get("RECORDS", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("expected a record list or a RECORDS field")
        return {
            str(record["_id"]): str(record["name"])
            for record in records
            if isinstance(record, dict) and record.get("_id") and record.get("name")
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Ignoring unreadable legacy CodeREval {language} task mapping: {error}")
        return {}


@lru_cache(maxsize=None)
def load_codereval_entry_points(language: str | None) -> dict[str, str]:
    if language not in {"python", "java"}:
        return {}
    entry_points = load_legacy_entry_points(language)
    try:
        from datasets import DownloadConfig, load_dataset

        dataset = load_dataset(
            f"vitaleantonio/codereval-{language}",
            split="train",
            download_config=DownloadConfig(local_files_only=True),
        )
        entry_points.update({
            str(record["id"]): entry_point
            for record in dataset
            if (entry_point := entry_point_from_input(str(record.get("input", "")), language)) is not None
        })
    except Exception as error:
        if not entry_points:
            print(f"No local CodeREval {language} task mapping available: {error}")
    return entry_points


def predictions_root(experiment_root: Path) -> Path:
    if experiment_root.name == "predictions":
        return experiment_root
    direct_predictions = experiment_root / "predictions"
    if direct_predictions.is_dir():
        return direct_predictions
    return experiment_root


def clean_prediction_file(input_path: Path) -> tuple[Path, int] | None:
    model_family = model_family_from_path(input_path)
    language = language_from_path(input_path)
    entry_points = load_codereval_entry_points(language)
    records = []
    candidates_processed = 0
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                record = json.loads(line)
                candidates = record.get("generate_results", record.get("raw_generation", []))
                if not isinstance(candidates, list) or not all(isinstance(code, str) for code in candidates):
                    raise ValueError(f"Line {line_number} must contain a list of string generate_results.")
                task_id = record.get("_id", record.get("id"))
                records.append({
                    "_id": task_id,
                    "generate_results": [
                        clean_candidate_for_task(candidate, language, model_family, entry_points.get(str(task_id)))
                        for candidate in candidates
                    ],
                })
                candidates_processed += len(candidates)
    except json.JSONDecodeError as error:
        print(f"Skipped incomplete JSONL: {input_path} ({error})")
        return None

    output_path = input_path.with_name("predictions_cleaned.jsonl")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=input_path.parent, prefix=".cleaning-", suffix=".jsonl") as output_file:
        temporary_path = Path(output_file.name)
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, output_path)
    return output_path, candidates_processed


def clean_predictions(
    root: Path,
    exclude_models: set[str] | None = None,
    include_models: set[str] | None = None,
    languages: set[str] | None = None,
) -> tuple[int, int]:
    files_processed = 0
    candidates_processed = 0
    excluded = exclude_models or set()
    included = include_models or set()
    selected_languages = languages or set()
    if excluded and included:
        raise ValueError("--include-model and --exclude-model cannot be used together")
    for input_path in root.rglob("predictions.jsonl"):
        try:
            run_info = parse_run_name(input_path.parent.name)
        except ValueError as error:
            print(f"Skipped unsupported prediction run: {input_path} ({error})")
            continue
        if run_info.model in excluded:
            print(f"Skipped excluded model: {input_path}")
            continue
        if included and run_info.model not in included:
            continue
        if selected_languages and run_info.language not in selected_languages:
            continue
        result = clean_prediction_file(input_path)
        if result is None:
            continue
        output_path, candidate_count = result
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
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Exact parsed model name to skip; may be repeated (e.g. gpt-20b).",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Exact parsed model name to process; may be repeated (e.g. gpt-20b).",
    )
    parser.add_argument("--language", choices=("java", "python"), action="append", dest="languages")
    args = parser.parse_args()
    root = predictions_root(args.experiment_root).resolve()
    if not root.is_dir():
        parser.error(f"Predictions directory does not exist: {root}")

    if args.exclude_model and args.include_model:
        parser.error("--include-model and --exclude-model cannot be used together")
    file_count, candidate_count = clean_predictions(
        root, set(args.exclude_model), set(args.include_model), set(args.languages or [])
    )
    print(f"Cleaned {candidate_count} candidate(s) in {file_count} prediction file(s).")


if __name__ == "__main__":
    main()
