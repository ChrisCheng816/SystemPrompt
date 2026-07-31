"""Dataset adapter for the Python and Java HumanEval-X code-generation tasks.

HumanEval-X has a test split only.  The few-shot/retrieval examples therefore
come from the same language-specific CodeXGLUE training data used by CodeEval;
none of the HumanEval-X test tasks are used as demonstrations.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset


HUMANEVAL_X_DATASETS = ("zai-org/humaneval-x", "THUDM/humaneval-x")
SUPPORT_DATASETS = {
    "java": ("google/code_x_glue_tc_text_to_code", "nl", "code"),
    "python": ("google/code_x_glue_tc_nl_code_search_adv", "docstring", "code"),
}


def load_humaneval_x_test(language: str) -> Dataset:
    """Load one HumanEval-X language config, with a legacy Hub fallback."""
    errors = []
    for dataset_name in HUMANEVAL_X_DATASETS:
        try:
            return load_dataset(dataset_name, language, split="test")
        except Exception as error:  # The older repository can be unavailable on newer Datasets.
            errors.append(f"{dataset_name}: {error}")
    details = "\n  ".join(errors)
    raise RuntimeError(
        f"Could not load HumanEval-X ({language}). Tried:\n  {details}"
    )


def _load_support_examples(language: str) -> Dataset:
    try:
        dataset_name, instruction_column, output_column = SUPPORT_DATASETS[language]
    except KeyError as error:
        raise ValueError("HumanEval-X supports only 'java' and 'python'.") from error

    train = load_dataset(dataset_name, split="train")
    return train.rename_columns(
        {instruction_column: "instruction", output_column: "output"}
    )


def load_humaneval_x(language: str) -> DatasetDict:
    """Return normalized ``train`` and ``test`` splits for the experiment runner.

    Test columns are retained so that ``test`` and ``example_test`` remain
    available to external functional-correctness evaluators.  ``task_id`` is
    renamed to the runner's standard ``id`` field.
    """
    language = language.lower()
    if language not in SUPPORT_DATASETS:
        raise ValueError("HumanEval-X supports only 'java' and 'python'.")

    train = _load_support_examples(language)
    test = load_humaneval_x_test(language).rename_columns(
        {
            "task_id": "id",
            "prompt": "instruction",
            "canonical_solution": "output",
        }
    )
    return DatasetDict({"train": train, "test": test})
