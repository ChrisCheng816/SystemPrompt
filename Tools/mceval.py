"""Official McEval code-generation data for the supported languages."""

from datasets import DatasetDict, load_dataset


MCEVAL_BASE_URL = "https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval/resolve/main"
MCEVAL_INSTRUCT_URL = (
    "https://huggingface.co/datasets/Multilingual-Multimodal-NLP/"
    "McEval-Instruct/resolve/main/McEval-Instruct.json"
)
MCEVAL_GENERATION_FILES = {
    "java": "generation/Java.jsonl",
    "python": "generation/Python.jsonl",
}


def load_mceval(language: str) -> DatasetDict:
    """Load official McEval generation and McEval-Instruct examples by language."""
    try:
        data_file = MCEVAL_GENERATION_FILES[language]
    except KeyError as error:
        raise ValueError("McEval supports only java and python in this project.") from error

    test_data = load_dataset(
        "json", data_files=f"{MCEVAL_BASE_URL}/{data_file}", split="train"
    )
    train_data = load_dataset("json", data_files=MCEVAL_INSTRUCT_URL, split="train")
    train_data = train_data.filter(
        lambda example: example["language"].casefold() == language.casefold()
    )
    return DatasetDict({"train": train_data, "test": test_data})
