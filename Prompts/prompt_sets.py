"""The original and two wording-variant system-prompt sets."""

from __future__ import annotations

from Prompts.gen_prompts import gen_prompts


_A_BASE = (
    "You are an expert code generator. Your task is to produce an executable method "
    "from the given natural-language description."
)
_A_RULES = (
    "Follow exactly the function signature, parameter requirements, and output type "
    "given in the docstring or leading comments.",
    "Use suitable exception handling so invalid arguments and problems during execution are "
    "dealt with correctly.",
    "Reason internally, in order, about what the comments or docstring require of the "
    "implementation before you write the code.",
    "The implementation should address enough edge cases to pass all potential unit tests.",
)

_B_BASE = (
    "You are a highly skilled code-generation assistant. Your task is to create an "
    "executable method from the natural-language description provided."
)
_B_RULES = (
    "The function signature, parameter requirements, and output type specified in the "
    "docstring or leading comments must be followed strictly.",
    "The implementation must handle potential invalid inputs and runtime issues using "
    "appropriate exception-handling behavior.",
    "Carefully reason step by step about the method purpose stated in the docstring or "
    "leading comments before writing code, while keeping that reasoning private.",
    "The implementation must address enough edge cases to pass all potential unit tests.",
)


def _build_cumulative_prompts(base: str, rules: tuple[str, ...]) -> list[str]:
    prompts = [base]
    accumulated = base + "\nRules:"
    for number, rule in enumerate(rules, start=1):
        accumulated += f"\n{number}. {rule}"
        prompts.append(accumulated)
    return prompts


paraphrase_a_prompts = _build_cumulative_prompts(_A_BASE, _A_RULES)
paraphrase_b_prompts = _build_cumulative_prompts(_B_BASE, _B_RULES)

PROMPT_SETS = {
    "original": gen_prompts,
    "paraphrase_a": paraphrase_a_prompts,
    "paraphrase_b": paraphrase_b_prompts,
}


def get_prompt_set(name: str) -> list[str]:
    """Return one named five-prompt set."""
    try:
        return PROMPT_SETS[name]
    except KeyError as error:
        available = ", ".join(PROMPT_SETS)
        raise ValueError(f"Unknown prompt set {name!r}; choose from: {available}.") from error


def output_model_name_for_set(output_model_name: str, prompt_set: str) -> str:
    """Add a stable prompt-set suffix to an experiment model path."""
    get_prompt_set(prompt_set)
    return f"{output_model_name}-{prompt_set.replace('_', '-')}"
