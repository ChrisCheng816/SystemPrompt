import thought_chain as t
from common_methods import *

def generate_summarization_prompt(example, prompt_type, counter, steps):
    if counter is not None:
        counter[0] = counter[0] + 1
    if prompt_type == "naive":
        print(f"Summarization counter to {counter[0]} with naive prompt")
        prompt = f"### Example {counter[0]}:\nInput:\n{smart_join(example['code_tokens'])}\nOutput:\n{smart_join(example['docstring_tokens'])}\n\n"
    elif prompt_type == "cot":
        print(f"Summarization counter to {counter[0]} with cot prompt")
        print(steps[counter[0] - 1])
        prompt = (
            f"### Example {counter[0]}:\n Input:\n{smart_join(example['code_tokens'])}\n"
            f"Thought steps:\n{steps[counter[0] - 1]}\n"
            f"Output:\n{smart_join(example['docstring_tokens'])}\n\n"
        )
    else:
        raise ValueError("Unsupported prompt type.")
        
    return {"prompt":prompt}

def generate_generation_prompt(example, prompt_type, counter):
    if counter is not None:
        counter[0] = counter[0] + 1
    if prompt_type == "naive":
        prompt = f"### Example {counter[0]}:\nInput:\n{example['instruction'].strip()}\nOutput:\n{example['output'].strip()}\n\n"
    else:
        raise ValueError("Unsupported prompt type.")
        
    return {"prompt":prompt}

def generate_translation_prompt(example, prompt_type, counter, steps, order):
    cur = "java" if order == 0 else "cs"
    tar = "cs" if order == 0 else "java"
    if counter is not None:
        counter[0] = counter[0] + 1
    if prompt_type == "naive":
        prompt = f"### Example {counter[0]}:\nInput:\n{example[cur]}\nOutput:\n{example[tar]}\n\n"
    elif prompt_type == "cot":
        print(f"Translation counter to {counter[0]} with cot prompt")
        print(steps[counter[0] - 1])
        prompt = (
            f"### Example {counter[0]}:\n Input:\n{example[cur]}.\n"
            f"Thought steps:\n{steps[counter[0] - 1]}\n"
            f"Output:\n{example[tar]}\n\n"
        )
    else:
        raise ValueError("Unsupported prompt type.")
        
    return {"prompt":prompt}
    