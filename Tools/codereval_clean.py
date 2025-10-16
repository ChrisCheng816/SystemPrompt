import json
import os
import re

root_dir = "../generation_results_codereval/tmp"

def remove_assistant(code):
    if "assistantfinal" in code:
        return code.split("assistantfinal")[-1].strip()
    return code.strip()

def process_code(text: str) -> str:
    """
    Process code strings according to the new rules:
    - Remove leading ``` while preserving subsequent content
    - Remove any ``` appearing in the middle and truncate all content following it
    """
    result = text
    # Remove leading ```
    if result.startswith("```"):
        result = result[3:]

    # Find middle ```
    idx = result.find("```")
    if idx != -1:
        # Truncate ```
        result = result[:idx]
    
    return result.strip()

def extract_java_python_blocks(text: str) -> str:
    """
    Extract all ```java or ```python code block content from any position within the text and concatenate them in sequence.
    """
    # Match ```java or ```python at the beginning, until the next ``` ends
    pattern1 = r"```(java|python)\n(.*?)```"
    matches1 = re.findall(pattern1, text, re.DOTALL)
    pattern2 = r"```(java|python)\n(.*?)"
    matches2 = re.findall(pattern2, text, re.DOTALL)
    if matches1:
        return "".join(code for _, code in matches1)
    elif matches2:
        return "".join(code for _, code in matches2)
    else:
        return text


for dirpath, dirnames, filenames in os.walk(root_dir):
    if "predictions.jsonl" in filenames:
        jsonl_path = os.path.join(dirpath, "predictions.jsonl")
        lines = []
        output_path = os.path.join(dirpath, "predictions_cleaned.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Determine whether `generate_results` exists and is not empty.
                if "generate_results" in data and len(data["generate_results"]) > 0:
                    code = data["generate_results"][0]
                    # Remove leading ```java\n or ```python\n
                    if "openai" in dirpath:
                        print(f"{dirpath} has been cleaned. :-）")
                        code = remove_assistant(code)

                    code = extract_java_python_blocks(code)
                    code = process_code(code)
                    data["generate_results"][0] = code
                lines.append(data)

        # Write back to the JSONL file
        with open(output_path, "w", encoding="utf-8") as f:
            for data in lines:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "predictions_cleaned.jsonl" in filenames:
        jsonl_path = os.path.join(dirpath, "predictions_cleaned.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Determine whether `generate_results` exists and is not empty.
                if "generate_results" in data and len(data["generate_results"]) > 0:
                    code = data["generate_results"][0]
                    if "```" in code:
                        raise ValueError("The input text contains ``` which is not allowed.")
            print("no false")
