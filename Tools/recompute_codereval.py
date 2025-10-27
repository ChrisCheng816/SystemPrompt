import re
import gc
import os
# import torch
# import scann
import json
import subprocess
# import numpy as np
# import torch.nn.functional as F
from codebleu import calc_codebleu
import evaluate
from datasets import load_dataset
from datasets import DatasetDict

bleu_metric = evaluate.load("bleu")
metric = evaluate.load("dvitel/codebleu")

def find_prediction_output_pairs(root_dir):
    pairs = [] 

    for dirpath, _, filenames in os.walk(root_dir):
        if "predictions_cleaned.jsonl" in filenames and "output.json" in filenames:
            predictions_path = os.path.join(dirpath, "predictions_cleaned.jsonl")
            output_path = os.path.join(dirpath, "output.json")
            pairs.append((predictions_path, output_path))

    return pairs

def append_to_outputs(output_path, BLEU_Smooth, CodeBleu):

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add new data items
    data["BLEU_Smooth"] = BLEU_Smooth
    data["CodeBleu"] = CodeBleu

    # Write back to file (preserve original data + new data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_lang_from_output(cleaned_path_jsonl, output_path):
    """
    Parse the first line of output.txt and return lang (lowercase)
    """
    pre_path =  f"{os.path.dirname(cleaned_path_jsonl)}/predictions.txt"
    predictions = []
    with open(cleaned_path_jsonl, "r", encoding="utf-8") as fin, open(pre_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            predictions.append(data["generate_results"][0].strip())
            text = data["generate_results"][0].strip().replace("\n", "")
            fout.write(text + "\n")

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        lang = data["language"]
    
    if lang == "java":
        dataset = load_dataset("vitaleantonio/codereval-java")["train"]
    else:
        dataset = load_dataset("vitaleantonio/codereval-python")["train"]

    txt_path = f"../generation_results_codereval/{lang}_references.txt"

    outputs = dataset["output"]  # Hugging Face datasets support direct column-wise retrieval
    references = [text.strip() for text in outputs]
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in dataset["output"]:
            f.write(line.strip().replace("\n", "") + "\n")

    return predictions, references, lang, txt_path, pre_path

def compute(pairs):
    for idx, (cleaned_path_jsonl, out_path) in enumerate(pairs, start=1):
        print(f"第 {idx} 个文件对") if idx % 10 == 0 else None
        predictions, references, lang, txt_path, pre_path = get_lang_from_output(cleaned_path_jsonl, out_path)
        try:
            cmd = f"python3 calc_code_bleu.py --refs {txt_path} --hyp {pre_path} --lang {lang} --params 0.25,0.25,0.25,0.25"
            codebleu = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            BLEU_Smooth = bleu_metric.compute(predictions=predictions, references=references, smooth=True)
            codebleu_score = calc_codebleu(references, predictions, lang)
            ngram_match, weighted_ngram_match = extract_value(codebleu.stdout)
            CodeBleu_result = ngram_match*25 + weighted_ngram_match*25 + codebleu_score["syntax_match_score"]*25 + codebleu_score["dataflow_match_score"]*25
            CodeBleu = {'codebleu': CodeBleu_result, 'ngram_match_score': ngram_match, 'weighted_ngram_match_score': weighted_ngram_match, 'syntax_match_score': codebleu_score["syntax_match_score"], 'dataflow_match_score': codebleu_score["dataflow_match_score"]}
        except Exception as e:
            print(f"Error occurred while running evaluation: {e}") 
        append_to_outputs(out_path, BLEU_Smooth, CodeBleu)

def extract_value(text):
    pattern = r"ngram match:\s*([0-9.]+),\s*weighted ngram match:\s*([0-9.]+)"

    match = re.search(pattern, text)
    if match:
        ngram_match = float(match.group(1))
        weighted_ngram_match = float(match.group(2))
        return ngram_match, weighted_ngram_match
    else:
        print("No match found")

if __name__ == "__main__":
    root_dir = "../generation_results_codereval/tmp"  # 修改这里
    pairs = find_prediction_output_pairs(root_dir)
    # cleaned_pairs = process_predictions(pairs)
    print(f"共找到 {len(pairs)} 对文件")
    compute(pairs)
    