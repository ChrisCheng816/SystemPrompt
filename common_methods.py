import re
import gc
import os
import torch
import json
import scann
import subprocess
import numpy as np
import logging
import torch.nn.functional as F
from codebleu import calc_codebleu
import evaluate
from exact_match import em_compute, exact_match_no_punct
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel

bleu_metric = evaluate.load("bleu")
em_metric = evaluate.load("exact_match")
# calc_codebleu = evaluate.load("dvitel/codebleu")
# ---------------------
# Public methods
# ---------------------
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm").propagate = False
logging.basicConfig(level=logging.ERROR)

MAX_MODEL_LEN = 10240
MAX_OUTPUT_TOKENS = 1024

def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

def clamp_max_input_length(max_length):
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        max_length = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS
    return max(1, min(max_length, MAX_MODEL_LEN - MAX_OUTPUT_TOKENS))

def truncate_prompts(prompts, tokenizer, max_length):
    max_length = clamp_max_input_length(max_length)
    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        tokenized = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    finally:
        tokenizer.truncation_side = old_truncation_side
    return tokenizer.batch_decode(tokenized["input_ids"], skip_special_tokens=False)

def load_model(model_name, tensor_parallel_size, gpu_memory_utilization, batch_size, reservation=None):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    ensure_pad_token(tokenizer)
    if reservation is not None:
        reservation.release()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return tokenizer, llm, batch_size

def load_prompt(length, task_description, src_key, tgt_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length=4096):
    src_data = list(test_data[src_key])
    tgt_data = list(test_data[tgt_key])
    references = []
    prompts = []
    counter = 0
    for idx in range(length):
        if tgt_key == "docstring_tokens":
            references.append(smart_join(tgt_data[idx]).strip())
            input_block = f"{task_description}Input:\n{smart_join(src_data[idx])}\nOutput:\n"
        else:
            references.append(tgt_data[idx].strip())
            input_block = f"{task_description}Input:\n{src_data[idx]}\nOutput:\n"

        counter += 1
        if counter % 500 == 0:
            print(f"Processing \033[1;32m{counter}\033[0m instances...")
        
        if isinstance(base_prompt, list):
            prompt = base_prompt[idx]
        else:
            prompt = base_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}{input_block}"}
        ]

        prompts.append(messages)

    full_prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    full_prompt = truncate_prompts(full_prompt, tokenizer, max_length)
    return full_prompt, references

def load_prompt_gen(length, task_description, src_key, test_data, base_prompt, tokenizer, system_prompt, max_length=4096):
    src_data = list(test_data[src_key])
    prompts = []
    counter = 0
    for idx in range(length):
        input_block = f"{task_description}Input:\n{src_data[idx]}\nOutput:\n"
        counter += 1
        if counter % 500 == 0:
            print(f"Processing \033[1;32m{counter}\033[0m instances...")
        
        if isinstance(base_prompt, list):
            prompt = base_prompt[idx]
        else:
            prompt = base_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}{input_block}"}
        ]

        prompts.append(messages)

    full_prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    full_prompt = truncate_prompts(full_prompt, tokenizer, max_length)
    return full_prompt

def evaluate_metric_sum(predictions, references, path):
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)
    with open(f"{filepath}/predictions.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            f.write(f"{i}\t{item}\n")
    
    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            record = {"id": i, "prediction": item}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(f"{filepath}/references.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(references):
            f.write(f"{i}\t{item}\n")
    try:
        cmd = f"python3 evaluator_ct/evaluator.py {filepath}/references.txt < {filepath}/predictions.txt"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"Error occurred while running evaluation: {e}")

    bleu_result = bleu_metric.compute(predictions=predictions, references=references, smooth=True)

    return filepath, result.stdout, bleu_result

def evaluate_metric_tran(predictions, references, path, lang):
    for i, item in enumerate(predictions):
        predictions[i] = clean_code_blocks(item)
        
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)
    with open(f"{filepath}/predictions.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            lines = [ln.strip() for ln in item.splitlines() if ln.strip()]
            s = "".join(lines)
            f.write(f"{s}\n")

    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            record = {"id": i, "prediction": item}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(f"{filepath}/references.txt", "w", encoding="utf-8") as f:
        for i, item in enumerate(references):
            f.write(f"{item}\n")
    try:
        cmd = f"python3 calc_code_bleu.py --refs ../{filepath}/references.txt --hyp ../{filepath}/predictions.txt --lang {lang} --params 0.25,0.25,0.25,0.25"
        codebleu = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="./Tools")
    except Exception as e:
        print(f"Error occurred while running evaluation: {e}")

    BLEU_Smooth = bleu_metric.compute(predictions=predictions, references=references, smooth=True)

    BLEU = bleu_metric.compute(predictions=predictions, references=references, smooth=False)
    print(f"BLEU: {BLEU['bleu']*100}")
    EM = em_compute(predictions, references)

    codebleu_score = calc_codebleu(references, predictions, lang)
    ngram_match, weighted_ngram_match = extract_value(codebleu.stdout)

    CodeBleu_result = ngram_match*25 + weighted_ngram_match*25 + codebleu_score["syntax_match_score"]*25 + codebleu_score["dataflow_match_score"]*25
    CodeBleu = {'codebleu': CodeBleu_result, 'ngram_match_score': ngram_match, 'weighted_ngram_match_score': weighted_ngram_match, 'syntax_match_score': codebleu_score["syntax_match_score"], 'dataflow_match_score': codebleu_score["dataflow_match_score"]}
    
    return filepath, BLEU_Smooth, EM, CodeBleu

def evaluate_metric_gen1(predictions, path, saving_name = None, lang=None):
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)

    data = []
    with open(f"Tools/{saving_name}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    for i, item in enumerate(data):
        item["raw_generation"] = [predictions[i]]

    with open(f"{filepath}/{lang}.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    try:
        cmd = f"python3 -u eval_all.py --result_path ../{filepath} --save_path ../{filepath}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="./eval")
    except Exception as e:
        print(f"Error occurred while running evaluation: {e}")
    print("bingo!!!!!!!!!!!")

    tmp_dir = os.path.join(os.path.dirname(filepath), "tmp")
    if os.path.isdir(tmp_dir):
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    return filepath, result.stdout

def evaluate_metric_mceval(predictions, path, test_data):
    """Save McEval records with their official fields and raw generations."""
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)

    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for example, prediction in zip(test_data, predictions, strict=True):
            record = dict(example)
            record["raw_generation"] = prediction if isinstance(prediction, list) else [prediction]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return filepath

def evaluate_metric_gen2(predictions, path, test_data = None, lang=None):
    counter = 0
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    filepath = f"{path}_{counter}"
    os.makedirs(filepath, exist_ok=True)

    data = []
    ids = test_data["id"]

    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            candidates = item if isinstance(item, list) else [item]
            record = {"_id": ids[i], "generate_results": candidates}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return filepath

def extract_value(text):
    pattern = r"ngram match:\s*([0-9.]+),\s*weighted ngram match:\s*([0-9.]+)"

    match = re.search(pattern, text)
    if match:
        ngram_match = float(match.group(1))
        weighted_ngram_match = float(match.group(2))
        return ngram_match, weighted_ngram_match
    else:
        print("No match found")

def compute_metric(prompts, batch_size, tokenizer, model, references, max_length):
    predictions = []
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, tokenizer, model, max_length, 512)
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1

        # Strong explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Starting to compute...")

    return predictions

def compute_metric_tran(prompts, batch_size, tokenizer, model, references, max_length, lang):
    predictions = []
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, tokenizer, model, max_length, 512)
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1
        # Strong explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    for i, item in enumerate(predictions):
        predictions[i] = [ln.strip() for ln in item.splitlines() if ln.strip()]
        predictions[i] = "".join(predictions[i])

    print("Starting to compute...")

    return predictions

def compute_metric_gen(prompts, batch_size, tokenizer, model, max_length, temperature, num_candidates):
    predictions = []
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = run_batch(
            batch_prompts,
            tokenizer,
            model,
            max_length,
            512,
            temperature=temperature,
            num_candidates=num_candidates,
        )
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1

        # Strong explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Starting to compute...")

    return predictions

def run_batch(batch_prompts, tokenizer, model, input_max_len, output_max_tokens, temperature=0.0, num_candidates=1):
    ensure_pad_token(tokenizer)

    sampling_params = SamplingParams(
        max_tokens=output_max_tokens,
        temperature=temperature,
        n=num_candidates,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )

    outputs = model.generate(batch_prompts, sampling_params)
    # Decode input and output to strings

    if num_candidates == 1:
        # Keep the historical return shape for translation and summarization callers.
        predictions = [output.outputs[0].text.strip() for output in outputs]
    else:
        predictions = [
            [candidate.text.strip() for candidate in output.outputs]
            for output in outputs
        ]

    return predictions

# Construct Prompt
def build_prompt(top_k_examples):
    prompt = ""
    counter = 0
    for ex in top_k_examples:
        counter = counter + 1
        prompt += f"### Example {counter}:\nInput:\n{ex['source_code'].strip()}\nOutput:\n{ex['target_code'].strip()}\n\n"
    return prompt
    
def get_retrieval_prompt(query_code_arr, example_db, k=3, device="cuda:0", reservation=None):
    # jinaai/jina-code-embeddings-1.5b
    if reservation is not None:
        reservation.release()
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device,
        # model_kwargs={"dtype": torch.bfloat16},
        # tokenizer_kwargs={"padding_side": "left"}
    )

    print("Retriever model loaded.")
    support_inputs = [item["source_code"] for item in example_db]
    query_inputs = query_code_arr

    support_embeddings = retriever.encode(support_inputs, normalize_embeddings=True)
    query_embeddings = retriever.encode(query_inputs, normalize_embeddings=True)

    print("Support set encoded.")
    similarity_matrix = retriever.similarity(query_embeddings, support_embeddings)
    print("Starting retrieval...")
    prompts = []
    top_k_sims_list = []
    for i, sim in enumerate(similarity_matrix):
        topk = torch.topk(sim, k=k)  #return values 和 indices
        top_k_idx = topk.indices.cpu().tolist()      # top-k index
        top_k_sims = topk.values.cpu().tolist()     # top-k sims

        retrieved_examples = [example_db[idx] for idx in top_k_idx]
        final_prompt = build_prompt(retrieved_examples)
        prompts.append(final_prompt)
        top_k_sims_list.append(top_k_sims)

    del retriever, support_embeddings, query_embeddings, similarity_matrix
    gc.collect()
    device_index = int(device.split(":", 1)[1])
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device_index)
    if reservation is not None:
        reservation.reserve()
    return prompts, top_k_sims_list

def print_info(model_name, style, example_num, system_prompt, language=None, direction=None):
    var = "Direction" if direction is not None else "Language"
    value = direction if direction is not None else language
    model_base = re.split(r"[\\/]", model_name)[-1]
    print(f"Model:{model_base} Style:{style} Example_Number:{example_num} {var}:{value}")
    print(f"\033[34m{system_prompt}\033[0m")

def smart_join(arr):
    result = ''
    for i, word in enumerate(arr):
        if word in {'.', ',', '?', '!', ';', ':'}:
            result = result.rstrip() + word  
        elif word == '(':
            result += word
        elif word == ')':
            result = result.rstrip() + word
        else:
            result += word + ' '
    return result.strip()

def clean_code_blocks(text: str) -> str:
    text = re.sub(r"^```(?:csharp|java)\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()

def check_prompt(number, language, task):
    prompts_file = f"Intermediate_output/{task}/{number}_{language}.jsonl"
    return os.path.exists(prompts_file)

def save_prompt(number, language, task, prompts, sims):
    # Save prompts and references to disk
    os.makedirs(f"Intermediate_output/{task}", exist_ok=True)
    with open(f"Intermediate_output/{task}/{number}_{language}.jsonl", "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    with open(f"Intermediate_output/{task}/{number}_{language}_sims.jsonl", "w", encoding="utf-8") as f:
        json.dump(sims, f, ensure_ascii=False, indent=2)

    print(f"Prompts saved for {language}.")

def extract_prompt(number, language, task, prompt=None):
    with open(f"Intermediate_output/{task}/{number}_{language}.jsonl", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    return prompts

def save_result_trans(filepath, model_name, direction, style, example_num, counter, elapsed_time, system_prompt, BLEU_Smooth, EM, CodeBleu):
    with open(f"{filepath}/output.json", "a", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "direction": direction,
            "style": style,
            "example_num": example_num,
            "counter": counter[0],
            "elapsed_time": elapsed_time,
            "system_prompt": system_prompt,
            "BLEU_Smooth": BLEU_Smooth,
            "EM": EM,
            "CodeBleu": CodeBleu
        }, f, ensure_ascii=False, indent=2)

def save_result_gen(filepath, model_name, lang, style, example_num, counter, elapsed_time, system_prompt, temperature, pass_at):
    with open(f"{filepath}/output.json", "a", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "language": lang,
            "style": style,
            "example_num": example_num,
            "counter": counter[0],
            "elapsed_time": elapsed_time,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "pass_at": pass_at,
        }, f, ensure_ascii=False, indent=2)

def save_result_sum(filepath, model_name, language, style, example_num, counter, elapsed_time, system_prompt, result, bleu_result):
    with open(f"{filepath}/output.json", "a", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "language": language,
            "style": style,
            "example_num": example_num,
            "counter": counter[0],
            "elapsed_time": elapsed_time,
            "system_prompt": system_prompt,
            "BLEU_Normal": result,
            "Bleu_HF": bleu_result
        }, f, ensure_ascii=False, indent=2)

def generation_data_selector(datatype):
    if datatype == 0:
        source = "instruction"
        prompt = "instruction"
        output = "canonical_solution"
        lang = "java"
        saving_name = "mceval_java"
    elif datatype == 1:
        source = "instruction"
        prompt = "instruction"
        output = "canonical_solution"
        lang = "python"
        saving_name = "mceval_python"
    elif datatype == 2:
        source = "instruction"
        prompt = "instruction"
        output = "output"
        lang = "java"
        saving_name = "codereval_java"
    elif datatype == 3:
        source = "instruction"
        prompt = "instruction"
        output = "output"
        lang = "python"
        saving_name = "codereval_python"
    else:
        raise ValueError("Unsupported datatype.")

    return source, prompt, output, lang, saving_name
