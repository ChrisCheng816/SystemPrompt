import re
import gc
import os
import shutil
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
MAX_OUTPUT_TOKENS = 4096
_HARMONY_ENCODING = None

try:
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        load_harmony_encoding,
    )
except ImportError:
    Conversation = None
    DeveloperContent = None
    HarmonyEncodingName = None
    Message = None
    Role = None
    SystemContent = None
    load_harmony_encoding = None

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

def is_gpt_oss_model_name(model_name):
    return "gpt-oss" in str(model_name).lower()

def render_gpt_oss_prompt(user_content, system_prompt):
    encoding = get_harmony_encoding()
    if encoding is None or Message is None:
        return None
    messages = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new())]
    if system_prompt:
        developer_content = DeveloperContent(instructions=system_prompt)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))
    messages.append(Message.from_role_and_content(Role.USER, user_content))
    tokens = encoding.render_conversation_for_completion(
        Conversation.from_messages(messages),
        Role.ASSISTANT,
    )
    return encoding.decode_utf8(tokens)

def render_chat_prompts(prompts, tokenizer, max_length, model_name=None):
    if is_gpt_oss_model_name(model_name) or is_gpt_oss_tokenizer(tokenizer):
        rendered_prompts = []
        for messages in prompts:
            system_prompt = ""
            user_content = ""
            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                elif message["role"] == "user":
                    user_content = message["content"]
            rendered_prompt = render_gpt_oss_prompt(user_content, system_prompt)
            if rendered_prompt is not None:
                rendered_prompts.append(rendered_prompt)
            else:
                rendered_prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        return truncate_prompts(rendered_prompts, tokenizer, max_length)

    full_prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    return truncate_prompts(full_prompt, tokenizer, max_length)

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

    full_prompts = render_chat_prompts(prompts, tokenizer, max_length, model)
    return full_prompts, references

def load_prompt_gen(length, task_description, src_key, test_data, base_prompt, tokenizer, system_prompt, max_length=4096, model=None):
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

    full_prompts = render_chat_prompts(prompts, tokenizer, max_length, model)
    return full_prompts

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

def normalize_candidate_list(item):
    return item if isinstance(item, list) else [item]

def save_tokenized_results(filepath, ids, token_metadata):
    if token_metadata is None:
        return

    with open(f"{filepath}/predictions_tokenized.jsonl", "w", encoding="utf-8") as f:
        for i, metadata in enumerate(token_metadata):
            record = {"_id": ids[i], **metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def prepare_generation_output_dir(path, run_index=None):
    if run_index is None:
        counter = 0
        while os.path.exists(f"{path}_{counter}"):
            counter += 1
        filepath = f"{path}_{counter}"
    else:
        filepath = f"{path}_{int(run_index)}"
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        elif os.path.exists(filepath):
            os.remove(filepath)
    os.makedirs(filepath, exist_ok=True)
    return filepath

def evaluate_metric_mceval(predictions, path, test_data, token_metadata=None, run_index=None):
    """Save McEval predictions using the same compact schema as CodeEval."""
    filepath = prepare_generation_output_dir(path, run_index)

    ids = test_data["task_id"]
    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for i, prediction in enumerate(predictions):
            candidates = normalize_candidate_list(prediction)
            record = {"_id": ids[i], "generate_results": candidates}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    save_tokenized_results(filepath, ids, token_metadata)

    return filepath

def evaluate_metric_gen2(predictions, path, test_data = None, lang=None, token_metadata=None, run_index=None):
    filepath = prepare_generation_output_dir(path, run_index)

    data = []
    ids = test_data["id"]

    with open(f"{filepath}/predictions.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            candidates = normalize_candidate_list(item)
            record = {"_id": ids[i], "generate_results": candidates}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    save_tokenized_results(filepath, ids, token_metadata)

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

def is_gpt_oss_tokenizer(tokenizer):
    model_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    return "gpt-oss" in model_name

def get_harmony_encoding():
    global _HARMONY_ENCODING
    if load_harmony_encoding is None:
        return None
    if _HARMONY_ENCODING is None:
        _HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _HARMONY_ENCODING

def get_stop_token_ids(tokenizer):
    if is_gpt_oss_tokenizer(tokenizer):
        encoding = get_harmony_encoding()
        if encoding is not None:
            return encoding.stop_tokens_for_assistant_actions()
    return [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

def message_channel_name(message):
    channel = getattr(message, "channel", None)
    if channel is None and hasattr(message, "to_dict"):
        channel = message.to_dict().get("channel")
    return getattr(channel, "value", channel)

def message_content_text(message):
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is None and hasattr(message, "to_dict"):
        content = message.to_dict().get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return ""

def extract_gpt_oss_final_text_from_tokens(token_ids):
    encoding = get_harmony_encoding()
    if encoding is None or not token_ids:
        return None
    messages = encoding.parse_messages_from_completion_tokens(
        token_ids,
        role=Role.ASSISTANT,
        strict=False,
    )
    final_parts = [
        message_content_text(message)
        for message in messages
        if message_channel_name(message) == "final"
    ]
    final_text = "".join(final_parts).strip()
    return final_text or None

def extract_gpt_oss_final_text_from_string(text):
    if "<|channel|>final<|message|>" not in text:
        return None
    final_text = text.rsplit("<|channel|>final<|message|>", 1)[-1]
    final_text = re.split(r"<\|(?:return|end|call)\|>", final_text, maxsplit=1)[0]
    return final_text.strip() or None

def looks_like_gpt_oss_reasoning(text):
    return re.match(r"^\s*analysis(?:\b|[A-Z])", text, flags=re.IGNORECASE) is not None

def json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def candidate_token_metadata(candidate, final_text):
    token_ids = list(getattr(candidate, "token_ids", []) or [])
    return {
        "text": getattr(candidate, "text", ""),
        "token_ids": token_ids,
        "final_text": final_text,
        "finish_reason": json_safe_value(getattr(candidate, "finish_reason", None)),
        "stop_reason": json_safe_value(getattr(candidate, "stop_reason", None)),
    }

def decode_candidate(candidate, tokenizer):
    text = candidate.text.strip()
    if not is_gpt_oss_tokenizer(tokenizer):
        return text
    token_ids = list(getattr(candidate, "token_ids", []) or [])
    try:
        final_text = extract_gpt_oss_final_text_from_tokens(token_ids)
    except Exception:
        final_text = None
    return (final_text or "").strip()

def decode_candidate_with_metadata(candidate, tokenizer):
    final_text = decode_candidate(candidate, tokenizer)
    return final_text, candidate_token_metadata(candidate, final_text)

def output_token_metadata(output, candidate_metadata):
    return {
        "prompt_token_ids": list(getattr(output, "prompt_token_ids", []) or []),
        "tokenized_results": normalize_candidate_list(candidate_metadata),
    }

def compute_metric(prompts, batch_size, tokenizer, model, references, max_length):
    predictions = []
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, tokenizer, model, max_length, MAX_OUTPUT_TOKENS)
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
        batch_predictions = run_batch(batch_prompts, tokenizer, model, max_length, MAX_OUTPUT_TOKENS)
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

def compute_metric_gen(prompts, batch_size, tokenizer, model, max_length, temperature, num_candidates, include_token_metadata=False):
    predictions = []
    token_metadata = [] if include_token_metadata else None
    counter = 1
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_result = run_batch(
            batch_prompts,
            tokenizer,
            model,
            max_length,
            MAX_OUTPUT_TOKENS,
            temperature=temperature,
            num_candidates=num_candidates,
            include_token_metadata=include_token_metadata,
        )
        if include_token_metadata:
            batch_predictions, batch_token_metadata = batch_result
            token_metadata.extend(batch_token_metadata)
        else:
            batch_predictions = batch_result
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1

        # Strong explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Starting to compute...")

    if include_token_metadata:
        return predictions, token_metadata
    return predictions

def run_batch(batch_prompts, tokenizer, model, input_max_len, output_max_tokens, temperature=0.0, num_candidates=1, include_token_metadata=False):
    ensure_pad_token(tokenizer)

    sampling_params = SamplingParams(
        max_tokens=output_max_tokens,
        temperature=temperature,
        n=num_candidates,
        stop_token_ids=get_stop_token_ids(tokenizer)
    )

    outputs = model.generate(batch_prompts, sampling_params)
    # Decode input and output to strings

    if num_candidates == 1:
        # Keep the historical return shape for translation and summarization callers.
        if include_token_metadata:
            decoded = [decode_candidate_with_metadata(output.outputs[0], tokenizer) for output in outputs]
            predictions = [item[0] for item in decoded]
            token_metadata = [
                output_token_metadata(output, item[1])
                for output, item in zip(outputs, decoded, strict=True)
            ]
        else:
            predictions = [decode_candidate(output.outputs[0], tokenizer) for output in outputs]
    else:
        if include_token_metadata:
            decoded = [
                [decode_candidate_with_metadata(candidate, tokenizer) for candidate in output.outputs]
                for output in outputs
            ]
            predictions = [[item[0] for item in output] for output in decoded]
            token_metadata = [
                output_token_metadata(output, [item[1] for item in output_decoded])
                for output, output_decoded in zip(outputs, decoded, strict=True)
            ]
        else:
            predictions = [
                [decode_candidate(candidate, tokenizer) for candidate in output.outputs]
                for output in outputs
            ]

    if include_token_metadata:
        return predictions, token_metadata
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

def save_result_gen(filepath, model_name, lang, style, example_num, counter, elapsed_time, system_prompt, temperature, pass_at, prompt_index=None):
    with open(f"{filepath}/output.json", "a", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "language": lang,
            "style": style,
            "example_num": example_num,
            "counter": counter[0],
            "prompt_index": prompt_index,
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
