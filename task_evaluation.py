import random
import time
from common_methods import *
from datasets import load_dataset
import thought_chain as t
from datasets import DatasetDict
from datetime import timedelta
from model_map import model_map
from generate_prompts import generate_translation_prompt, generate_summarization_prompt, generate_generation_prompt

GREEN = "\033[92m"
RESET = "\033[0m"

# ---------------------
# Importing data sets and metrics
# ---------------------
dataset_translation = load_dataset("google/code_x_glue_cc_code_to_code_trans")

# ---------------------
# Code summarization
# ---------------------

def evaluate_summarization(model_name, style, example_num = None, test_num = None, max_length=256, shuffled = False, system_prompt = None, dataset_summarization = None):
    base_prompt = ""
    language = dataset_summarization['test']['language'][0] if isinstance(dataset_summarization, DatasetDict) else dataset_summarization['language'][0]
    print_info(model_name, style, example_num, system_prompt, language = language, direction = None)
    tokenizer, model, batch_size = load_model(model_name)
    train_data = dataset_summarization["train"].select(range(example_num)) if example_num else dataset_summarization["train"]
    test_data = dataset_summarization["test"].select(range(test_num)) if test_num else dataset_summarization["test"]
    length = len(test_data["code_tokens"])
    print(length)
    # Build shared prompt using training examples
    counter = [0]
    if style == "naive" or style == "cot":
        base_prompt = train_data.map(lambda e: generate_summarization_prompt(e, style, counter, t.summarization[language]), load_from_cache_file=False)["prompt"]
        pre_prompt = "The following are a few examples for code summarization.\n" if style == "naive" else "The following are a few examples with thought steps for code summarization.\n"
        base_prompt = "".join(base_prompt)
        base_prompt = pre_prompt + base_prompt
        print(f"Counter = {counter[0]}")
    elif style == "retrieval":
        if check_prompt(example_num, language, task = "summarization") == False:
            example_db = [
                {
                    "source_code": smart_join(dataset_summarization["train"][i]["code_tokens"]),  # Join tokens into a single string
                    "target_code": smart_join(dataset_summarization["train"][i]["docstring_tokens"])  # Join tokens into a single string
                }
                for i in range(100000)
            ]
            print("Example database constructed.")
            query_code_arr = []
            for i in range(0, length, 500):
                batch = test_data["code_tokens"][i:i+500]
                query_code_arr.extend(smart_join(tokens) for tokens in batch)
            # query_code_arr = [smart_join(test_data["code_tokens"][i]) for i in range(length)]
            print("Starting to integrate example database...")
            base_prompt, top_k_sims_list = get_retrieval_prompt(query_code_arr, example_db, example_num)
            pre_prompt = "The following are a few retrieval-based examples for code summarization.\n"
            base_prompt = [pre_prompt + prompt for prompt in base_prompt]
            print("Retrieval data integration completed")
            save_prompt(example_num, language, task = "summarization", prompts = base_prompt, sims = top_k_sims_list)
        else:
            base_prompt = extract_prompt(example_num, language, task = "summarization")

    after_description = "Let's think step-by-step to understand this method first, as shown in the example(s) if provided. Please do not output your thought steps if exist, just output the answer directly ###\n" if style == "cot" else "Please output the answer directly as shown in the examples if provided.###\n"
    task_description = f"### It is your turn now! Summarizing the follwing code into summary. {after_description}"

    if system_prompt == None:
        system_prompt = "You are a software documentation assistant. Given a code snippet, your task is to generate a concise and informative natural language summary that describes the purpose and behavior of the code."
    src_key, tgt_key = "code_tokens", "docstring_tokens"

    predictions= []
    print(f"Loading {len(test_data[src_key])} prompts...")
    prompts, references = load_prompt(len(test_data[src_key]), task_description, src_key, tgt_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length)
    print("Prompts loaded successfully")
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    print("Starting to generate...")
    start_time = time.time()
    predictions = compute_metric(prompts, batch_size, tokenizer, model, references, max_length)
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    filepath, result, bleu_result = evaluate_metric_sum(predictions=predictions, references=references, path=f"summarization_results/{model_map[model_name]}_{language}_{style}_{example_num}-shot")
    print("---------BLEU RESULT------------")
    print(f"Normal {result}, HF: {bleu_result}")

    save_result_sum(filepath, model_name, language, style, example_num, counter, elapsed_time, system_prompt, result, bleu_result)
    # with open(f"{filepath}/output.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{model_name} Language:{dataset_summarization['test']['language'][0]} {style} {example_num}-shot counter={counter[0]} Time: {elapsed_time}:\n{system_prompt}\n\nBleu: {result}\nHFBleu: {bleu_result}\n")
    del train_data, test_data, base_prompt, prompts, references, predictions, model, tokenizer, result
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ---------------------
# Code translation
# ---------------------

def evaluate_translation(model_name, style, example_num = None, test_num = None, max_length=256, shuffled = False, system_prompt = None, order = None):
    language = "java" if order == 0 else "cs"
    lang = "c_sharp" if order == 0 else "java"
    direction = "Java2Cs" if order == 0 else "Cs2Java"
    print_info(model_name, style, example_num, system_prompt, direction = direction)
    tokenizer, model, batch_size = load_model(model_name)
    length = len(dataset_translation["train"])
    print(length)
    train_data = dataset_translation["train"].select(range(example_num)) if example_num else dataset_translation["train"]
    test_data = dataset_translation["test"].select(range(test_num)) if test_num else dataset_translation["test"]
    counter = [0]
    # Build shared prompt using training examples
    if style == "naive" or style == "cot":
        base_prompt = train_data.map(lambda e: generate_translation_prompt(e, style, counter, t.translation[language], order))["prompt"]
        pre_prompt = "The following are a few example(s) for code translation.\n" if style == "naive" else "The following are a few example(s) with thought steps for code translation.\n"
        base_prompt = "".join(base_prompt)
        base_prompt = pre_prompt + base_prompt
        print(f"Counter = {counter[0]}")
    elif style == "retrieval":
        if check_prompt(example_num, language, task="translation") == False:
            if order == 0:
                example_db = [
                    {
                        "source_code": dataset_translation["train"][i]["java"],
                        "target_code": dataset_translation["train"][i]["cs"]
                    }
                    for i in range(length)
                ]
                query_code_arr = [test_data["java"][i] for i in range(len(test_data["java"]))]
                base_prompt, top_k_sims_list = get_retrieval_prompt(query_code_arr, example_db, example_num)
                pre_prompt = "The following are a few retrieval-based example(s) for code translation."
                base_prompt = [pre_prompt + prompt for prompt in base_prompt]
            else:
                example_db = [
                    {
                        "source_code": dataset_translation["train"][i]["cs"],
                        "target_code": dataset_translation["train"][i]["java"]
                    }
                    for i in range(length)
                ]
                query_code_arr = [test_data["cs"][i] for i in range(len(test_data["cs"]))]
                base_prompt, top_k_sims_list = get_retrieval_prompt(query_code_arr, example_db, example_num)
                pre_prompt = "The following are a few retrieval-based example(s) for code translation."
                base_prompt = [pre_prompt + prompt for prompt in base_prompt]
            save_prompt(example_num, language, task = "translation", prompts = base_prompt, sims = top_k_sims_list)
        else:
            base_prompt = extract_prompt(example_num, language, task = "translation")
    else:
        base_prompt = ""

    after_description = "Let's think step-by-step to understand this translation first, as shown in the example(s) if provided. Please do not output your thought steps if exist, just output the answer directly." if style == "cot" else "Please output the answer directly as shown in the examples if provided."
    task_description = f"### It is your turn now! {after_description} Translate the following Java code into Csharp code.\n" if order == 0 else f"### It is your turn now! {after_description} Translate the following Csharp code into Java code.\n"
    if system_prompt == None:
        system_prompt = "You are a professional code translator trained to convert source code between programming languages while preserving the original behavior and semantics."
    
    src_key = "java" if order == 0 else "cs"
    tgt_key = "cs" if order == 0 else "java"

    predictions= []
    print(f"Loading {len(test_data[src_key])} prompts...")
    prompts, references = load_prompt(len(test_data[src_key]), task_description, src_key, tgt_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length)
    print(f"{len(prompts)} Prompts loaded successfully")
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    print("Starting to generate...")
    start_time = time.time()
    predictions = compute_metric_tran(prompts, batch_size, tokenizer, model, references, max_length, lang)
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    filepath, BLEU_Smooth, EM, CodeBleu = evaluate_metric_tran(predictions=predictions, references=references, path=f"translation_results/{model_map[model_name]}_{direction}_{style}_{example_num}-shot", lang=lang)
    print("----------BLEU RESULT------------")
    print(BLEU_Smooth)

    save_result_trans(filepath, model_name, direction, style, example_num, counter, elapsed_time, system_prompt, BLEU_Smooth, EM, CodeBleu)

    del train_data, test_data, base_prompt, prompts, references, predictions, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ---------------------
# Code generation
# ---------------------
def evaluate_generation(model_name, style, example_num = None, test_num = None, max_length=256, shuffled = False, system_prompt = None, dataset_generation = None, datatype = None):
    source, prompt_input, output, lang, saving_name = generation_data_selector(datatype)
    base_prompt = ""
    print_info(model_name, style, example_num, system_prompt, language = lang, direction = None)
    tokenizer, model, batch_size = load_model(model_name)
    train_data = dataset_generation["train"].select(range(example_num)) if example_num else dataset_generation["train"]
    test_data = dataset_generation["test"].select(range(test_num)) if test_num else dataset_generation["test"]
    length = len(test_data)
    print(length)
    print(len(dataset_generation["train"]))
    # Build shared prompt using training examples
    counter = [0]
    if style == "naive" or style == "cot":
        base_prompt = train_data.map(lambda e: generate_generation_prompt(e, style, counter), load_from_cache_file=False)["prompt"]
        pre_prompt = "The following are a few examples for code generation.\n" if style == "naive" else "The following are a few examples with thought steps for code generation.\n"
        base_prompt = "".join(base_prompt)
        base_prompt = pre_prompt + base_prompt
        print(f"Counter = {counter[0]}")
    elif style == "retrieval":
        example_db = [
            {
                "source_code": dataset_generation["train"][i][source],  # Join tokens into a single string
                "target_code": dataset_generation["train"][i][output]  # Join tokens into a single string
            }
            for i in range(100000)
        ]
        print("Example database constructed.")
        query_code_arr = []
        for i in range(0, length, 500):
            batch = test_data[source][i:i+500]
            query_code_arr.extend(batch)
            
        print("Starting to integrate example database...")
        base_prompt, top_k_sims_list = get_retrieval_prompt(query_code_arr, example_db, example_num)
        pre_prompt = "The following are a few retrieval-based examples for code generation.\n"
        base_prompt = [pre_prompt + prompt for prompt in base_prompt]
        print("Retrieval data integration completed")
        save_prompt(example_num, lang, task = "generation", prompts = base_prompt, sims = top_k_sims_list)

    after_description = "Let's think step-by-step to understand this method first, as shown in the example(s) if provided. Please do not output your thought steps if exist, just output the answer directly ###\n" if style == "cot" else "Please output the complete method directly as shown in the examples if provided.###\n"
    task_description = f"### It is your turn now! Generating the code based on the instruction provided. {after_description}"
    src_key = source

    predictions= []
    print(f"Loading {len(test_data[src_key])} prompts...")
    prompts = load_prompt_gen(len(test_data[src_key]), task_description, src_key, test_data, base_prompt, tokenizer, system_prompt, model, max_length)
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    start_time = time.time()
    predictions = compute_metric_gen(prompts, batch_size, tokenizer, model, max_length)
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    if datatype == 0 or datatype == 1:
        filepath, result = evaluate_metric_gen1(predictions=predictions, path=f"generation_results_mceval/{model_map[model_name]}_{lang}_{style}_{example_num}-shot", saving_name = saving_name, lang=lang)
    else:
        filepath = evaluate_metric_gen2(predictions=predictions, path=f"generation_results_codereval/predictions/{model_map[model_name]}_{lang}_{style}_{example_num}-shot", test_data = test_data, lang=lang)

    save_result_gen(filepath, model_name, lang, style, example_num, counter, elapsed_time, system_prompt)
    del train_data, test_data, base_prompt, prompts, predictions, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    