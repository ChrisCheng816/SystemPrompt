import random
import time
from common_methods import *
from datasets import load_dataset
from datasets import DatasetDict
from datetime import timedelta
from model_map import model_map
from generate_prompts import generate_generation_prompt

GREEN = "\033[92m"
RESET = "\033[0m"

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
        # query_code_arr = []
        # for i in range(0, length, 500):
        #     batch = test_data[source][i:i+500]
        #     query_code_arr.extend(batch)
            
        print("Starting to integrate example database...")
        base_prompt, top_k_sims_list = get_retrieval_prompt(test_data[source], example_db, example_num)
        pre_prompt = "The following are a few retrieval-based examples for code generation.\n"
        base_prompt = [pre_prompt + prompt for prompt in base_prompt]
        print("Retrieval data integration completed")
        save_prompt(example_num, lang, task = "generation", prompts = base_prompt, sims = top_k_sims_list)

    # x = "Let's think step-by-step to understand this method first, as shown in the example(s) if provided. Please do not output your thought steps if exist, just output the answer directly ###\n"
    
    after_description = "Please only output the complete method directly as shown in the examples if provided. Do NOT include any import statements or class declarations. Do not wrap the output in markdown code fences or quote blocks, output raw code only.###\n"
    task_description = f"### It is your turn now! Generating the code based on the instruction provided. {after_description}"
    src_key = source

    predictions= []
    print(f"Loading {len(test_data[src_key])} prompts...")
    prompts = load_prompt_gen(len(test_data[src_key]), task_description, src_key, test_data, base_prompt, tokenizer, system_prompt, max_length)
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    start_time = time.time()
    predictions = compute_metric_gen(prompts, batch_size, tokenizer, model, max_length, model_name)
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