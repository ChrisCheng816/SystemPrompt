import random
import time
from pathlib import Path
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
def evaluate_generation(
    model_name,
    style,
    example_num=None,
    test_num=None,
    max_length=256,
    shuffled=False,
    system_prompt=None,
    dataset_generation=None,
    datatype=None,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.94,
    batch_size=4,
    temperature=0.0,
    pass_at=1,
    also_save_pass_at_1=False,
    pass_at_1_output_root=None,
    retriever_device="cuda:0",
    output_root="experiments_results_codereval/pass@1_t0",
    prompt_index=None,
    reservation=None,
    retriever_reservation=None,
):
    source, prompt_input, output, lang, saving_name = generation_data_selector(datatype)
    output_model_name = model_map.get(model_name, model_name)
    base_prompt = ""
    print_info(model_name, style, example_num, system_prompt, language = lang, direction = None)
    tokenizer, model, batch_size = load_model(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        batch_size=batch_size,
        reservation=reservation,
    )
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
                "target_code": dataset_generation["train"][i]["output"]  # Join tokens into a single string
            }
            for i in range(min(100000, len(dataset_generation["train"])))
        ]
        print("Example database constructed.")
        # query_code_arr = []
        # for i in range(0, length, 500):
        #     batch = test_data[source][i:i+500]
        #     query_code_arr.extend(batch)
            
        print("Starting to integrate example database...")
        base_prompt, top_k_sims_list = get_retrieval_prompt(
            test_data[source],
            example_db,
            example_num,
            device=retriever_device,
            reservation=retriever_reservation,
        )
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
    prompts = load_prompt_gen(
        len(test_data[src_key]),
        task_description,
        src_key,
        test_data,
        base_prompt,
        tokenizer,
        system_prompt,
        max_length,
        model=model_name,
    )
    prompts = random.sample(prompts, len(prompts)) if shuffled == True else prompts
    start_time = time.time()
    include_token_metadata = is_gpt_oss_model_name(model_name)
    generation_result = compute_metric_gen(
        prompts,
        batch_size,
        tokenizer,
        model,
        max_length,
        temperature=temperature,
        num_candidates=pass_at,
        include_token_metadata=include_token_metadata,
    )
    if include_token_metadata:
        predictions, token_metadata = generation_result
    else:
        predictions = generation_result
        token_metadata = None
    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))

    outputs_to_save = [(pass_at, output_root, predictions, token_metadata)]
    prediction_paths = []
    if also_save_pass_at_1:
        if pass_at_1_output_root is None:
            raise ValueError("pass_at_1_output_root is required when also_save_pass_at_1 is enabled.")
        first_candidates = [item[0] if isinstance(item, list) else item for item in predictions]
        first_token_metadata = None
        if token_metadata is not None:
            first_token_metadata = [
                {
                    **item,
                    "tokenized_results": item["tokenized_results"][:1],
                }
                for item in token_metadata
            ]
        outputs_to_save.append((1, pass_at_1_output_root, first_candidates, first_token_metadata))

    for saved_pass_at, saved_output_root, saved_predictions, saved_token_metadata in outputs_to_save:
        run_path = Path(saved_output_root) / "predictions" / f"{output_model_name}_{lang}_{style}_{example_num}-shot"
        if datatype == 0 or datatype == 1:
            filepath = evaluate_metric_mceval(
                predictions=saved_predictions,
                path=str(run_path),
                test_data=test_data,
                token_metadata=saved_token_metadata,
                run_index=prompt_index,
            )
        else:
            filepath = evaluate_metric_gen2(
                predictions=saved_predictions,
                path=str(run_path),
                test_data=test_data,
                lang=lang,
                token_metadata=saved_token_metadata,
                run_index=prompt_index,
            )
            prediction_paths.append(Path(filepath) / "predictions.jsonl")

        save_result_gen(
            filepath,
            model_name,
            lang,
            style,
            example_num,
            counter,
            elapsed_time,
            system_prompt,
            temperature,
            saved_pass_at,
            prompt_index=prompt_index,
        )
    del train_data, test_data, base_prompt, prompts, predictions, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if reservation is not None:
        reservation.reserve()
    if retriever_reservation is not None:
        retriever_reservation.reserve()
    return prediction_paths
