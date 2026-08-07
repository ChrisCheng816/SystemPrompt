"""Command-line entry point for SystemPrompt code-generation experiments."""

from __future__ import annotations

import argparse
import os
import re

from runtime import GPUReservation, configure_cuda_visibility, parse_gpu_devices


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "openai/gpt-oss-20b",
]
DEFAULT_METHODS = ("zero", "retrieval")
AVAILABLE_METHODS = ("zero", "naive", "retrieval")
FIXED_PROMPT_INDICES = range(5)
MAX_LENGTHS = {0: 1024, 3: 8192}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed SystemPrompt generation experiments with configurable runtime settings."
    )
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        metavar="HF_MODEL_ID",
        help="Model to run. Repeat the option for multiple models. Defaults to the three CodeLlama models.",
    )
    parser.add_argument(
        "--language",
        nargs="+",
        choices=("java", "python"),
        default=("java", "python"),
        help="Datasets to run (default: both).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("codereval", "mceval"),
        dest="datasets",
        help="Benchmark to run. Repeat to choose a subset; default: both CodeEval and McEval.",
    )
    parser.add_argument(
        "--gpu-devices",
        default="0,1,2,3",
        help="Physical GPU IDs exposed to this run, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor-parallel size (default: number of --gpu-devices).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per vLLM request batch.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.94,
        help="Fraction of each selected GPU vLLM may use (default: 0.94).",
    )
    parser.add_argument(
        "--gpu-reserve-mb",
        type=int,
        default=None,
        help="Fixed MiB temporarily reserved on every selected GPU; default fills currently free memory.",
    )
    parser.add_argument(
        "--gpu-reserve-free-mb",
        type=int,
        default=512,
        help="Free MiB left on every selected GPU when --gpu-reserve-mb is omitted (default: 512).",
    )
    parser.add_argument(
        "--retriever-device",
        default=None,
        help="Retriever CUDA device index relative to --gpu-devices; default: last selected vLLM GPU.",
    )
    parser.add_argument(
        "--retriever-gpu-device",
        default=None,
        help="Physical GPU ID dedicated to retrieval, outside --gpu-devices. It is appended after the vLLM GPUs.",
    )
    parser.add_argument("--temperature", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pass-at", type=int, choices=(1, 5), default=1)
    parser.add_argument(
        "--also-save-pass-at-1",
        action="store_true",
        help="With --temperature 1 --pass-at 5, also save each prompt's first candidate to pass@1_t1 without generating again.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional base root. With both benchmarks, _codereval and _mceval are appended.",
    )
    parser.add_argument(
        "--pass-at-1-output-root",
        default=None,
        help="Optional base root for extra pass@1 output; benchmark suffixes are added when both run.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=AVAILABLE_METHODS,
        dest="methods",
        help="Run only selected fixed methods. Repeat to choose multiple methods.",
    )
    parser.add_argument(
        "--prompt-index",
        action="append",
        type=int,
        choices=FIXED_PROMPT_INDICES,
        dest="prompt_indices",
        help="Run only selected system prompt indices (0-4). Repeat to choose multiple indices.",
    )
    parser.add_argument("--test-num", type=int, default=None, help="Optional limit on test examples per language.")
    args = parser.parse_args()

    if args.temperature == 0 and args.pass_at != 1:
        parser.error("--temperature 0 only supports --pass-at 1.")
    if args.also_save_pass_at_1 and (args.temperature != 1 or args.pass_at != 5):
        parser.error("--also-save-pass-at-1 requires --temperature 1 --pass-at 5.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.gpu_reserve_mb is not None and args.gpu_reserve_mb < 0:
        parser.error("--gpu-reserve-mb cannot be negative.")
    if args.gpu_reserve_free_mb < 0:
        parser.error("--gpu-reserve-free-mb cannot be negative.")
    try:
        args.vllm_devices = parse_gpu_devices(args.gpu_devices)
    except ValueError as error:
        parser.error(str(error))
    if args.retriever_gpu_device is not None:
        if not args.retriever_gpu_device.isdigit():
            parser.error("--retriever-gpu-device must be one physical GPU ID, such as 3.")
        if args.retriever_gpu_device in args.vllm_devices:
            parser.error("--retriever-gpu-device must be outside --gpu-devices.")
        if args.retriever_device is not None:
            parser.error("Use either --retriever-device or --retriever-gpu-device, not both.")
        args.visible_devices = args.vllm_devices + [args.retriever_gpu_device]
        args.resolved_retriever_device = f"cuda:{len(args.vllm_devices)}"
    else:
        args.visible_devices = args.vllm_devices
        args.resolved_retriever_device = args.retriever_device or f"cuda:{len(args.vllm_devices) - 1}"
        retriever_match = re.fullmatch(r"cuda:(\d+)", args.resolved_retriever_device)
        if retriever_match is None:
            parser.error("--retriever-device must use a visible CUDA index such as cuda:1.")
        if int(retriever_match.group(1)) >= len(args.vllm_devices):
            parser.error("--retriever-device must refer to one of the GPUs selected by --gpu-devices.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    if args.test_num is not None and args.test_num < 1:
        parser.error("--test-num must be positive.")
    return args


def main():
    args = parse_args()
    visible_devices = configure_cuda_visibility(",".join(args.visible_devices))
    tensor_parallel_size = args.tensor_parallel_size or len(args.vllm_devices)
    if tensor_parallel_size < 1 or tensor_parallel_size > len(args.vllm_devices):
        raise ValueError("--tensor-parallel-size must be between 1 and the number of selected GPUs.")

    models = args.model_names or DEFAULT_MODELS

    # This must precede imports of task_evaluation/common_methods, which import torch and vLLM.
    reservation = GPUReservation(
        args.gpu_reserve_mb,
        len(args.vllm_devices),
        args.gpu_reserve_free_mb,
        external_guard_dir=os.environ.get("SYSTEMPROMPT_GPU_GUARD_DIR"),
        external_guard_ids=tuple(args.vllm_devices),
    )
    reservation.reserve()
    retriever_reservation = None
    if args.retriever_gpu_device is not None:
        retriever_reservation = GPUReservation(
            args.gpu_reserve_mb,
            1,
            args.gpu_reserve_free_mb,
            device_offset=len(args.vllm_devices),
            external_guard_dir=os.environ.get("SYSTEMPROMPT_GPU_GUARD_DIR"),
            external_guard_ids=(args.retriever_gpu_device,),
        )
        retriever_reservation.reserve()

    from task_evaluation import evaluate_generation
    from Prompts.gen_prompts import gen_prompts
    methods = args.methods or DEFAULT_METHODS
    prompt_indices = args.prompt_indices or FIXED_PROMPT_INDICES
    selected_datasets = args.datasets or ("codereval", "mceval")
    default_roots = {
        "codereval": "experiments_results_codereval",
        "mceval": "experiments_results_mceval",
    }
    if args.output_root is None:
        output_roots = {
            name: os.path.join(default_roots[name], f"pass@{args.pass_at}_t{args.temperature}")
            for name in selected_datasets
        }
    elif len(selected_datasets) == 1:
        output_roots = {selected_datasets[0]: args.output_root}
    else:
        output_roots = {name: f"{args.output_root}_{name}" for name in selected_datasets}

    pass_at_1_output_roots = {}
    if args.also_save_pass_at_1:
        if args.pass_at_1_output_root is None:
            pass_at_1_output_roots = {
                name: os.path.join(default_roots[name], "pass@1_t1")
                for name in selected_datasets
            }
        elif len(selected_datasets) == 1:
            pass_at_1_output_roots = {selected_datasets[0]: args.pass_at_1_output_root}
        else:
            pass_at_1_output_roots = {
                name: f"{args.pass_at_1_output_root}_{name}" for name in selected_datasets
            }
    dataset_loaders = {}
    if "codereval" in selected_datasets:
        from Tools.codereval import codereval_java, codereval_python

        dataset_loaders["codereval"] = {
            "java": (codereval_java, 2),
            "python": (codereval_python, 3),
        }
    if "mceval" in selected_datasets:
        from Tools.mceval import load_mceval

        dataset_loaders["mceval"] = {
            language: (load_mceval(language), 0 if language == "java" else 1)
            for language in args.language
        }

    try:
        for model_name in models:
            for method in methods:
                example_num = 0 if method == "zero" else 3
                max_length = MAX_LENGTHS[example_num]
                for prompt_index in prompt_indices:
                    for dataset_name in selected_datasets:
                        for language in args.language:
                            dataset, datatype = dataset_loaders[dataset_name][language]
                            evaluate_generation(
                                model_name,
                                method,
                                example_num=example_num,
                                test_num=args.test_num,
                                max_length=max_length,
                                system_prompt=gen_prompts[prompt_index],
                                dataset_generation=dataset,
                                datatype=datatype,
                                tensor_parallel_size=tensor_parallel_size,
                                gpu_memory_utilization=args.gpu_memory_utilization,
                                batch_size=args.batch_size,
                                temperature=float(args.temperature),
                                pass_at=args.pass_at,
                                also_save_pass_at_1=args.also_save_pass_at_1,
                                pass_at_1_output_root=pass_at_1_output_roots.get(dataset_name),
                                retriever_device=args.resolved_retriever_device,
                                output_root=output_roots[dataset_name],
                                prompt_index=prompt_index,
                                reservation=reservation,
                                retriever_reservation=retriever_reservation,
                            )
    finally:
        reservation.release()
        if retriever_reservation is not None:
            retriever_reservation.release()


if __name__ == "__main__":
    main()
