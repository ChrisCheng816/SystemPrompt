"""Command-line entry point for SystemPrompt code-generation experiments."""

from __future__ import annotations

import argparse
import os
import re

from runtime import GPUReservation, configure_cuda_visibility


DEFAULT_MODELS = [
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
]
FIXED_METHODS = ("zero", "naive", "retrieval")
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
        default="cuda:3",
        help="Retriever CUDA device index relative to --gpu-devices (default: cuda:3, the fourth selected GPU).",
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
        help="Experiment root. Default: experiments_results/pass@{pass_at}_t{temperature}.",
    )
    parser.add_argument(
        "--pass-at-1-output-root",
        default=None,
        help="Optional root for the extra pass@1 output produced by --also-save-pass-at-1.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=FIXED_METHODS,
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
    retriever_match = re.fullmatch(r"cuda:(\d+)", args.retriever_device)
    if retriever_match is None:
        parser.error("--retriever-device must use a visible CUDA index such as cuda:3.")
    if int(retriever_match.group(1)) >= len([d for d in args.gpu_devices.split(",") if d.strip()]):
        parser.error("--retriever-device must refer to one of the GPUs selected by --gpu-devices.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    if args.test_num is not None and args.test_num < 1:
        parser.error("--test-num must be positive.")

    return args


def main():
    args = parse_args()
    visible_devices = configure_cuda_visibility(args.gpu_devices)
    tensor_parallel_size = args.tensor_parallel_size or len(visible_devices)
    if tensor_parallel_size < 1 or tensor_parallel_size > len(visible_devices):
        raise ValueError("--tensor-parallel-size must be between 1 and the number of selected GPUs.")

    output_root = args.output_root or os.path.join(
        "experiments_results", f"pass@{args.pass_at}_t{args.temperature}"
    )
    pass_at_1_output_root = None
    if args.also_save_pass_at_1:
        pass_at_1_output_root = args.pass_at_1_output_root or os.path.join(
            "experiments_results", "pass@1_t1"
        )
    models = args.model_names or DEFAULT_MODELS

    # This must precede imports of task_evaluation/common_methods, which import torch and vLLM.
    reservation = GPUReservation(
        args.gpu_reserve_mb,
        len(visible_devices),
        args.gpu_reserve_free_mb,
    )
    reservation.reserve()

    from task_evaluation import evaluate_generation
    from Prompts.gen_prompts import gen_prompts
    from Tools.codereval import codereval_java, codereval_python

    methods = args.methods or FIXED_METHODS
    prompt_indices = args.prompt_indices or FIXED_PROMPT_INDICES
    datasets = {
        "java": (codereval_java, 2),
        "python": (codereval_python, 3),
    }

    try:
        for model_name in models:
            for method in methods:
                example_num = 0 if method == "zero" else 3
                max_length = MAX_LENGTHS[example_num]
                for prompt_index in prompt_indices:
                    for language in args.language:
                        dataset, datatype = datasets[language]
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
                            pass_at_1_output_root=pass_at_1_output_root,
                            retriever_device=args.retriever_device,
                            output_root=output_root,
                            reservation=reservation,
                        )
    finally:
        reservation.release()


if __name__ == "__main__":
    main()
