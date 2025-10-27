import evaluate
from datasets import load_dataset
from task_evaluation import evaluate_generation
from Prompts.gen_prompts import gen_prompts as GEN
from Tools.codereval import codereval_java, codereval_python

# ---------------------
# Importing data sets and metrics
# ---------------------

if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "openai/gpt-oss-20b",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    ]

    methods = ["zero", "naive", "retrieval"]
    example_nums = [0, 3]
    max_lengths = {0: 1024, 1: 2048, 3: 8196}
    orders = [0, 1]
    indexs = [0,1,2,3,4]

    for model in models:
        for method in methods:
            for example_num in example_nums:
                if method == "zero" and example_num != 0:
                    continue
                if method == "retrieval" and example_num == 0:
                    continue
                if method == "naive" and example_num == 0:
                    continue
                if method == "cot" and example_num == 0:
                    continue
                for index in indexs:
                    evaluate_generation(model, method, example_num=example_num, max_length=max_lengths[example_num], system_prompt = GEN[index], dataset_generation = codereval_java, datatype = 2)
                    evaluate_generation(model, method, example_num=example_num, max_length=max_lengths[example_num], system_prompt = GEN[index], dataset_generation = codereval_python, datatype = 3)
