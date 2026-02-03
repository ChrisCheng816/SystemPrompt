import os
import re
import csv
import json
from collections import defaultdict

root_dir = "./"

# 存放最终结果
# { (model, task, method, shot) : [ {prompt, prompt_len, bleu, em, codebleu} ] }
results = defaultdict(list)
Mapping = {"0":"BASE", "1":"STRUCT", "2":"ROBUST", "3":"REASONING", "4": "EDGE"}
# 收集全局所有出现过的长度
all_prompt_lengths = set()
count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "Results" in dirnames:
        dirnames.remove("Results")

    count = count +1 
    output_path = ""
    for f in filenames:
        if f == "predictions_cleaned.jsonl_out.jsonl":
            target_file = f
            output_path = os.path.join(dirpath, target_file)
            break
    if output_path == "":
        continue
    parent_folder = os.path.basename(os.path.dirname(output_path))
    # 解析文件夹名
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot, index = parts[0], parts[1], parts[2], parts[3], parts[4]

    task_key = f"{model}_{task}_{method}_{shot}"
    if task == "python":
        values = []
        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            print(output_path)
            for line in f:
                if line.strip():  # 跳过空行
                    obj = json.loads(line)
                    data.append(obj)

        count = 0
        for i, line in enumerate(data):
            is_pass = False
            for v in line["generate_results"]:
                if v["is_pass"] == True:
                    is_pass = True
            if is_pass == False:
                values.append(0)
            else:
                count=count+1
                values.append(1)
        print(count)
        
        cur_result = {"prompt": index, "values": values, "name": Mapping[index]}

        results[task_key].append(cur_result)

# 进行组队并写入CSV
output_dir = "./csv_results_python"
os.makedirs(output_dir, exist_ok=True)

for task_key, entries in results.items():
    # 转成字典方便查找
    index_to_values = {e["prompt"]: e["values"] for e in entries}

    # 只保留出现过的 index
    available_indices = [idx for idx in ["0", "1", "2", "3", "4"] if idx in index_to_values]
    if not available_indices:
        continue

    # 生成CSV文件路径
    csv_path = os.path.join(output_dir, f"{task_key}.csv")

    # 找出最短长度，防止不齐
    min_len = min(len(index_to_values[idx]) for idx in available_indices)

    # 写入CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # 表头：按 Mapping 顺序
        header = [Mapping[idx] for idx in available_indices]
        writer.writerow(header)

        # 每行依次写入各列的值
        for i in range(min_len):
            row = [index_to_values[idx][i] for idx in available_indices]
            writer.writerow(row)