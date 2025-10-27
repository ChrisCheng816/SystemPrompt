import os
import re
import csv
import json
from collections import defaultdict

root_dir = "./"

# Store the final result
# { (model, task, method, shot) : [ {prompt, prompt_len, bleu, em, codebleu} ] }
results = defaultdict(list)
Mapping = {"0":40, "1":185, "2":294, "3":480, "4": 638}
# Collect all prompt lengths
all_prompt_lengths = set()
count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    count = count +1 
    output_path = ""
    for f in filenames:
        if f.endswith(".txt"):
            target_file = f
            output_path = os.path.join(dirpath, target_file)
            break
    if output_path == "":
        continue
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(output_path)))
    # Analyze folder names
    parts = parent_folder.rsplit("_", 4)
    if len(parts) < 4:
        continue

    model, task, method, shot, index = parts[0], parts[1], parts[2], parts[3], parts[4]

    task_key = f"{model}_{task}_{method}_{shot}"
    
    target_value = None
    with open(output_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    for i, line in enumerate(lines):
        if line.strip() == "finish_overall":
            if i >= 2:  # Two lines above
                try:
                    target_value = float(lines[i - 2]) * 100
                    break
                except ValueError:
                    print(f"Warning: cannot convert line to float -> {lines[i - 2]}")
            else:
                print("Error: not enough lines above 'finish_overall'")
            break

    cur_result = {"prompt": index, "prompt_len": Mapping[index]}

    if target_value is not None:
        print("Extracted value:", target_value)
        cur_result["pass@1"] = target_value

    # task_key needs to be determined based on the directory or other means
    results[task_key].append(cur_result)

    all_prompt_lengths.add(Mapping[index])

# Sort all lengths
all_prompt_lengths = sorted(all_prompt_lengths)
print(all_prompt_lengths)
# Export CSV
csv_path = "overall_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Dynamically generate header
    header = ["model", "task", "method", "shot"]
    for metric in ["pass@1"]:
        header.append(metric)
        for plen in all_prompt_lengths:
            header.append(f"{metric}_{plen}")
    writer.writerow(header)

    for task_key, result_list in sorted(results.items()):
        model, task, method, shot = task_key.split("_")
        row = [model, task, method, shot]

        # Sort by prompt_len and BLEU score
        sorted_list = sorted(
            result_list,
            key=lambda x: (x["prompt_len"], x.get("pass@1", 0)),
            reverse=True
        )

        # Deduplication: For entries of the same length, retain only the one with the highest BLEU score.
        unique_by_len = {}
        for res in sorted_list:
            plen = res["prompt_len"]
            if plen not in unique_by_len:
                unique_by_len[plen] = res

        # Output according to the global header
        for metric in ["pass@1"]:
            row.append(metric)
            for plen in all_prompt_lengths:
                res = unique_by_len.get(plen, None)
                if res is None or metric not in res:
                    row.append("")
                else:
                    row.append(f"{res[metric]:.2f}")


        writer.writerow(row)

json_path = "output.json"
all_records = []
seen_prompts = set()

# Deduplication and collection of all records
for result_list in results.values():
    for res in result_list:
        sp = res["prompt"]
        if sp in seen_prompts:
            continue
        seen_prompts.add(sp)
        all_records.append({
            "system_prompt": sp,
            "prompt_len": res["prompt_len"]
        })

# Sort by prompt_len
all_records.sort(key=lambda x: x["prompt_len"])

# Output to file, each record on a new line in JSON format
with open(json_path, "w", encoding="utf-8") as jf:
    for record in all_records:
        json.dump(record, jf, ensure_ascii=False)
        jf.write("\n")

print("code generation results have been saved to output_summ.csv and output_summ_simple.json")