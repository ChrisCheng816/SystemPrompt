import os
import subprocess

root_dir = "./predictions"  # 修改成你的根目录

results_root = "./predictions/Results"  # 输出根目录

for dirpath, dirnames, filenames in os.walk(root_dir):
    jsonl_path1 = dirpath + "/predictions.jsonl"
    jsonl_path2 = dirpath + "/predictions_cleaned.jsonl"
    jsonl_path = ""
    if os.path.isfile(jsonl_path2):
        jsonl_path = jsonl_path2
    elif os.path.isfile(jsonl_path1):
        jsonl_path = jsonl_path1
    else:
        continue

    if os.path.isfile(jsonl_path):
        # 构造结果保存路径
        save_dir = os.path.dirname(jsonl_path)
        save_name = os.path.basename(jsonl_path)
        save_path = os.path.join(save_dir, f"{save_name}_out.jsonl")
        with open(save_path, "w") as f:
            pass

        rel_path = os.path.relpath(dirpath, root_dir)
        result_dir = os.path.join(results_root, rel_path)
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n=== Processing {jsonl_path} ===")

        # 每次运行结果存储的文件名
        output_file = os.path.join(result_dir, f"{rel_path}.txt")
        
        if os.path.exists(output_file):
            print(f"Warning: {output_file} exists and will be overwritten.")
        else:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            if "python" in jsonl_path:
                subprocess.run(
                    ["python", "PythonExec.py", f"{jsonl_path}", "5"],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            else:
                subprocess.run(
                    ["python", "JavaExec.py", f"../{jsonl_path}", "5"],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd="./java"
                )

# zero0 shot0