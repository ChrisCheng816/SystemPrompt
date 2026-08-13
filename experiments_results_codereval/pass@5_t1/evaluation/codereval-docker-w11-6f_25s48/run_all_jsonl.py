
import os
import subprocess

root_dir = "./predictions"
results_root = "./predictions/Results"

jobs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    jsonl_path1 = os.path.join(dirpath, "predictions.jsonl")
    jsonl_path2 = os.path.join(dirpath, "predictions_cleaned.jsonl")
    if os.path.isfile(jsonl_path2):
        jsonl_path = jsonl_path2
    elif os.path.isfile(jsonl_path1):
        jsonl_path = jsonl_path1
    else:
        continue

    rel_path = os.path.relpath(dirpath, root_dir)
    jobs.append((rel_path, jsonl_path))

print(f"[CoderEval] queued {len(jobs)} prediction file(s)", flush=True)
failures = []
for index, (rel_path, jsonl_path) in enumerate(sorted(jobs), start=1):
    result_dir = os.path.join(results_root, rel_path)
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"{rel_path}.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    language = "python" if "python" in jsonl_path else "java"
    print(f"[CoderEval {index}/{len(jobs)}] {language}: {rel_path}", flush=True)
    with open(output_file, "w", encoding="utf-8") as output:
        if language == "python":
            result = subprocess.run(["python", "PythonExec.py", jsonl_path, "5"], stdout=output, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(["python", "JavaExec.py", f"../{jsonl_path}", "5"], stdout=output, stderr=subprocess.STDOUT, cwd="./java")
    print(f"[CoderEval {index}/{len(jobs)}] exit={result.returncode}: {rel_path}", flush=True)
    if result.returncode != 0:
        failures.append((rel_path, output_file, result.returncode))

if failures:
    print(f"[CoderEval] {len(failures)} executor failure(s); preserving logs below.", flush=True)
    for rel_path, output_file, return_code in failures:
        print(f"[CoderEval] executor failure exit={return_code}: {rel_path}", flush=True)
        with open(output_file, encoding="utf-8", errors="replace") as failure_log:
            print(failure_log.read(), end="", flush=True)
    raise SystemExit(1)
