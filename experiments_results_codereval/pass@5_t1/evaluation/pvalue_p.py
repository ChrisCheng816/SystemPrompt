import os
import pandas as pd
import sys
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

all_results = []
root_dir = "./csv_results_python"

global_id = 0

for dirpath, dirnames, filenames in os.walk(root_dir):
    for f in filenames:
        if not f.endswith(".csv"):
            continue

        output_path = os.path.join(dirpath, f)
        parent_folder = os.path.splitext(f)[0]

        parts = parent_folder.rsplit("_", 3)
        if len(parts) != 4:
            continue

        model, task, method, shot = parts[0], parts[1], parts[2], parts[3]

        df = pd.read_csv(output_path)

        base_col = "BASE"
        if base_col not in df.columns:
            continue

        other_cols = [c for c in df.columns if c != base_col]
        if len(other_cols) == 0:
            continue

        pvals_raw = []
        row_indices = []

        for col in other_cols:
            a = ((df[base_col] == 1) & (df[col] == 1)).sum()
            b = ((df[base_col] == 1) & (df[col] == 0)).sum()
            c = ((df[base_col] == 0) & (df[col] == 1)).sum()
            d = ((df[base_col] == 0) & (df[col] == 0)).sum()

            table = [[a, b], [c, d]]
            result = mcnemar(table, exact=True, correction=False)

            pvals_raw.append(float(result.pvalue))

            or_smooth = (b + 1) / (c + 1)

            global_id += 1
            all_results.append({
                "ID": global_id,
                "Model": model,
                "Task": task,
                "Method": method,
                "Shot": shot,
                "Compare": f"{base_col} vs {col}",
                "OR": round(float(or_smooth), 3),
                "p_value": round(float(result.pvalue), 5)
            })
            row_indices.append(len(all_results) - 1)

        holm_adj = multipletests(pvals_raw, alpha=0.05, method="holm")[1]
        fdr_adj = multipletests(pvals_raw, alpha=0.05, method="fdr_bh")[1]

        for j, row_idx in enumerate(row_indices):
            all_results[row_idx]["p_value_holm"] = round(float(holm_adj[j]), 5)
            all_results[row_idx]["p_value_fdr_bh"] = round(float(fdr_adj[j]), 5)

output_csv = "mcnemar_results_holm_p.csv"
pd.DataFrame(all_results).to_csv(output_csv, index=False)