import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv("output_detail.csv")

# Columns to calculate standard deviation
prompt_cols = ["pass@1_40", "pass@1_185", "pass@1_294", "pass@1_480", "pass@1_638"]

# Calculate standard deviation (using sample standard deviation ddof=1)
df["std_pass@1"] = round(df[prompt_cols].std(axis=1, ddof=1), 2)

# Output results
print(df[["model", "task", "method", "shot", "std_pass@1"]])

# If you need to save to a new file
df.to_csv("with_std.csv", index=False)