import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

results_dir = Path("/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve")
output_csv = results_dir / "finetune_results_reve.csv"

# Find all JSON files
json_files = list(results_dir.glob("reve_metrics_fold*.json"))

print(f"Found {len(json_files)} JSON metric files.")

# Extract data
data = []
for jf in json_files:
    # Example filename: reve_metrics_fold1_EC_baseline.json
    m = re.match(r"reve_metrics_fold(\d+)_(.+)\.json", jf.name)
    if m:
        fold = int(m.group(1))
        condition = m.group(2)
        with open(jf, "r") as f:
            metrics = json.load(f)
        
        data.append({
            "Condition": condition,
            "Fold": fold,
            "Val_AUC": metrics.get("auc"),
            "Val_Acc": metrics.get("acc"),
            "Val_BAcc": metrics.get("bacc"),
            "Val_F1": metrics.get("f1")
        })

if not data:
    print("No data found to process.")
    exit()

df = pd.DataFrame(data)

# Sort strictly by condition and fold
df = df.sort_values(by=["Condition", "Fold"])

# Calculate Average and StdDev per condition
final_rows = []
for condition, group in df.groupby("Condition"):
    # Add regular fold rows
    for _, row in group.iterrows():
        final_rows.append({
            "Fold": row["Fold"],
            "Condition": row["Condition"],
            "Val_AUC": row["Val_AUC"],
            "Val_Acc": row["Val_Acc"],
            "Val_BAcc": row["Val_BAcc"],
            "Val_F1": row["Val_F1"]
        })
    
    # Add Average row
    avg_row = {
        "Fold": "Average",
        "Condition": condition,
        "Val_AUC": group["Val_AUC"].mean(),
        "Val_Acc": group["Val_Acc"].mean(),
        "Val_BAcc": group["Val_BAcc"].mean(),
        "Val_F1": group["Val_F1"].mean()
    }
    final_rows.append(avg_row)
    
    # Add StdDev row
    std_row = {
        "Fold": "StdDev",
        "Condition": condition,
        "Val_AUC": group["Val_AUC"].std(ddof=0),
        "Val_Acc": group["Val_Acc"].std(ddof=0),
        "Val_BAcc": group["Val_BAcc"].std(ddof=0),
        "Val_F1": group["Val_F1"].std(ddof=0)
    }
    final_rows.append(std_row)

final_df = pd.DataFrame(final_rows)

print(f"Writing rebuilt CSV to {output_csv}...")
final_df.to_csv(output_csv, index=False)
print("Done! Here is a preview:")
print(final_df.head(14))
