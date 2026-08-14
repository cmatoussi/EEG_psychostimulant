#!/usr/bin/env python3
"""
Rebuilds finetune_results_cbramod_lp_ft.csv from the individual fold JSON files.
"""
import json
import glob
import pandas as pd
from pathlib import Path

JSON_DIR = Path("/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod/lp_ft")
CSV_OUT  = JSON_DIR / "finetune_results_cbramod_lp_ft.csv"

rows = []
for jf in sorted(JSON_DIR.glob("cbramod_metrics_fold*_lp_ft.json")):
    with open(jf) as f:
        data = json.load(f)

    m = data["metrics"]
    rows.append({
        "Fold":      f"Fold {data['fold']}",
        "Condition": data["condition"],
        "Val_AUC":   m["auc"],
        "Val_Acc":   m["acc"],
        "Val_BAcc":  m["bacc"],
        "Val_F1":    m["f1"],
    })

df = pd.DataFrame(rows).sort_values(["Condition", "Fold"]).reset_index(drop=True)

# Append per-condition Mean and Std summary rows
summary_rows = []
metrics_cols = ["Val_AUC", "Val_Acc", "Val_BAcc", "Val_F1"]
for condition, grp in df.groupby("Condition"):
    mean_vals = grp[metrics_cols].mean()
    std_vals  = grp[metrics_cols].std()
    summary_rows.append({"Fold": "Average", "Condition": condition, **mean_vals.to_dict()})
    summary_rows.append({"Fold": "Std",     "Condition": condition, **std_vals.to_dict()})

df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

df.to_csv(CSV_OUT, index=False)
print(f"Written {len(rows)} fold rows (+{len(summary_rows)} summary rows) → {CSV_OUT}")
print(df.to_string())
