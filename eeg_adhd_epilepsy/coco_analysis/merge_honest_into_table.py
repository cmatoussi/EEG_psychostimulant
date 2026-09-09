"""Add the honest (out-of-fold calibrated) balanced accuracy into an existing
embedding_performance results table. The table already carries:
  balanced_accuracy_mean          -> default (0.5 threshold, no tuning)
  balanced_accuracy_optimal_mean  -> oracle  (threshold picked on the test fold)
This adds:
  balanced_accuracy_calibrated_mean -> honest (threshold from the OTHER folds)
so default / honest / oracle sit side-by-side. Idempotent (overwrites the column).
"""
from __future__ import annotations
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--honest", default="/home/mat/scratch/results/embedding_performance/baseline/allrun/honest_vs_oracle.csv")
    args = ap.parse_args()

    tab = pd.read_csv(args.table)
    h = pd.read_csv(args.honest)[["model", "condition", "aggregation", "head", "ba_honest"]]
    h = h.rename(columns={"model": "fm_model", "ba_honest": "balanced_accuracy_calibrated_mean"})
    tab["condition"] = tab["condition"].astype(str)
    merged = tab.merge(h, on=["fm_model", "condition", "aggregation", "head"], how="left")
    # place the honest column right after the oracle column for readability
    cols = list(merged.columns)
    if "balanced_accuracy_optimal_std" in cols:
        cols.remove("balanced_accuracy_calibrated_mean")
        i = cols.index("balanced_accuracy_optimal_std") + 1
        cols.insert(i, "balanced_accuracy_calibrated_mean")
        merged = merged[cols]
    merged.to_csv(args.table, index=False)
    n = merged["balanced_accuracy_calibrated_mean"].notna().sum()
    print(f"--> {args.table}: added honest column to {n}/{len(merged)} rows", flush=True)


if __name__ == "__main__":
    main()
