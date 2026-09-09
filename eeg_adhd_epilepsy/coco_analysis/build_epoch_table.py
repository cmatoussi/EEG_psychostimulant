"""Build an EPOCH-level embedding-extraction + head table from saved per-fold
predictions -- no model re-fit. Epoch-level = score each epoch independently
(subject-grouped CV), no per-subject pooling. Only the `averaged_predictions`
runs carry true per-epoch predictions (the head was fit on epochs); the
`averaged_epochs` runs pre-average and are skipped here.

Emits roc_auc + balanced accuracy at three operating points:
  ba_default (0.5) / ba_honest (out-of-fold threshold) / ba_oracle (test-optimal),
matching the subject-level honest/oracle framing.

Usage:
  python build_epoch_table.py --runs-dir <dir with *_averaged_predictions.json>
                              --out <baseline/epoch/<cohort>/results_embedding_all.csv>
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd

from compare_honest_vs_oracle import _metrics_for_head

COLUMNS = ["fm_model", "condition", "aggregation", "head", "status", "n_windows",
           "roc_auc_mean", "balanced_accuracy_mean", "balanced_accuracy_calibrated_mean",
           "balanced_accuracy_optimal_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = []
    for d in args.runs_dir:
        files += glob.glob(f"{d}/**/*_averaged_predictions*.json", recursive=True)
    files = sorted(set(f for f in files if not f.endswith("_posthoc_metrics.json")))

    pat = re.compile(r"(?P<model>[a-z0-9]+)_(?P<cond>EO|EC)_averaged_predictions")
    rows = []
    for f in files:
        m = pat.search(Path(f).name)
        if not m:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for head, node in data.get("results", {}).items():
            folds = node.get("predictions", [])
            res = _metrics_for_head(folds, "epoch_level")   # <-- epoch level, no pooling
            if res is None:
                continue
            auc, default, oracle, honest = res
            n_win = sum(len(fl.get("y_true", [])) for fl in folds)
            rows.append({"fm_model": m["model"], "condition": m["cond"], "aggregation": "per_epoch",
                         "head": head, "status": "success", "n_windows": n_win,
                         "roc_auc_mean": round(auc, 4),
                         "balanced_accuracy_mean": round(default, 4),
                         "balanced_accuracy_calibrated_mean": round(honest, 4),
                         "balanced_accuracy_optimal_mean": round(oracle, 4)})
            print(f"  {m['model']:9s} {m['cond']} {head:7s}: auc={auc:.3f} "
                  f"ba_default={default:.3f} ba_honest={honest:.3f} ba_oracle={oracle:.3f}", flush=True)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(["fm_model", "condition", "head"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n--> wrote {args.out}  ({len(df)} rows, {df.fm_model.nunique()} models)", flush=True)


if __name__ == "__main__":
    main()
