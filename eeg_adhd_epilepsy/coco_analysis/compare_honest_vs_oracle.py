"""Honest (calibrated) vs oracle balanced accuracy for the embedding-extraction +
classical-head results, for ALL models -- computed from the already-saved per-fold
predictions, no model re-run.

- oracle  (= existing balanced_accuracy_optimal): threshold chosen on the SAME
  test fold that is being scored (optimistic).
- honest  (out-of-fold calibrated): for each fold, choose the balanced-accuracy-
  maximizing threshold on the POOLED OTHER folds, then apply it to the held-out
  fold. The threshold never sees the fold it scores -> honest.

Level per aggregation matches the table's convention:
  averaged_predictions -> subject_level (aggregate proba by subject first)
  averaged_epochs      -> epoch_level  (one sample per subject already)

Reads:  <runs_dir>/{model}_{cond}_{aggregation}.json
Writes: honest_vs_oracle.csv  (model, condition, aggregation, head, roc_auc,
        ba_oracle, ba_honest, gap)
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

_GRID = np.linspace(0.01, 0.99, 197)


def _agg_subject(y, p, g):
    u = np.unique(g)
    sy = np.array([int(round(float(y[g == k].mean()))) for k in u])
    sp = np.array([float(p[g == k].mean()) for k in u])
    return sy, sp


def _best_thr(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    return float(_GRID[np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in _GRID])])


def _fold_arrays(fold, level):
    y = np.asarray(fold["y_true"])
    pr = np.asarray(fold["y_proba"])
    p = pr[:, 1] if pr.ndim == 2 else pr
    if level == "subject_level":
        g = np.asarray(fold.get("group"))
        if g is None or g.size != y.size:
            return None
        y, p = _agg_subject(y, p, g)
    return y, p


def _metrics_for_head(folds, level):
    fa = [_fold_arrays(f, level) for f in folds]
    fa = [x for x in fa if x is not None and len(np.unique(x[0])) > 1]
    if len(fa) < 2:
        return None
    aucs, default, oracle, honest = [], [], [], []
    for i, (yi, pi) in enumerate(fa):
        aucs.append(roc_auc_score(yi, pi))
        default.append(balanced_accuracy_score(yi, (pi >= 0.5).astype(int)))  # no threshold tuning
        oracle.append(balanced_accuracy_score(yi, (pi >= _best_thr(yi, pi)).astype(int)))
        oy = np.concatenate([fa[j][0] for j in range(len(fa)) if j != i])
        op = np.concatenate([fa[j][1] for j in range(len(fa)) if j != i])
        thr = _best_thr(oy, op)  # threshold from the OTHER folds only
        honest.append(balanced_accuracy_score(yi, (pi >= thr).astype(int)))
    return float(np.mean(aucs)), float(np.mean(default)), float(np.mean(oracle)), float(np.mean(honest))


_AGG_LEVEL = {"averaged_predictions": "subject_level", "averaged_epochs": "epoch_level"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/mat/scratch/results/embedding_performance/subject_level/runs/ALL")
    ap.add_argument("--extra-runs", nargs="*", default=[
        "/home/mat/scratch/results/embedding_performance/baseline/allrun/runs"],
        help="additional run dirs (e.g. moirai/neurolm baseline runs)")
    ap.add_argument("--out", default="/home/mat/scratch/results/embedding_performance/baseline/allrun/honest_vs_oracle.csv")
    args = ap.parse_args()

    files = []
    for d in [args.runs_dir, *args.extra_runs]:
        files += glob.glob(f"{d}/*_averaged_predictions.json") + glob.glob(f"{d}/*_averaged_epochs.json")
        # baseline runs use a _fast/_svm suffix per head-group
        files += glob.glob(f"{d}/*_averaged_predictions_*.json") + glob.glob(f"{d}/*_averaged_epochs_*.json")
    files = sorted(set(f for f in files if not f.endswith("_posthoc_metrics.json")))

    rows = []
    pat = re.compile(r"(?P<model>[a-z0-9]+)_(?P<cond>EO|EC)_(?P<agg>averaged_predictions|averaged_epochs)")
    for f in files:
        m = pat.search(Path(f).name)
        if not m:
            continue
        agg = m["agg"]; level = _AGG_LEVEL[agg]
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for head, node in data.get("results", {}).items():
            folds = node.get("predictions", [])
            res = _metrics_for_head(folds, level)
            if res is None:
                continue
            auc, default, oracle, honest = res
            rows.append({"model": m["model"], "condition": m["cond"], "aggregation": agg,
                         "head": head, "roc_auc": round(auc, 4),
                         "ba_default": round(default, 4), "ba_honest": round(honest, 4),
                         "ba_oracle": round(oracle, 4), "gap": round(oracle - honest, 4)})
            print(f"  {m['model']:9s} {m['cond']} {agg:20s} {head:7s}: auc={auc:.3f} "
                  f"ba_default={default:.3f} ba_honest={honest:.3f} ba_oracle={oracle:.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["model", "condition", "aggregation", "head"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n--> wrote {args.out}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
