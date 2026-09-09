"""Baseline embedding-head performance for MOIRAI + NeuroLM, in the exact
`embedding_performance` table format (same COLUMNS / HEADS / aggregations / CV /
post-hoc metrics as run_embedding_cohorts.py), so the rows drop straight into the
comparison alongside luna/labram/biot/etc.

Difference from run_embedding_cohorts: the MOIRAI/NeuroLM extractors saved CSVs
with `study_id`+`epilepsy` columns (not the loader's `subject` schema), and only
at epoch level. So we load those CSVs directly and, for the `averaged_epochs`
aggregation, mean-pool the epoch embeddings per subject ourselves. Everything
downstream (`ra.run_embed_head` -> Experiment -> augment_posthoc) is identical.

Output: /home/mat/scratch/results/embedding_performance/baseline/allrun/
        results_embedding_all.csv   (fm_model in {moirai, neurolm})
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import run_analysis as ra
from run_embedding_cohorts import HEADS, COLUMNS, _METRICS, AGG, _empty_metrics, N_SPLITS

MODELS = ["moirai", "neurolm"]
CONDITIONS = ["EO_baseline", "EC_baseline"]
EMB_DIR = Path("/home/mat/scratch/results/extraceted_embeddings_files")
OUT_DIR = Path("/home/mat/scratch/results/embedding_performance/baseline/allrun")
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def _load_epoch(model, cond):
    """Our extractor CSV -> (X, y, groups) at epoch level."""
    f = EMB_DIR / model / f"{model}_{cond}_epoch_embeddings.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    df = pd.read_csv(f)
    cols = [c for c in df.columns if c.startswith("embedding_")]
    X = df[cols].to_numpy(np.float32)
    y = df["epilepsy"].to_numpy(int)
    groups = df["study_id"].astype(str).to_numpy()
    return X, y, groups


def _avg_by_subject(X, y, groups):
    u = np.unique(groups)
    Xs = np.stack([X[groups == g].mean(0) for g in u]).astype(np.float32)
    ys = np.array([int(y[groups == g][0]) for g in u])
    return Xs, ys, u


def main():
    label_df = ra.normalize_label_df(pd.read_csv(LABEL_CSV))
    label_df["study_id"] = label_df["study_id"].astype(str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_root = OUT_DIR / "runs"
    out_csv = OUT_DIR / "results_embedding_all.csv"
    rows = []

    def _flush():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

    for model in MODELS:
        for cond in CONDITIONS:
            cond_s = cond.replace("_baseline", "")
            try:
                Xep, yep, gep = _load_epoch(model, cond)
            except FileNotFoundError as e:
                print(f"  {model}/{cond_s}: MISSING {e}", flush=True)
                for agg in AGG:
                    for h in HEADS:
                        rows.append({"fm_model": model, "condition": cond_s, "aggregation": agg,
                                     "head": h, "status": "missing_embeddings", "n_subjects": 0,
                                     "n_windows": 0, "adhd": 0, "epilepsy": 0, "autism": 0,
                                     **_empty_metrics()})
                _flush(); continue

            for agg, spec in AGG.items():
                base = {"fm_model": model, "condition": cond_s, "aggregation": agg}
                metric_level = spec["metric_level"]
                # averaged_predictions: score epochs, aggregate by subject.
                # averaged_epochs: mean-pool embeddings per subject, then predict.
                if agg == "averaged_epochs":
                    X, y, groups = _avg_by_subject(Xep, yep, gep)
                else:
                    X, y, groups = Xep, yep, gep

                present = set(groups.tolist())
                sub = label_df[label_df["study_id"].isin(present)].drop_duplicates("study_id")
                counts = {"n_subjects": len(present), "n_windows": int(X.shape[0]),
                          "adhd": int((sub.adhd == 1).sum()),
                          "epilepsy": int((sub.epilepsy == 1).sum()),
                          "autism": int((sub.autism == 1).sum())}

                # NOTE: do NOT set cv.subject_level_metrics -- that makes the engine
                # pre-aggregate epochs to subjects before fitting, which breaks the
                # averaged_predictions semantics (fit on epochs, aggregate the
                # predictions post-hoc). Match run_embedding_cohorts exactly: fit at
                # the given level, and let augment_posthoc emit both epoch_level and
                # subject_level from the saved per-epoch predictions+groups.
                # Run the fast heads and the (epoch-level) O(n^2) RBF SVM as SEPARATE
                # flushed groups, so a slow/timed-out SVM never blocks the fast-head
                # rows from being written.
                run_root.mkdir(parents=True, exist_ok=True)
                head_groups = [("fast", {k: v for k, v in HEADS.items() if k != "svm"})]
                if "svm" in HEADS:
                    head_groups.append(("svm", {"svm": HEADS["svm"]}))
                for gname, gheads in head_groups:
                    acfg = {"model_key": model, "target_col": "epilepsy", "embedding_level": "epoch",
                            "models": gheads,
                            "cv": {"strategy": "stratified_group_kfold", "n_splits": N_SPLITS},
                            "metrics": ["accuracy", "roc_auc", "balanced_accuracy", "f1"]}
                    rname = f"{model}_{cond_s}_{agg}_{gname}"
                    try:
                        ra.run_embed_head(acfg, X, y, groups, run_root, result_name=rname)
                        posthoc = json.loads(
                            (run_root / f"{rname}_posthoc_metrics.json").read_text())["metrics"]
                    except Exception as e:  # noqa: BLE001
                        print(f"  {model}/{cond_s}/{agg}/{gname}: ERROR {e}", flush=True)
                        for h in gheads:
                            rows.append({**base, "head": h, "status": "error", **counts, **_empty_metrics()})
                        _flush(); continue

                    for h in gheads:
                        lvl = posthoc.get(h, {}).get(metric_level, {})
                        mcols = {}
                        for m in _METRICS:
                            mean = lvl.get(m, {}).get("mean", np.nan) if lvl else np.nan
                            std = lvl.get(m, {}).get("std", np.nan) if lvl else np.nan
                            mcols[f"{m}_mean"] = round(mean, 4) if mean == mean else np.nan
                            mcols[f"{m}_std"] = round(std, 4) if std == std else np.nan
                        status = "success" if mcols["roc_auc_mean"] == mcols["roc_auc_mean"] else "degenerate"
                        rows.append({**base, "head": h, "status": status, **counts, **mcols})
                        print(f"  {model}/{cond_s}/{agg}/{h}: roc_auc={mcols['roc_auc_mean']} "
                              f"bal_acc_opt={mcols['balanced_accuracy_optimal_mean']} "
                              f"(subj={counts['n_subjects']}, win={counts['n_windows']})", flush=True)
                    _flush()
    _flush()
    print(f"--> wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
