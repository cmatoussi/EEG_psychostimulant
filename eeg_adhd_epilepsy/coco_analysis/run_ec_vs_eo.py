"""EC-vs-EO decoding from pre-extracted FM embeddings + classical heads.

Same machinery as run_embedding_cohorts.py, but the TARGET is the recording
condition (EO=0, EC=1) instead of Epilepsy. For each (model, cohort) we stack a
subject's averaged EO and EC embeddings and predict the condition. Only the
'averaged_epochs' aggregation is used (one averaged vector per subject per
condition; metrics at that row level). CV is StratifiedGroupKFold grouped by
subject, so a subject's EO and EC rows never split across folds (no subject
leakage). f1 is WEIGHTED (average='weighted'), computed here from fold
predictions (the shared posthoc uses binary f1).

Usage:
    python run_ec_vs_eo.py --cohort-group sex --out-dir <dir>
    python run_ec_vs_eo.py --cohort-group age --cohort 9-12 --out-dir <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402

LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
N_SPLITS  = 5
MODELS    = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
# condition -> class label
CONDITIONS = [("EO_baseline", 0), ("EC_baseline", 1)]

HEADS = {
    "logreg": {"method": "LogisticRegression", "max_iter": 500, "class_weight": "balanced"},
    "rf":     {"method": "RandomForestClassifier", "n_estimators": 200, "class_weight": "balanced"},
    "svm":    {"method": "SVC", "kernel": "rbf", "probability": True, "class_weight": "balanced"},
    "histgb": {"method": "HistGradientBoostingClassifier", "class_weight": "balanced"},
    "dummy":  {"method": "DummyClassifier", "strategy": "stratified", "random_state": 42},
}

# cohort-group -> {cohort key: (csv name, mask fn on normalized label_df)}  (0-4 and with_asd excluded)
COHORT_GROUPS = {
    "sex": {
        "ALL": ("results_embedding_all.csv",    lambda d: pd.Series(True, index=d.index)),
        "F":   ("results_embedding_female.csv", lambda d: d.sex == "F"),
        "M":   ("results_embedding_male.csv",   lambda d: d.sex == "M"),
    },
    "age": {
        "5-8":   ("results_embedding_age_5_8.csv",   lambda d: d.age_group == "5-8"),
        "9-12":  ("results_embedding_age_9_12.csv",  lambda d: d.age_group == "9-12"),
        "13-18": ("results_embedding_age_13_18.csv", lambda d: d.age_group == "13-18"),
    },
    "comorbidity": {
        "no_comorbidity":    ("results_embedding_no_comorbidity.csv",    lambda d: (d.autism == 0) & (d.adhd == 0)),
        "with_adhd":         ("results_embedding_with_adhd.csv",         lambda d: (d.autism == 0) & (d.adhd == 1)),
        "with_adhd_and_asd": ("results_embedding_with_adhd_and_asd.csv", lambda d: (d.autism == 1) & (d.adhd == 1)),
    },
}

COLUMNS = [
    "fm_model", "target", "aggregation", "head", "status", "n_subjects", "n_windows",
    "n_eo", "n_ec",
    "accuracy_mean", "accuracy_std",
    "balanced_accuracy_mean", "balanced_accuracy_std",
    "balanced_accuracy_optimal_mean", "balanced_accuracy_optimal_std",
    "f1_mean", "f1_std",           # f1 = WEIGHTED
    "roc_auc_mean", "roc_auc_std",
]
_METRICS = ["accuracy", "balanced_accuracy", "balanced_accuracy_optimal", "f1", "roc_auc"]
AGG = "averaged_epochs"
LOAD_LEVEL = "subject"   # averaged_epochs -> per-subject-per-condition embeddings


def _empty_metrics():
    return {f"{m}_{s}": np.nan for m in _METRICS for s in ("mean", "std")}


def _score(yt, yp, p1):
    """Row-level metrics for one fold; f1 is WEIGHTED."""
    two = len(np.unique(yt)) > 1
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "balanced_accuracy_optimal": ra._balanced_accuracy_optimal(yt, p1) if two else float("nan"),
        "f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(yt, p1)) if two else float("nan"),
    }


def _metrics_from_result(result_json: Path, head: str):
    """Mean/std over folds of weighted metrics for one head, from saved predictions."""
    data = json.loads(result_json.read_text())
    node = data.get("results", {}).get(head, {})
    folds = []
    for fold in node.get("predictions", []):
        yt = np.asarray(fold["y_true"]); yp = np.asarray(fold["y_pred"])
        proba = np.asarray(fold["y_proba"]); p1 = proba[:, 1] if proba.ndim == 2 else proba
        folds.append(_score(yt, yp, p1))
    if not folds:
        return None
    out = {}
    for m in _METRICS:
        vals = [f[m] for f in folds]
        out[f"{m}_mean"] = round(float(np.nanmean(vals)), 4)
        out[f"{m}_std"] = round(float(np.nanstd(vals)), 4)
    return out


def _load_condition(model, cond, cohort_df):
    acfg = {"model_key": model, "target_col": "epilepsy", "embedding_level": LOAD_LEVEL}
    X, _, groups = ra.load_precomputed_embeddings(
        acfg, {"paths": {}}, cohort_df, {"conditions": [cond]}
    )
    return X, groups


def run_one_cohort(cohort_key, csv_name, mask_fn, label_df, out_dir):
    cohort_df = label_df[mask_fn(label_df)].copy()
    run_root = Path(out_dir) / "runs" / cohort_key
    out_csv = Path(out_dir) / csv_name
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"=== cohort {cohort_key}: {cohort_df['study_id'].nunique()} subjects ===", flush=True)

    def _flush():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

    for model in MODELS:
        base = {"fm_model": model, "target": "EC_vs_EO", "aggregation": AGG}
        # stack EO + EC averaged-subject embeddings, label = condition
        try:
            Xs, ys, gs = [], [], []
            for cond, lab in CONDITIONS:
                Xc, gc = _load_condition(model, cond, cohort_df)
                Xs.append(Xc); ys.append(np.full(len(Xc), lab, dtype=int)); gs.append(gc)
            X = np.vstack(Xs); y = np.concatenate(ys); groups = np.concatenate(gs)
            n_eo = int((y == 0).sum()); n_ec = int((y == 1).sum())
        except FileNotFoundError as e:
            print(f"  {model}: MISSING ({e})", flush=True)
            for h in HEADS:
                rows.append({**base, "head": h, "status": "missing_embeddings",
                             "n_subjects": 0, "n_windows": 0, "n_eo": 0, "n_ec": 0, **_empty_metrics()})
            _flush(); continue

        counts = {"n_subjects": len(np.unique(groups)), "n_windows": int(X.shape[0]),
                  "n_eo": n_eo, "n_ec": n_ec}
        if len(np.unique(y)) < 2 or len(np.unique(groups)) < N_SPLITS or min(n_eo, n_ec) < N_SPLITS:
            print(f"  {model}: SKIP (n_eo={n_eo}, n_ec={n_ec}, subj={counts['n_subjects']})", flush=True)
            for h in HEADS:
                rows.append({**base, "head": h, "status": "skipped_insufficient", **counts, **_empty_metrics()})
            _flush(); continue

        acfg = {
            "model_key": model, "models": HEADS,
            "cv": {"strategy": "stratified_group_kfold", "n_splits": N_SPLITS},
            "metrics": ["accuracy", "roc_auc", "balanced_accuracy", "f1"],
        }
        run_root.mkdir(parents=True, exist_ok=True)
        result_name = f"{model}_ec_vs_eo_{AGG}"
        try:
            ra.run_embed_head(acfg, X, y, groups, run_root, result_name=result_name)
        except Exception as e:  # noqa: BLE001
            print(f"  {model}: ERROR {e}", flush=True)
            for h in HEADS:
                rows.append({**base, "head": h, "status": "error", **counts, **_empty_metrics()})
            _flush(); continue

        result_json = run_root / f"{result_name}.json"
        for h in HEADS:
            mcols = _metrics_from_result(result_json, h)
            if mcols is None:
                rows.append({**base, "head": h, "status": "degenerate", **counts, **_empty_metrics()})
                continue
            status = "success" if mcols["roc_auc_mean"] == mcols["roc_auc_mean"] else "degenerate"
            rows.append({**base, "head": h, "status": status, **counts, **mcols})
            print(f"  {model}/{h}: roc_auc={mcols['roc_auc_mean']} "
                  f"bal_acc={mcols['balanced_accuracy_mean']} f1_w={mcols['f1_mean']} "
                  f"(EO={n_eo}, EC={n_ec})", flush=True)
        _flush()

    _flush()
    print(f"--> wrote {len(rows)} rows to {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-group", required=True, choices=list(COHORT_GROUPS))
    ap.add_argument("--cohort", default=None, help="single cohort key; omit for all in the group")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    cohorts = COHORT_GROUPS[args.cohort_group]
    keys = [args.cohort] if args.cohort else list(cohorts)
    for key in keys:
        if key not in cohorts:
            raise SystemExit(f"Unknown cohort {key!r}; options: {list(cohorts)}")
        csv_name, mask_fn = cohorts[key]
        run_one_cohort(key, csv_name, mask_fn, label_df, args.out_dir)


if __name__ == "__main__":
    main()
