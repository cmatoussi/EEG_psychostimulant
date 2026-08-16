"""Coarse-to-fine (binary-search-style) grid search for the LogisticRegression
and RandomForest classification heads, restricted to the 'all' cohort at
subject level, for either pre-extracted FM embeddings or handcrafted sensor
features.

Each hyperparameter is searched in 3 rounds: round 1 scans the full bounds,
picks the best point, then round 2 re-grids a narrower window centered on it,
round 3 narrows further. RandomForest's two axes (n_estimators, max_depth) are
tuned sequentially (n_estimators first, then max_depth with n_estimators fixed
at its best value) rather than as a joint grid, keeping each axis a clean 1-D
coarse-to-fine search.

Usage:
    python run_hp_search.py --source embeddings --model reve --condition EO_baseline \
        --target-col epilepsy --out-dir <dir>
    python run_hp_search.py --source handcrafted --condition EO_baseline \
        --target-col epilepsy --out-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402
import pooled_metrics  # noqa: E402

N_SPLITS = 5
SEED = 42
LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
FEATDIR = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
           "signal_features/descriptors/combined")


def cv_score(clf_fn, X, y, g):
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_arrays = []
    for tr, te in sgkf.split(X, y, g):
        if len(np.unique(y[te])) < 2:
            continue
        clf = clf_fn()
        clf.fit(X[tr], y[tr])
        fold_arrays.append((y[te], clf.predict_proba(X[te])[:, 1]))
    if not fold_arrays:
        return float("nan")
    r = pooled_metrics.pooled(fold_arrays, calibrated=False)
    return r["roc_auc"]["mean"] if r else float("nan")


def refine_1d(score_fn, lo, hi, log_space, n_rounds=3, points=(7, 5, 5), is_int=False):
    """Coarse-to-fine 1-D search. Returns (best_value, best_score, history)."""
    history = []
    best_v, best_s = None, -np.inf
    cur_lo, cur_hi = lo, hi
    for rnd in range(n_rounds):
        n = points[min(rnd, len(points) - 1)]
        if log_space:
            grid = np.geomspace(cur_lo, cur_hi, n)
        else:
            grid = np.linspace(cur_lo, cur_hi, n)
        if is_int:
            grid = sorted(set(int(round(v)) for v in grid))
        else:
            grid = sorted(set(round(float(v), 6) for v in grid))
        for v in grid:
            s = score_fn(v)
            history.append({"round": rnd + 1, "value": v, "roc_auc": s})
            if s > best_s:
                best_s, best_v = s, v
        # narrow window around the best point found so far
        if log_space:
            half_span_log = (np.log10(hi) - np.log10(lo)) / (2 * (rnd + 2))
            cur_lo = 10 ** (np.log10(best_v) - half_span_log)
            cur_hi = 10 ** (np.log10(best_v) + half_span_log)
        else:
            half_span = (hi - lo) / (2 * (rnd + 2))
            cur_lo = max(lo, best_v - half_span)
            cur_hi = min(hi, best_v + half_span)
            if is_int and cur_hi - cur_lo < 1:
                cur_lo, cur_hi = max(lo, best_v - 1), min(hi, best_v + 1)
    return best_v, best_s, history


def search_lr(X, y, g):
    def score(c):
        return cv_score(lambda: make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, max_iter=1000, class_weight="balanced")),
                         X, y, g)
    best_c, best_s, hist = refine_1d(score, 1e-3, 1e3, log_space=True)
    return {"head": "logreg", "best_params": {"C": best_c}, "best_roc_auc": best_s, "history": hist}


def search_rf(X, y, g):
    def score_n(n_est, depth=None):
        return cv_score(lambda: RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, class_weight="balanced",
            random_state=SEED, n_jobs=4), X, y, g)

    best_n, best_s_n, hist_n = refine_1d(lambda v: score_n(v), 50, 2000,
                                         log_space=False, is_int=True)
    best_d, best_s_d, hist_d = refine_1d(lambda v: score_n(best_n, v), 3, 30,
                                         log_space=False, is_int=True)
    for h in hist_n:
        h["axis"] = "n_estimators"
    for h in hist_d:
        h["axis"] = "max_depth"
    return {"head": "rf", "best_params": {"n_estimators": best_n, "max_depth": best_d},
            "best_roc_auc": best_s_d, "history": hist_n + hist_d}


def load_embeddings_subject(model, condition, target_col, label_df):
    acfg = {"model_key": model, "target_col": target_col, "embedding_level": "subject"}
    X, y, groups = ra.load_precomputed_embeddings(
        acfg, {"paths": {}}, label_df, {"conditions": [condition]})
    return X, y, groups


def load_handcrafted_subject(condition, target_col, label_df):
    path = f"{FEATDIR}/sensor_subject_features.csv"
    df = pd.read_csv(path, low_memory=False)
    df["study_id"] = df["study_id"].astype(str)
    if "condition" in df.columns:
        df = df[df["condition"] == condition]
    feats = json.load(open(f"{FEATDIR}/sensor_subject_features_feature_columns.json"))
    feats = [c for c in feats if c in df.columns]
    lab = label_df[["study_id", target_col]].drop_duplicates("study_id").copy()
    lab["study_id"] = lab["study_id"].astype(str)
    lab = lab.rename(columns={target_col: "_label"})
    df = df.merge(lab, on="study_id", how="inner")
    y = df["_label"].astype(int).values
    groups = df["study_id"].values
    X = df[feats].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    if np.isnan(X).any():
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isnan(cm), 0.0, cm)
        X = np.where(np.isnan(X), cm, X)
    return X, y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["embeddings", "handcrafted"])
    ap.add_argument("--model", default=None, help="FM model key (embeddings source only)")
    ap.add_argument("--condition", required=True, choices=["EO_baseline", "EC_baseline"])
    ap.add_argument("--target-col", default="epilepsy")
    ap.add_argument("--label-csv", default=LABEL_CSV)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.source == "embeddings" and not args.model:
        raise SystemExit("--model required for --source embeddings")

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if args.target_col == "asm_resistant":
        label_df = label_df[label_df.epilepsy == 1].copy()
        print(f"asm_resistant: restricted to epilepsy==1 -> {len(label_df)} subjects", flush=True)

    if args.source == "embeddings":
        X, y, groups = load_embeddings_subject(args.model, args.condition, args.target_col, label_df)
        tag = args.model
    else:
        X, y, groups = load_handcrafted_subject(args.condition, args.target_col, label_df)
        tag = "handcrafted"

    print(f"[{tag}] loaded {X.shape} y={np.bincount(y).tolist()} n_subjects={len(np.unique(groups))}", flush=True)
    if X.shape[0] < 20 or len(np.unique(y)) < 2:
        print(f"[{tag}] too few samples/classes; skipping", flush=True)
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cshort = args.condition.replace("_baseline", "")

    results = {}
    for head_name, search_fn in [("logreg", search_lr), ("rf", search_rf)]:
        print(f"[{tag}/{cshort}] searching {head_name}...", flush=True)
        res = search_fn(X, y, groups)
        results[head_name] = res
        print(f"  {head_name}: best={res['best_params']} roc_auc={res['best_roc_auc']:.4f}", flush=True)

    row = {"source": args.source, "model": tag, "condition": cshort, "target_col": args.target_col,
           "n_subjects": int(len(np.unique(groups))), "n_features": int(X.shape[1])}
    for head_name, res in results.items():
        row[f"{head_name}_best_params"] = json.dumps(res["best_params"])
        row[f"{head_name}_roc_auc"] = round(res["best_roc_auc"], 4)

    summary_path = out / f"hp_search_{tag}_{cshort}.csv"
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    detail_path = out / f"hp_search_{tag}_{cshort}_history.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"--> wrote {summary_path} and {detail_path}", flush=True)


if __name__ == "__main__":
    main()
