"""
Multi-model + multi-condition ENSEMBLE (late fusion) of frozen FM embeddings —
the cheap "maximize the FM number" move that needs no finetuning.

Each model produces different epoch counts per subject (different windowing), so
we fuse at the SUBJECT level: mean-pool each model's epoch embeddings to one
vector per subject (per condition), which aligns everything. We then test both
fusion styles and isolate where the gains come from:

  single (model, cond)        - baselines
  multi-model  (per cond)     - fuse models within a condition
  multi-cond   (per model)    - fuse EO+EC within a model
  full ensemble (all x both)  - fuse everything

Fusion styles:
  concat : z-score each block, concatenate, logreg          (feature-level)
  avg    : per-block logreg -> per-subject proba, average   (prediction-level)

5-fold StratifiedGroupKFold on the common-subject set. Reports subject-level
ROC-AUC and honest calibrated balanced accuracy (train threshold -> test).

Usage:
    python ensemble_fm.py --out-json <path>
"""
from __future__ import annotations
import argparse, json, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402

MODELS = ["reve", "cbramod", "luna", "labram", "biot"]
CONDS = ["EO_baseline", "EC_baseline"]
N_SPLITS = 5
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def _subject_means(model, cond, label_df):
    acfg = {"model_key": model, "target_col": "epilepsy", "embedding_level": "epoch"}
    X, y, g = ra.load_precomputed_embeddings(acfg, {"paths": {}}, label_df, {"conditions": [cond]})
    g = np.asarray(g).astype(str)
    u = np.unique(g)
    means = {s: X[g == s].mean(0).astype(np.float32) for s in u}
    lab = {s: int(y[g == s][0]) for s in u}
    return means, lab


def _calib(y_tr, p_tr, y_te, p_te):
    from sklearn.metrics import balanced_accuracy_score
    bt, bb = 0.5, -1.0
    for t in np.unique(p_tr):
        ba = balanced_accuracy_score(y_tr, (p_tr >= t).astype(int))
        if ba > bb:
            bb, bt = ba, t
    return float(balanced_accuracy_score(y_te, (p_te >= bt).astype(int)))


def _evaluate(blocks, subj, y, style):
    """blocks: list of (n_subj, d) arrays aligned to subj order. Returns roc, calib_bacc."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    groups = np.arange(len(subj))
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    rocs, cals = [], []
    for tr, te in sgkf.split(blocks[0], y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        if style == "concat":
            scs = [StandardScaler().fit(B[tr]) for B in blocks]
            Xtr = np.hstack([s.transform(B[tr]) for s, B in zip(scs, blocks)])
            Xte = np.hstack([s.transform(B[te]) for s, B in zip(scs, blocks)])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr])
            p_tr = clf.predict_proba(Xtr)[:, 1]; p_te = clf.predict_proba(Xte)[:, 1]
        else:  # avg of per-block logreg probabilities
            ptr_list, pte_list = [], []
            for B in blocks:
                sc = StandardScaler().fit(B[tr])
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(B[tr]), y[tr])
                ptr_list.append(clf.predict_proba(sc.transform(B[tr]))[:, 1])
                pte_list.append(clf.predict_proba(sc.transform(B[te]))[:, 1])
            p_tr = np.mean(ptr_list, 0); p_te = np.mean(pte_list, 0)
        rocs.append(float(roc_auc_score(y[te], p_te)))
        cals.append(_calib(y[tr], p_tr, y[te], p_te))
    return float(np.mean(rocs)), float(np.mean(cals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()
    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))

    data = {}          # (model,cond) -> (means dict, lab dict)
    subj_sets = []
    for model in MODELS:
        for cond in CONDS:
            try:
                means, lab = _subject_means(model, cond, label_df)
                data[(model, cond)] = (means, lab)
                subj_sets.append(set(means))
                print(f"loaded {model}/{cond}: {len(means)} subjects", flush=True)
            except Exception as e:
                print(f"MISSING {model}/{cond}: {e}", flush=True)
    common = sorted(set.intersection(*subj_sets))
    print(f"\ncommon subjects across all {len(data)} blocks: {len(common)}", flush=True)
    any_lab = data[next(iter(data))][1]
    y = np.array([any_lab[s] for s in common])
    print(f"classes: {np.bincount(y).tolist()}\n", flush=True)

    def block(model, cond):
        means = data[(model, cond)][0]
        return np.stack([means[s] for s in common])

    results = {}

    def run(name, keys, style):
        blocks = [block(m, c) for (m, c) in keys]
        roc, cal = _evaluate(blocks, common, y, style)
        results[f"{name}|{style}"] = {"name": name, "roc_auc": roc, "calib_bacc": cal,
                                      "n_blocks": len(keys), "style": style}
        print(f"{name:34s} [{style:6s}] roc_auc={roc:.3f}  calib_bacc={cal:.3f}", flush=True)

    print("---- single (model, condition) ----")
    for m in MODELS:
        for c in CONDS:
            run(f"{m}/{c.split('_')[0]}", [(m, c)], "concat")
    print("---- multi-MODEL within a condition ----")
    for c in CONDS:
        run(f"ALLMODELS/{c.split('_')[0]}", [(m, c) for m in MODELS], "concat")
        run(f"ALLMODELS/{c.split('_')[0]}", [(m, c) for m in MODELS], "avg")
    print("---- multi-CONDITION within a model ----")
    for m in MODELS:
        run(f"{m}/EO+EC", [(m, c) for c in CONDS], "concat")
    print("---- FULL ensemble (all models x both conditions) ----")
    allkeys = list(itertools.product(MODELS, CONDS))
    run("FULL_ensemble", allkeys, "concat")
    run("FULL_ensemble", allkeys, "avg")

    best_single = max(v["roc_auc"] for v in results.values()
                      if "ALL" not in v["name"] and "+" not in v["name"] and v["name"] != "FULL_ensemble")
    full = max(v["roc_auc"] for v in results.values() if v["name"] == "FULL_ensemble")
    print(f"\nbest single (model,cond) roc_auc = {best_single:.3f}  |  "
          f"FULL ensemble roc_auc = {full:.3f}  (delta {full - best_single:+.3f})", flush=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(
        {"n_common": len(common), "classes": np.bincount(y).tolist(),
         "best_single": best_single, "results": results}, indent=2))
    print(f"--> wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
