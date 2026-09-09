"""
Test SUBJECT-ALIGNMENT (domain alignment across patients) on the finetuning
readout, for every FM. The "align features before the head" idea: force each
patient's embedding distribution to look alike so the head learns the disease
signature, not per-patient quirks — improving cross-subject generalization.

We test it on frozen epoch-level embeddings + a logistic-regression head (the
method operates BEFORE the head, so this isolates the alignment effect cheaply
and applies to any model). For each model we compare three feature treatments
under the SAME subject-grouped 5-fold CV:
  baseline       : global StandardScaler only
  subject-center : per-subject mean-centering (SubjectStandardScaler-style) then scale
  subject-zscore : per-subject full z-normalisation then scale (most aggressive)

Per-subject stats use only that subject's own epochs (transductive, label-free →
no leakage), applied to train and test alike. Caveat this test will reveal:
epilepsy is a between-subject property, so aligning subjects can REMOVE the
signal — hence we report whether each strength improves or hurts.

Reports epoch- and subject-level (averaged-predictions) ROC-AUC and an honest
calibrated balanced accuracy (threshold chosen on train, applied to test).

Usage:
    python test_subject_alignment.py --condition EO_baseline --out-json <path>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from coco_pipe.decoding._metrics import _balanced_accuracy_optimal_score  # noqa: E402
import run_analysis as ra  # noqa: E402

MODELS = ["reve", "cbramod", "luna", "bendr", "labram", "biot"]
N_SPLITS = 5
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"
MIN_EP = 3   # subjects with fewer epochs are left un-centred (mean of 1-2 epochs is unstable)


def _subject_align(X, groups, zscore):
    """Per-subject mean-centre (and optionally z-score) using each subject's own
    epochs. Leak-free: uses no labels; each subject normalised by itself."""
    Xa = X.astype(np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < MIN_EP:
            continue
        mu = X[m].mean(0)
        Xa[m] = X[m] - mu
        if zscore:
            sd = X[m].std(0)
            sd[sd < 1e-6] = 1.0
            Xa[m] = Xa[m] / sd
    return Xa.astype(np.float32)


def _score(y, p1, groups):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score  # noqa: F401
    out = {"roc_auc": float(roc_auc_score(y, p1)),
           "bacc_opt": _balanced_accuracy_optimal_score(y, p1)}
    u = np.unique(groups)
    sy = np.array([int(round(float(y[groups == g].mean()))) for g in u])
    sp = np.array([float(p1[groups == g].mean()) for g in u])
    out["subj_roc_auc"] = float(roc_auc_score(sy, sp)) if len(np.unique(sy)) > 1 else float("nan")
    out["subj_bacc_opt"] = _balanced_accuracy_optimal_score(sy, sp) if len(np.unique(sy)) > 1 else float("nan")
    return out, (sy, sp)


def _calib(y_tr, p_tr, y_te, p_te):
    from sklearn.metrics import balanced_accuracy_score
    bt, bb = 0.5, -1.0
    for t in np.unique(p_tr):
        ba = balanced_accuracy_score(y_tr, (p_tr >= t).astype(int))
        if ba > bb:
            bb, bt = ba, t
    return float(balanced_accuracy_score(y_te, (p_te >= bt).astype(int)))


def _run(X, y, groups, mode):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xa = X if mode == "baseline" else _subject_align(X, groups, zscore=(mode == "subject-zscore"))
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold = []
    for tr, te in sgkf.split(Xa, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(Xa[tr], y[tr])
        p_te = clf.predict_proba(Xa[te])[:, 1]
        p_tr = clf.predict_proba(Xa[tr])[:, 1]
        m, (syte, spte) = _score(y[te], p_te, groups[te])
        m["bacc_calib"] = _calib(y[tr], p_tr, y[te], p_te)
        _, (sytr, sptr) = _score(y[tr], p_tr, groups[tr])
        m["subj_bacc_calib"] = _calib(sytr, sptr, syte, spte) if len(np.unique(syte)) > 1 else float("nan")
        fold.append(m)
    return {k: float(np.nanmean([f[k] for f in fold])) for k in fold[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    cond = args.condition
    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    results = {}
    print(f"{'model':9s} | {'treatment':15s} | epoch_roc subj_roc bacc_cal subj_bacc_cal", flush=True)
    print("-" * 72)
    for model in MODELS:
        acfg = {"model_key": model, "target_col": "epilepsy", "embedding_level": "epoch"}
        try:
            X, y, groups = ra.load_precomputed_embeddings(acfg, {"paths": {}}, label_df, {"conditions": [cond]})
        except Exception as e:
            print(f"{model:9s} | MISSING embeddings: {e}", flush=True)
            continue
        results[model] = {}
        for mode in ("baseline", "subject-center", "subject-zscore"):
            r = _run(X, y, groups, mode)
            results[model][mode] = r
            print(f"{model:9s} | {mode:15s} | {r['roc_auc']:.3f}     {r['subj_roc_auc']:.3f}    "
                  f"{r['bacc_calib']:.3f}    {r['subj_bacc_calib']:.3f}", flush=True)
        b = results[model]["baseline"]["roc_auc"]
        for mode in ("subject-center", "subject-zscore"):
            d = results[model][mode]["roc_auc"] - b
            verdict = "IMPROVES" if d > 0.005 else ("HURTS" if d < -0.005 else "no change")
            print(f"           -> {mode}: epoch ROC {d:+.3f}  [{verdict}]", flush=True)
        print("-" * 72, flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"condition": cond.replace("_baseline", ""),
                                               "results": results}, indent=2))
    print(f"--> wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
