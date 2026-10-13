"""Per-cohort metrics for a finetune run, computed by slicing the held-out
test-fold predictions (no per-cohort retraining). Given the pooled per-fold
(y, proba, groups=study_id), it reports metrics for the whole cohort plus each
demographic subset (sex / age / comorbidity), at the finetune's own granularity.

Balanced accuracy at the honest (out-of-fold threshold) operating point is the
headline; oracle + roc_auc kept for reference.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

import run_analysis as ra
import pooled_metrics
import cohorts

# cohort -> mask, built from the shared definitions in cohorts.py
# (skips small/redundant cohorts per standing rules; keeps this file's row names)
COHORTS = {
    "all":        cohorts.MASKS["all"]["all"],
    "female":     cohorts.MASKS["sex"]["F"],
    "male":       cohorts.MASKS["sex"]["M"],
    "age_5_8":    cohorts.MASKS["age"]["5-8"],
    "age_9_12":   cohorts.MASKS["age"]["9-12"],
    "age_13_18":  cohorts.MASKS["age"]["13-18"],
    "como_none":  cohorts.MASKS["comorbidity"]["none"],
    "como_adhd":  cohorts.MASKS["comorbidity"]["adhd"],
    "como_both":  cohorts.MASKS["comorbidity"]["both"],
}


# which cohort belongs to which cohort_group folder
GROUP_OF = {
    "all": "all",
    "female": "sex", "male": "sex",
    "age_5_8": "age", "age_9_12": "age", "age_13_18": "age",
    "como_none": "comorbidity", "como_adhd": "comorbidity", "como_both": "comorbidity",
}


def write_split(out_base, level, strategy, model, cond, whole_metrics, cohort_metrics,
                provenance=None):
    """Write cohort-split results into the folder layout:
        {out_base}/{level}/{strategy}/{group}/results_{model}_{cond}.json
    group in {all, sex, age, comorbidity}. The 'all' group file also carries the
    whole-model fold metrics. Returns the list of paths written."""
    groups = {}
    for cname, cm in cohort_metrics.items():
        groups.setdefault(GROUP_OF.get(cname, "all"), {})[cname] = cm
    written = []
    for g, group_cohorts in groups.items():
        d = Path(out_base) / level / strategy / g
        d.mkdir(parents=True, exist_ok=True)
        payload = {"model": model, "condition": cond, "level": level, "strategy": strategy,
                   "cohort_group": g, "cohorts": group_cohorts}
        if provenance:
            payload["provenance"] = provenance   # label csv, cv seed, git commit
        if g == "all":
            payload["whole_metrics"] = whole_metrics   # the driver's own 5-fold metrics block
        p = d / f"results_{model}_{cond}.json"
        p.write_text(json.dumps(payload, indent=2))
        written.append(str(p))
    return written


def save_preds(out_base, level, strategy, model, cond, fold_preds):
    """Pickle the raw held-out fold predictions so cohort metrics can be re-derived
    later WITHOUT re-training (future metric changes become a cheap offline pass)."""
    import pickle
    d = Path(out_base) / level / strategy / "_preds"
    d.mkdir(parents=True, exist_ok=True)
    obj = [(np.asarray(y), np.asarray(p), np.asarray(g).astype(str)) for y, p, g in fold_preds]
    with open(d / f"{model}_{cond}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


def _norm(s):
    """Canonical study_id (shared normalizer in run_analysis)."""
    return ra.norm_study_id(s)


def _pool_by_subject(y, p, g):
    """Pool epoch predictions to one per subject: mean probability, majority label
    (the honest subject-level aggregation -- no waveform averaging)."""
    u = np.unique(g)
    sy = np.array([int(round(float(y[g == k].mean()))) for k in u])
    sp = np.array([float(p[g == k].mean()) for k in u])
    return sy, sp, u          # u = per-subject group ids (each its own bootstrap unit)


def compute(fold_preds, label_csv):
    """fold_preds: list of (y_te, proba_te, groups_te[study_id]) per outer fold.
    Slices held-out predictions per cohort and scores them POOLED (shared helper).
    The 'all' cohort also carries a `per_fold` block for comparison."""
    lab = ra.normalize_label_df(pd.read_csv(label_csv))
    lab["study_id"] = lab["study_id"].map(_norm)          # canonical ids (fix padding)
    lab = lab.drop_duplicates("study_id").set_index("study_id")
    out = {}
    for name, mask_fn in COHORTS.items():
        sids = set(lab[mask_fn(lab)].index)
        fa, fa_subj, n = [], [], 0
        for y, p, g in fold_preds:
            g = np.array([_norm(x) for x in np.asarray(g)])  # canonical ids on both sides
            m = np.isin(g, list(sids))
            if m.sum() == 0:
                continue
            ym, pm, gm = np.asarray(y)[m], np.asarray(p)[m], g[m]
            fa.append((ym, pm, gm)); n += int(m.sum())
            fa_subj.append(_pool_by_subject(ym, pm, gm))   # epoch preds -> one per subject
        res = pooled_metrics.pooled(fa, calibrated=True, per_fold=(name == "all"))
        if res is not None:
            res["n"] = n
            # honest subject-level: pool epoch predictions per subject, then score.
            # (For subject-level runs each group is already one row, so this is a no-op.)
            sres = pooled_metrics.pooled(fa_subj, calibrated=True)
            if sres is not None:
                res["subject_pooled"] = sres
            out[name] = res
    return out
