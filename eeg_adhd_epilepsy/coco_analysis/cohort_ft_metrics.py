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

# cohort -> mask on the normalized label_df (skip small/redundant per standing rules)
COHORTS = {
    "all":    lambda d: pd.Series(True, index=d.index),
    "female": lambda d: d.sex == "F",
    "male":   lambda d: d.sex == "M",
    "age_5_8":   lambda d: d.age_group == "5-8",
    "age_9_12":  lambda d: d.age_group == "9-12",
    "age_13_18": lambda d: d.age_group == "13-18",
    "como_none": lambda d: (d.autism == 0) & (d.adhd == 0),
    "como_adhd": lambda d: (d.autism == 0) & (d.adhd == 1),
    "como_both": lambda d: (d.autism == 1) & (d.adhd == 1),
}


# which cohort belongs to which cohort_group folder
GROUP_OF = {
    "all": "all",
    "female": "sex", "male": "sex",
    "age_5_8": "age", "age_9_12": "age", "age_13_18": "age",
    "como_none": "comorbidity", "como_adhd": "comorbidity", "como_both": "comorbidity",
}


def write_split(out_base, level, strategy, model, cond, whole_metrics, cohort_metrics):
    """Write cohort-split results into the folder layout:
        {out_base}/{level}/{strategy}/{group}/results_{model}_{cond}.json
    group in {all, sex, age, comorbidity}. The 'all' group file also carries the
    whole-model fold metrics. Returns the list of paths written."""
    groups = {}
    for cname, cm in cohort_metrics.items():
        groups.setdefault(GROUP_OF.get(cname, "all"), {})[cname] = cm
    written = []
    for g, cohorts in groups.items():
        d = Path(out_base) / level / strategy / g
        d.mkdir(parents=True, exist_ok=True)
        payload = {"model": model, "condition": cond, "level": level, "strategy": strategy,
                   "cohort_group": g, "cohorts": cohorts}
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
    """Canonical study_id: strip zero-padding so the finetune's '0089' groups match
    the label file's plain '89'. Non-numeric ids pass through unchanged."""
    try:
        return str(int(str(s).strip()))
    except (ValueError, TypeError):
        return str(s)


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
        fa = []
        n = 0
        for y, p, g in fold_preds:
            g = np.array([_norm(x) for x in np.asarray(g)])  # canonical ids on both sides
            m = np.isin(g, list(sids))
            if m.sum() == 0:
                continue
            fa.append((np.asarray(y)[m], np.asarray(p)[m])); n += int(m.sum())
        res = pooled_metrics.pooled(fa, calibrated=True, per_fold=(name == "all"))
        if res is not None:
            res["n"] = n
            out[name] = res
    return out
