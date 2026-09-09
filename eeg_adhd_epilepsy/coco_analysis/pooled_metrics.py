"""Pooled cross-validation metrics — one honest aggregation shared by the finetune
cohort slicer, the embedding probe, and the handcrafted reports.

Instead of averaging tiny, degenerate-prone per-fold scores, we concatenate every
fold's HELD-OUT predictions (each subject predicted once, by a model that never
trained on it) and score ONCE. Uncertainty comes from a *stratified* subject
bootstrap (resample within each class, so class sizes stay fixed and no resample
is degenerate) — the right uncertainty for a per-class metric like balanced acc.

This changes only how predictions are aggregated, never which predictions are used,
so it introduces no leakage: pooled-`all` reproduces the independently-computed
whole-cohort score, and the direction of change is conservative (removes the
dropped-fold survivor bias).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score)

_GRID = np.linspace(0.05, 0.95, 91)


def _best_thr(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    return float(_GRID[np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in _GRID])])


def _perfold(fa):
    """Old per-fold averaging (drops single-class folds). Kept for side-by-side
    comparison on the 'all'/whole cohort only."""
    ba, bacal, auc, f1 = [], [], [], []
    for i, (yi, pi) in enumerate(fa):
        if len(np.unique(yi)) < 2:
            continue
        auc.append(roc_auc_score(yi, pi))
        ba.append(balanced_accuracy_score(yi, (pi >= 0.5).astype(int)))
        if len(fa) > 1:
            oy = np.concatenate([fa[j][0] for j in range(len(fa)) if j != i])
            op = np.concatenate([fa[j][1] for j in range(len(fa)) if j != i])
        else:
            oy, op = yi, pi
        bacal.append(balanced_accuracy_score(yi, (pi >= _best_thr(oy, op)).astype(int)))
        f1.append(f1_score(yi, (pi >= 0.5).astype(int), average="weighted", zero_division=0))
    if not ba:
        return None
    ms = lambda v: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
    return {"balanced_accuracy": ms(ba), "balanced_accuracy_calibrated": ms(bacal),
            "roc_auc": ms(auc), "weighted_f1": ms(f1), "n_folds": len(ba)}


def pooled(fa, calibrated=True, per_fold=False, n_boot=300, seed=0):
    """fa: list of (y, proba) per fold, already sliced to a cohort. Returns a dict
    of pooled point estimates with stratified-bootstrap std; None if <2 classes.
    metrics: accuracy, balanced_accuracy (+_calibrated if `calibrated`), roc_auc,
    weighted_f1, plus n, n_folds. `per_fold` adds the old averaging for comparison."""
    fa = [(np.asarray(y), np.asarray(p)) for (y, p) in fa if len(y)]
    if not fa:
        return None
    Y = np.concatenate([y for y, _ in fa])
    P = np.concatenate([p for _, p in fa])
    if len(np.unique(Y)) < 2:
        return None
    pred = (P >= 0.5).astype(int)
    acc, ba = accuracy_score(Y, pred), balanced_accuracy_score(Y, pred)
    auc, f1 = roc_auc_score(Y, P), f1_score(Y, pred, average="weighted", zero_division=0)

    cal = None
    if calibrated:                                # out-of-fold threshold per fold, pooled binaries
        cal = np.empty(len(Y), int); off = 0
        for i, (yi, pi) in enumerate(fa):
            if len(fa) > 1:
                oy = np.concatenate([fa[j][0] for j in range(len(fa)) if j != i])
                op = np.concatenate([fa[j][1] for j in range(len(fa)) if j != i])
            else:
                oy, op = yi, pi
            cal[off:off + len(yi)] = (pi >= _best_thr(oy, op)).astype(int); off += len(yi)

    # stratified subject bootstrap: resample within each class (sizes fixed)
    pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
    rng = np.random.RandomState(seed)
    bba, bauc, bf1, bcal = [], [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        bba.append(balanced_accuracy_score(Y[idx], pred[idx]))
        bauc.append(roc_auc_score(Y[idx], P[idx]))
        bf1.append(f1_score(Y[idx], pred[idx], average="weighted", zero_division=0))
        if cal is not None:
            bcal.append(balanced_accuracy_score(Y[idx], cal[idx]))
    sd = lambda v: round(float(np.std(v)), 4) if len(v) else 0.0
    mn = lambda x: round(float(x), 4)

    out = {"accuracy": {"mean": mn(acc)},
           "balanced_accuracy": {"mean": mn(ba), "std": sd(bba)},
           "roc_auc": {"mean": mn(auc), "std": sd(bauc)},
           "weighted_f1": {"mean": mn(f1), "std": sd(bf1)},
           "n_folds": len(fa), "n": int(len(Y))}
    if cal is not None:
        out["balanced_accuracy_calibrated"] = {"mean": mn(balanced_accuracy_score(Y, cal)), "std": sd(bcal)}
    if per_fold:
        pf = _perfold(fa)
        if pf is not None:
            out["per_fold"] = pf
    return out
