"""Shared 5-fold CV harness for the LoRA finetune drivers (run_finetune_lora,
finetune_tslm_lora). The two drivers build and train different model families, but
the fold loop, scoring, and per-cohort output are identical -- that lives here.

Each driver passes its own `fit_predict(X_tr, y_tr, X_te, strategy, k) -> P(class=1)`
closure; run_cv handles splitting, scoring, and writing the cohort breakdown.
"""
from __future__ import annotations
import numpy as np

N_SPLITS = 5
METRICS = ["accuracy", "balanced_accuracy", "roc_auc"]


def run_cv(model, cond, strategy, level, out_dir, label_csv, X, y, groups, fit_predict):
    """Run 5-fold StratifiedGroupKFold (subject-grouped), score each fold uniformly,
    save the held-out predictions, and write the per-cohort split. Returns the
    whole-model metrics dict."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    import cohort_ft_metrics

    folds = {m: [] for m in METRICS}
    fold_preds = []   # (y_te, proba1, groups_te) per fold, for the cohort breakdown
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {k}: single-class test, skip", flush=True); continue
        p1 = np.asarray(fit_predict(X[tr], y[tr], X[te], strategy, k), dtype=float)
        p1 = np.nan_to_num(p1, nan=0.5, posinf=1.0, neginf=0.0)
        pred = (p1 >= 0.5).astype(int)
        folds["accuracy"].append(float(accuracy_score(y[te], pred)))
        folds["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], pred)))
        folds["roc_auc"].append(float(roc_auc_score(y[te], p1)))
        fold_preds.append((y[te], p1, groups[te]))
        print(f"  fold {k}: acc={folds['accuracy'][-1]:.3f} "
              f"bacc={folds['balanced_accuracy'][-1]:.3f} auc={folds['roc_auc'][-1]:.3f}", flush=True)

    metrics = {m: {"mean": float(np.nanmean(v)) if v else float("nan"),
                   "std": float(np.nanstd(v)) if v else float("nan"), "folds": v}
               for m, v in folds.items()}
    cohort_ft_metrics.save_preds(out_dir, level, strategy, model, cond, fold_preds)
    cohort_metrics = cohort_ft_metrics.compute(fold_preds, label_csv)
    for cname, cm in cohort_metrics.items():
        hb = cm.get("youden_threshold_balanced_accuracy", {}).get("mean")
        print(f"    cohort {cname}: honest_bal_acc={hb} (n={cm.get('n')})", flush=True)
    written = cohort_ft_metrics.write_split(out_dir, level, strategy, model, cond, metrics, cohort_metrics)
    print(f"--> wrote {len(written)} group files under {out_dir}/{level}/{strategy}/ "
          f"(bacc={metrics['balanced_accuracy']['mean']:.3f} "
          f"auc={metrics['roc_auc']['mean']:.3f})", flush=True)
    return metrics
