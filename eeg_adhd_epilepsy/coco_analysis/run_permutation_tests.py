"""
Permutation-based statistical tests across EEG foundation model experiments.

Tests run:
  Q1 - vs chance          : each model x experiment vs. empirical null (ROC AUC)
  Q2 - model comparison   : cbramod vs reve vs biot within each experiment (subject-level paired permutation)
  Q3 - strategy           : ft_only vs lp_ft per model (subject-level paired permutation)
  Q4 - aggregation        : epoch-level vs subject-level ROC AUC per fold (Wilcoxon)

Q2/Q3 use the paired permutation test in stats.py (assess_paired_comparison) at the subject
level (unit="group"): predictions are paired within subject and model labels are permuted,
giving ~N_subjects of resolution. Wilcoxon over 5 folds was previously used but is incapable
of p<0.05 (its two-sided floor at n=5 is 2/2^5 = 0.0625).

Q4 stays on Wilcoxon across folds: it compares two scorings of the *same* model (epoch vs
subject aggregation), which does not map onto the paired model-label-swap permutation.

Results saved to permute_testing.csv (one row per test).
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/mat/projects/coco-pipe")
from coco_pipe.decoding.stats import (
    assess_paired_comparison,
    assess_post_hoc_permutation,
    benjamini_hochberg,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FT_ONLY_DIR = Path("/home/mat/scratch/results/fine_tune/adjusted_lr")
LP_FT_DIR   = Path("/home/mat/scratch/results/fine_tune/lp_ft/base")
OUT_CSV     = Path(__file__).parent / "permute_testing.csv"

MODELS     = ["cbramod", "reve", "biot"]
CONDITIONS = ["EO", "EC"]
# Chance-level test uses the library's group-aware label-shuffle permutation,
# which is a pure-Python per-row loop (~14s / 100 perms over ~23k pooled epochs).
# 200 perms gives p-resolution ~0.005, ample for a clearly-above-chance signal.
N_PERM_CHANCE = 200
# Paired model comparison (Q2/Q3): subject-level swap loop is pure-Python over
# ~900 subjects, so ~70s / 200 perms per comparison.
N_PERM_PAIRED = 200
SEED          = 42
METRIC        = "roc_auc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_res_cond(model: str, cond: str, exp_dir: Path) -> dict | None:
    p = exp_dir / f"results_{model}_{cond}.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    keys = list(data["results"].keys())
    if not keys:
        return None
    return data["results"][keys[0]]


def fold_roc(res: dict) -> np.ndarray:
    """ROC AUC per CV fold (epoch-level)."""
    scores = []
    for fold in res.get("predictions", []):
        y_true  = np.asarray(fold["y_true"])
        y_proba = np.asarray(fold["y_proba"])
        scores.append(roc_auc_score(y_true, y_proba[:, 1]))
    return np.array(scores)


def fold_roc_subject(res: dict) -> np.ndarray:
    """ROC AUC per CV fold after aggregating epochs to subject level."""
    scores = []
    for fold in res.get("predictions", []):
        y_true  = np.asarray(fold["y_true"])
        y_proba = np.asarray(fold["y_proba"])
        groups  = np.asarray(fold["group"])
        subj_y, subj_p = [], []
        for g in np.unique(groups):
            m = groups == g
            subj_y.append(int(np.bincount(y_true[m].astype(int)).argmax()))
            subj_p.append(float(y_proba[m, 1].mean()))
        scores.append(roc_auc_score(subj_y, subj_p))
    return np.array(scores)


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float, float]:
    """Return (mean_a, mean_b, p_value) via Wilcoxon signed-rank (two-sided)."""
    _, p = wilcoxon(scores_a, scores_b, alternative="two-sided")
    return float(np.mean(scores_a)), float(np.mean(scores_b)), float(p)


def folds_to_df(res: dict) -> pd.DataFrame:
    """Tidy per-epoch DataFrame pooled over all CV folds (for paired permutation)."""
    rows = []
    for fold_idx, fold in enumerate(res.get("predictions", [])):
        y_true  = np.asarray(fold["y_true"])
        y_proba = np.asarray(fold["y_proba"])
        groups  = np.asarray(fold["group"])
        sids    = np.asarray(fold.get("sample_id", np.arange(len(y_true))))
        for i in range(len(y_true)):
            rows.append({
                "SampleID":  sids[i],
                "Group":     groups[i],
                "Fold":      fold_idx,
                "y_true":    int(y_true[i]),
                "y_pred":    int(fold["y_pred"][i]),
                "y_proba_0": float(y_proba[i][0]),
                "y_proba_1": float(y_proba[i][1]),
            })
    return pd.DataFrame(rows)


def merge_models(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Merge two tidy DFs on SampleID+Fold with _A/_B suffixes for paired tests.

    Adds a bare shared y_true and keeps Group_A, which assess_paired_comparison
    uses to pair predictions within subject (unit='group').
    """
    merged = df_a.merge(df_b, on=["SampleID", "Fold"], suffixes=("_A", "_B"))
    merged["y_true"] = merged["y_true_A"]
    return merged


def paired_test(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Subject-level paired permutation. Returns (score_a, score_b, diff, p, n_units)."""
    merged = merge_models(df_a, df_b)
    comp = assess_paired_comparison(
        merged, metric=METRIC, unit="group",
        n_permutations=N_PERM_PAIRED, random_state=SEED,
    )
    return (
        float(comp["ScoreA"].iloc[0]),
        float(comp["ScoreB"].iloc[0]),
        float(comp["Difference"].iloc[0]),
        float(comp["PValue"].iloc[0]),
        int(comp["NUnits"].iloc[0]),
    )


def make_row(test_type, exp_a, exp_b, model_a, model_b, cond,
             score_a, score_b, diff, p_val, n_units, n_perm, notes=""):
    return {
        "test_type":      test_type,
        "experiment_a":   exp_a,
        "experiment_b":   exp_b,
        "model_a":        model_a,
        "model_b":        model_b,
        "condition":      cond,
        "metric":         METRIC,
        "score_a":        round(float(score_a), 4) if score_a is not None else None,
        "score_b":        round(float(score_b), 4) if score_b is not None else None,
        "difference":     round(float(diff), 4)    if diff    is not None else None,
        "p_value":        round(float(p_val), 4)   if p_val   is not None else None,
        "p_corrected":    None,
        "significant":    None,
        "n_units":        n_units,
        "n_permutations": n_perm,
        "notes":          notes,
    }


def bh_correct(df: pd.DataFrame) -> pd.DataFrame:
    """BH correction within each test_type group."""
    df = df.copy()
    for ttype in df["test_type"].unique():
        mask  = df["test_type"] == ttype
        pvals = df.loc[mask, "p_value"].values.astype(float)
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        # returns (adjusted_pvalues, rejected_bools)
        p_corr, reject = benjamini_hochberg(pvals[valid].tolist(), alpha=0.05)
        idxs = np.where(mask)[0][valid]
        for idx, pc, rej in zip(idxs, p_corr, reject):
            df.at[idx, "p_corrected"] = round(float(pc), 4)
            df.at[idx, "significant"] = bool(rej)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = []

    # Q1: vs chance (post-hoc permutation on pooled fold predictions)
    print("Q1: chance-level tests ...")
    for exp_label, exp_dir in [("ft_only", FT_ONLY_DIR), ("lp_ft", LP_FT_DIR)]:
        for model in MODELS:
            for cond in CONDITIONS:
                res = load_res_cond(model, cond, exp_dir)
                if res is None:
                    print(f"  SKIP {exp_label}/{model}/{cond} -- no file")
                    continue
                try:
                    perm  = assess_post_hoc_permutation(
                        res, metric=METRIC, unit=None,
                        n_permutations=N_PERM_CHANCE, random_state=SEED,
                    )
                    obs   = float(perm["Observed"].iloc[0])
                    p_val = float(perm["PValue"].iloc[0])
                    # obs score from pooled folds; compare to 0.5
                    rows.append(make_row(
                        "vs_chance", exp_label, "chance",
                        model, "chance_level", cond,
                        score_a=obs, score_b=0.5,
                        diff=obs - 0.5, p_val=p_val,
                        n_units=None, n_perm=N_PERM_CHANCE,
                        notes="group-level label shuffle permutation",
                    ))
                    print(f"  {exp_label}/{model}/{cond}: ROC={obs:.3f} p={p_val:.4f}")
                except Exception as e:
                    print(f"  ERROR {exp_label}/{model}/{cond}: {e}")

    # Q2: model comparison within experiment (subject-level paired permutation)
    print("\nQ2: model comparison (subject-level paired permutation) ...")
    for exp_label, exp_dir in [("ft_only", FT_ONLY_DIR), ("lp_ft", LP_FT_DIR)]:
        for cond in CONDITIONS:
            dfs = {}
            for model in MODELS:
                res = load_res_cond(model, cond, exp_dir)
                if res is not None:
                    dfs[model] = folds_to_df(res)

            for ma, mb in combinations(dfs.keys(), 2):
                try:
                    sa, sb, diff, p, n_u = paired_test(dfs[ma], dfs[mb])
                    rows.append(make_row(
                        "model_comparison", exp_label, exp_label,
                        ma, mb, cond,
                        score_a=sa, score_b=sb,
                        diff=diff, p_val=p,
                        n_units=n_u,
                        n_perm=N_PERM_PAIRED,
                        notes="subject-level paired permutation; positive diff = model_a better",
                    ))
                    print(f"  {exp_label} {ma} vs {mb} {cond}: "
                          f"{ma}={sa:.3f} {mb}={sb:.3f} diff={diff:+.3f} "
                          f"p={p:.4f} (n_subj={n_u})")
                except Exception as e:
                    print(f"  ERROR {exp_label} {ma} vs {mb} {cond}: {e}")

    # Q3: ft_only vs lp_ft per model (subject-level paired permutation)
    print("\nQ3: ft_only vs lp_ft (subject-level paired permutation) ...")
    for model in MODELS:
        for cond in CONDITIONS:
            res_ft = load_res_cond(model, cond, FT_ONLY_DIR)
            res_lp = load_res_cond(model, cond, LP_FT_DIR)
            if res_ft is None or res_lp is None:
                print(f"  SKIP {model}/{cond} -- missing result")
                continue
            try:
                # A=ft_only, B=lp_ft
                sa, sb, diff, p, n_u = paired_test(folds_to_df(res_ft), folds_to_df(res_lp))
                rows.append(make_row(
                    "strategy_comparison", "ft_only", "lp_ft",
                    model, model, cond,
                    score_a=sa, score_b=sb,
                    diff=diff, p_val=p,
                    n_units=n_u,
                    n_perm=N_PERM_PAIRED,
                    notes="subject-level paired permutation; positive diff = ft_only better, negative = lp_ft better",
                ))
                print(f"  {model}/{cond}: ft_only={sa:.3f} lp_ft={sb:.3f} "
                      f"diff={diff:+.3f} p={p:.4f} (n_subj={n_u})")
            except Exception as e:
                print(f"  ERROR {model}/{cond}: {e}")

    # Q4: epoch vs subject aggregation (Wilcoxon across folds)
    print("\nQ4: epoch vs subject-level aggregation (Wilcoxon across folds) ...")
    for model in MODELS:
        for cond in CONDITIONS:
            res = load_res_cond(model, cond, FT_ONLY_DIR)
            if res is None:
                continue
            try:
                epoch_rocs = fold_roc(res)
                subj_rocs  = fold_roc_subject(res)
                sa, sb, p  = wilcoxon_test(epoch_rocs, subj_rocs)
                rows.append(make_row(
                    "aggregation_comparison", "ft_only_epoch", "ft_only_subject",
                    model, model, cond,
                    score_a=sa, score_b=sb,
                    diff=sb - sa,           # positive = subject better
                    p_val=p,
                    n_units=len(epoch_rocs),
                    n_perm=0,
                    notes="Wilcoxon across folds; positive diff = subject-level better",
                ))
                print(f"  {model}/{cond}: epoch={sa:.3f} subject={sb:.3f} "
                      f"diff={sb-sa:+.3f} p={p:.4f}")
            except Exception as e:
                print(f"  ERROR {model}/{cond}: {e}")

    # BH correction within each test type
    print("\nApplying BH correction ...")
    df = bh_correct(pd.DataFrame(rows))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")

    sig = df[df["significant"] == True]
    print(f"\nSignificant results (corrected p < 0.05): {len(sig)}/{len(df)}")
    if len(sig):
        print(sig[["test_type", "experiment_a", "model_a", "model_b",
                    "condition", "score_a", "score_b", "p_corrected"]].to_string(index=False))


if __name__ == "__main__":
    main()
