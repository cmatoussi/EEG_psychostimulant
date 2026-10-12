"""
Dimensionality reduction on the pre-extracted FM embeddings (one model at a time),
at two levels:
  epoch          - one point per epoch (subsampled to EPOCH_CAP for tractability;
                   Isomap and the structure metrics are O(n^2)).
  averaged_epoch - one point per subject = the mean of that subject's epoch
                   embeddings (the *_subject_embeddings.csv).

Same MAD outlier rejection + robust-scale/clip + PCA/UMAP/Isomap x components +
self-contained HTML report as the handcrafted sweep (helpers reused from
run_dimred_cohorts). Each FM has its own embedding space, so results are written
per model. Reports are coloured by epilepsy / sex / age / comorbidity / source.

Usage:
    python run_dimred_embeddings.py --model cbramod \
        --out-dir /home/mat/scratch/results/dim_results/extracted_embeddings
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402
import run_dimred_cohorts as dc  # noqa: E402  (reuse reject/scale/meta/reduce helpers)
import cohort_balance  # noqa: E402

LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
MODELS = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
CONDITION = "EO_baseline"
# name -> embedding_level passed to load_precomputed_embeddings
LEVELS = {"epoch": "epoch", "averaged_epoch": "subject"}
EPOCH_CAP = 8000  # subsample cap for the epoch level (O(n^2) reducers/metrics)


def _subsample(X, y, groups, cap, seed=42):
    """Row cap by whole subjects, class-stratified to the natural ratio."""
    if X.shape[0] <= cap:
        return X, y, groups
    rng = np.random.RandomState(seed)
    keep = []
    for cls in np.unique(y):
        subs = rng.permutation(np.unique(groups[y == cls]))
        budget = int(round(cap * (y == cls).mean()))
        tot = 0
        for s in subs:                       # whole subjects until the class budget
            idx = np.where(groups == s)[0]
            keep.append(idx); tot += len(idx)
            if tot >= budget:
                break
    idx = np.sort(np.concatenate(keep))
    return X[idx], y[idx], groups[idx]


SKIP = {("age", "0-4"), ("comorbidity", "asd"), ("sex", "ALL")}  # small/redundant, per standing rules


def _norm(s):
    """Canonical study_id (shared normalizer in run_analysis)."""
    return ra.norm_study_id(s)


def _cohorts_for(group):
    """{subdir_path: (gname, mask_fn)} for a cohort-group; 'everything' unions all
    groups (all / sex / age / comorbidity), skipping the small/redundant ones."""
    out = {}
    groups = dc.COHORT_GROUPS if group == "everything" else {group: dc.COHORT_GROUPS[group]}
    for gname, gc in groups.items():
        for ckey, (subdir, mask_fn) in gc.items():
            if (gname, ckey) in SKIP:
                continue
            out["all" if gname == "all" else f"{gname}/{subdir}"] = (gname, mask_fn)
    return out


def run_model_level(model, level_name, emb_level, label_df, out_root, condition,
                    cohort_group="all", target_col="epilepsy", balanced=False):
    cond_short = condition.replace("_baseline", "")
    base = Path(out_root) / level_name / cond_short / model
    base.mkdir(parents=True, exist_ok=True)
    acfg = {"model_key": model, "target_col": target_col, "embedding_level": emb_level}
    try:
        X, y, groups = ra.load_precomputed_embeddings(
            acfg, {"paths": {}}, label_df, {"conditions": [condition]})
    except FileNotFoundError as e:
        print(f"[{model}/{level_name}] MISSING: {e}", flush=True)
        (base / "SKIPPED.txt").write_text(f"missing embeddings: {e}\n")
        return
    print(f"[{model}/{level_name}] loaded {X.shape} "
          f"(y {np.unique(y, return_counts=True)[1].tolist()})", flush=True)

    if level_name == "epoch" and X.shape[0] > EPOCH_CAP:
        X, y, groups = _subsample(X, y, groups, EPOCH_CAP)
        print(f"    subsampled epochs -> {X.shape[0]}", flush=True)

    feats = [f"emb_{i}" for i in range(X.shape[1])]
    gids = np.array([_norm(g) for g in np.asarray(groups)])
    for path, (gname, mask_fn) in _cohorts_for(cohort_group).items():  # per-cohort projections
        cohort_label_df = label_df[mask_fn(label_df)]
        if balanced:
            cohort_label_df, n_bal, drop_reason = cohort_balance.build_balanced(
                cohort_label_df, target_col, gname)
            if drop_reason:
                print(f"    cohort {path}: BALANCED SKIP ({drop_reason})", flush=True)
                continue
        keep = {_norm(s) for s in cohort_label_df["study_id"]}
        m = np.isin(gids, list(keep))
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            print(f"    cohort {path}: skip (n={int(m.sum())})", flush=True)
            continue
        cout = base / path; cout.mkdir(parents=True, exist_ok=True)
        # subset -> reject -> scale WITHIN the cohort (matches run_dimred_cohorts;
        # scaling on everyone first would use out-of-cohort stats).
        Xc, yc, gc = X[m], y[m], groups[m]
        Xc, yc, gc, _ = dc.reject_outlier_rows(Xc, yc, gc, feats, dc.MAD_Z, dc.OUTLIER_FRAC, cout)
        if Xc.shape[0] < 20 or len(np.unique(yc)) < 2:
            print(f"    cohort {path}: skip after rejection (n={Xc.shape[0]})", flush=True)
            continue
        Xc = dc.scale_features(Xc, clip=dc.CLIP)
        metac = dc._cohort_metadata(gc, label_df)
        print(f"    cohort {path}: {Xc.shape[0]} samples "
              f"(y {np.unique(yc, return_counts=True)[1].tolist()})", flush=True)
        dc.reduce_and_report(Xc, yc, gc, metac, cout,
                             f"Dim-reduction — {model} / {cond_short} / {level_name} / {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--condition", default=CONDITION)
    ap.add_argument("--label-csv", default=None,
                    help="override; default = earliest-per-condition file")
    ap.add_argument("--levels", default="both",
                    choices=["epoch", "averaged_epoch", "both"])
    ap.add_argument("--target-col", default="epilepsy",
                    help="label column used as y for the coloured/supervised reports "
                         "(e.g. epilepsy, asm_resistant).")
    ap.add_argument("--cohort-group", default="all",
                    choices=list(dc.COHORT_GROUPS) + ["everything"],
                    help="'all', a single group, or 'everything' (all + sex + age + comorbidity).")
    ap.add_argument("--balanced", action="store_true",
                    help="uniform sex x age (x comorbidity, where free) matched "
                         "case/control cohort instead of the natural baseline population; "
                         "cohorts that can't reach 30 matched subjects are skipped.")
    args = ap.parse_args()

    args.label_csv = ra.resolve_label_csv(args.condition, args.label_csv)
    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if args.target_col == "asm_resistant":
        label_df = ra.restrict_for_asm_resistant(label_df)
        print(f"asm_resistant target: treated epilepsy only (epilepsy==1 & asm==1) -> {len(label_df)} subjects", flush=True)
    levels = list(LEVELS) if args.levels == "both" else [args.levels]
    for level_name in levels:
        run_model_level(args.model, level_name, LEVELS[level_name],
                        label_df, args.out_dir, args.condition,
                        cohort_group=args.cohort_group, target_col=args.target_col,
                        balanced=args.balanced)


if __name__ == "__main__":
    main()
