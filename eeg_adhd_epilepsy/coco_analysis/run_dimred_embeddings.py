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

LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
MODELS = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
CONDITION = "EO_baseline"
# name -> embedding_level passed to load_precomputed_embeddings
LEVELS = {"epoch": "epoch", "averaged_epoch": "subject"}
EPOCH_CAP = 8000  # subsample cap for the epoch level (O(n^2) reducers/metrics)


def _subsample(X, y, groups, cap, seed=42):
    if X.shape[0] <= cap:
        return X, y, groups
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=cap, replace=False))
    return X[idx], y[idx], groups[idx]


def run_model_level(model, level_name, emb_level, label_df, out_root, condition,
                    target_col="epilepsy"):
    cond_short = condition.replace("_baseline", "")
    out_dir = Path(out_root) / level_name / cond_short / model
    out_dir.mkdir(parents=True, exist_ok=True)
    acfg = {"model_key": model, "target_col": target_col, "embedding_level": emb_level}
    try:
        X, y, groups = ra.load_precomputed_embeddings(
            acfg, {"paths": {}}, label_df, {"conditions": [condition]})
    except FileNotFoundError as e:
        print(f"[{model}/{level_name}] MISSING: {e}", flush=True)
        (out_dir / "SKIPPED.txt").write_text(f"missing embeddings: {e}\n")
        return
    print(f"[{model}/{level_name}] loaded {X.shape} "
          f"(epi {np.unique(y, return_counts=True)[1].tolist()})", flush=True)

    if level_name == "epoch" and X.shape[0] > EPOCH_CAP:
        X, y, groups = _subsample(X, y, groups, EPOCH_CAP)
        print(f"    subsampled epochs -> {X.shape[0]}", flush=True)

    feats = [f"emb_{i}" for i in range(X.shape[1])]
    X, y, groups, _ = dc.reject_outlier_rows(
        X, y, groups, feats, dc.MAD_Z, dc.OUTLIER_FRAC, out_dir)
    if X.shape[0] < 20:
        print(f"[{model}/{level_name}] too few samples after rejection; skipping", flush=True)
        (out_dir / "SKIPPED.txt").write_text(f"only {X.shape[0]} after MAD rejection\n")
        return
    X = dc.scale_features(X, clip=dc.CLIP)
    print(f"    robust-scaled + clipped to +/-{dc.CLIP}", flush=True)
    meta = dc._cohort_metadata(groups, label_df)
    dc.reduce_and_report(X, y, groups, meta, out_dir,
                         f"Dim-reduction — {model} / {cond_short} / {level_name} embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--condition", default=CONDITION)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    ap.add_argument("--levels", default="both",
                    choices=["epoch", "averaged_epoch", "both"])
    ap.add_argument("--target-col", default="epilepsy",
                    help="label column used as y for the coloured/supervised reports "
                         "(e.g. epilepsy, asm_resistant).")
    args = ap.parse_args()

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    levels = list(LEVELS) if args.levels == "both" else [args.levels]
    for level_name in levels:
        run_model_level(args.model, level_name, LEVELS[level_name],
                        label_df, args.out_dir, args.condition, target_col=args.target_col)


if __name__ == "__main__":
    main()
