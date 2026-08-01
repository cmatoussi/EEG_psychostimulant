"""
Dimensionality-reduction cohort sweep with HTML reports.

For each cohort (all / sex / age / comorbidity), reduces the handcrafted
feature matrix with several reducers x n_components, scores structure
preservation + supervised separation (epilepsy), writes a summary CSV +
embeddings, and renders a single-file HTML report (coloured by epilepsy /
sex / age / comorbidity).

Usage:
    python run_dimred_cohorts.py --cohort-group comorbidity --cohort none \
        --source adhd --out-dir <dir>
    python run_dimred_cohorts.py --cohort-group all --out-dir <dir>   # no source filter
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402
from coco_pipe.dim_reduction import DimReduction  # noqa: E402
from coco_pipe.io.quality import compute_row_outlier_scores  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
FEATURE_CSV = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
               "signal_features/descriptors/combined/sensor_subject_features.csv")

REDUCERS = ["PCA", "UMAP", "Isomap"]
COMPONENTS = [2, 3, 5, 10, 15]
METRICS = ["trustworthiness", "continuity", "lcmc", "separation_logreg_balanced_accuracy"]

# MAD-based row-outlier rejection: a feature value is an outlier if its robust
# z-score (median + 1.4826*MAD) exceeds MAD_Z, and a row is dropped when the
# fraction of outlier features exceeds OUTLIER_FRAC. This strips the extreme
# points that otherwise dominate the dim-reduction scatters. NOTE: coco-pipe's
# epoch-level default is 0.30, but on the subject-level sensor matrix (2888
# feats) no row reaches that; 0.11 sits in the natural gap below the two clear
# extremes (frac~0.116) and above the rest (<=0.105), so it drops exactly them.
MAD_Z = 5.0
OUTLIER_FRAC = 0.11
# coco-pipe's reducers feed the raw feature array straight to sklearn PCA/Isomap,
# which are scale-sensitive: handcrafted features span many orders of magnitude,
# so a single subject with an extreme value in one high-magnitude feature stretches
# an axis to ~1e8 and collapses everyone else to a dot. Robust-scale (median/IQR,
# immune to those extremes) then clip to +/-CLIP so no single feature value can
# dominate the projection.
CLIP = 8.0


def scale_features(X, clip=CLIP):
    """Robust (median/IQR) standardization + winsorising clip, so scale-sensitive
    reducers (PCA/Isomap) aren't dominated by a few extreme feature values."""
    Xs = RobustScaler().fit_transform(np.asarray(X, dtype=float))
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=clip, neginf=-clip)
    if clip:
        Xs = np.clip(Xs, -clip, clip)
    return Xs.astype(np.float32)


def reject_outlier_rows(X, y, groups, feats, z_threshold, frac_threshold, out_dir):
    """Drop MAD-outlier rows before reduction; log which were dropped."""
    fdf = pd.DataFrame(np.asarray(X), columns=list(feats))
    scores = compute_row_outlier_scores(fdf, list(feats), z_threshold=z_threshold)
    fractions = scores["outlier_fraction"].to_numpy()
    keep = fractions <= frac_threshold
    n_drop = int((~keep).sum())
    if n_drop:
        dropped = pd.DataFrame({
            "id": np.asarray(groups)[~keep],
            "outlier_fraction": fractions[~keep],
            "mad_z_max": scores["mad_z_max"].to_numpy()[~keep],
        }).sort_values("outlier_fraction", ascending=False)
        dropped.to_csv(out_dir / "dropped_outliers.csv", index=False)
    print(f"    MAD rejection (z>{z_threshold}, frac>{frac_threshold}): "
          f"dropped {n_drop}/{len(keep)} rows", flush=True)
    return X[keep], np.asarray(y)[keep], np.asarray(groups)[keep], int(n_drop)

# cohort-group -> {cohort key: (subdir label, mask fn on normalized label_df)}
COHORT_GROUPS = {
    "all": {"all": ("all", lambda d: pd.Series(True, index=d.index))},
    "comorbidity": {
        "none": ("none", lambda d: (d.TSA == 0) & (d.TDAH == 0)),
        "asd":  ("asd",  lambda d: (d.TSA == 1) & (d.TDAH == 0)),
        "adhd": ("adhd", lambda d: (d.TSA == 0) & (d.TDAH == 1)),
        "both": ("both", lambda d: (d.TSA == 1) & (d.TDAH == 1)),
    },
    "sex": {
        "F":   ("female", lambda d: d.Sex == "F"),
        "M":   ("male",   lambda d: d.Sex == "M"),
        "ALL": ("all",    lambda d: pd.Series(True, index=d.index)),
    },
    "age": {
        "0-4":   ("age_0_4",   lambda d: d.age_group == "0-4"),
        "5-8":   ("age_5_8",   lambda d: d.age_group == "5-8"),
        "9-12":  ("age_9_12",  lambda d: d.age_group == "9-12"),
        "13-18": ("age_13_18", lambda d: d.age_group == "13-18"),
    },
}


def _cohort_metadata(groups, label_df):
    """Per-sample metadata (aligned to embedding rows) for report colouring."""
    lut = label_df.drop_duplicates("Study ID").set_index(label_df["Study ID"].astype(str))
    cols = {"Epilepsy": "Epilepsy", "Sex": "Sex", "age_group": "age_group",
            "TSA": "TSA", "TDAH": "TDAH", "source_dataset": "source_dataset"}
    meta = {}
    for out, col in cols.items():
        if col in lut.columns:
            meta[out] = np.array([lut.at[str(g), col] if str(g) in lut.index else None
                                  for g in groups], dtype=object)
    return meta


def reduce_and_report(X, y, groups, meta, out_dir, title, cond=None):
    """Run every reducer x n_components on X, score, save embeddings + summary +
    a self-contained HTML report. When `cond` is given (e.g. EO/EC) every output
    is suffixed with it so both conditions coexist in one folder; when None the
    names are unsuffixed (used where the condition is already in the path).
    Shared by the handcrafted and embedding sweeps."""
    from coco_pipe.report.dim_reduction import make_reduction_report
    sfx = f"_{cond}" if cond else ""
    reductions, embeddings, rows = [], [], []
    for method in REDUCERS:
        for n in COMPONENTS:
            try:
                red = DimReduction(method=method, n_components=n)
                emb = np.asarray(red.fit_transform(X))
                red.score(emb, X=X, metrics=METRICS, labels=y, groups=groups)
                m = {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                     for k, v in red.get_metrics().items()}
                reductions.append(red)
                embeddings.append(emb)
                np.save(out_dir / f"embedding_{method}_c{n}{sfx}.npy", emb)
                rows.append({"reducer": method, "n_components": n,
                             "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
                             **{mm: m.get(mm) for mm in METRICS}})
                print(f"    {method} c{n}: sep={m.get('separation_logreg_balanced_accuracy')} "
                      f"trust={m.get('trustworthiness')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    {method} c{n}: ERROR {e}", flush=True)
                rows.append({"reducer": method, "n_components": n, "status": f"error:{e}"})

    pd.DataFrame(rows).to_csv(out_dir / f"dim_reduction_summary{sfx}.csv", index=False)
    np.save(out_dir / f"ids{sfx}.npy", np.asarray(groups))
    np.save(out_dir / f"labels{sfx}.npy", np.asarray(y))
    with open(out_dir / f"metadata{sfx}.json", "w") as f:
        json.dump({k: [None if x is None else str(x) for x in v] for k, v in meta.items()}, f)

    if reductions:
        try:
            make_reduction_report(
                reductions, embeddings=embeddings, labels=np.asarray(y),
                metadata=meta, title=title, output_path=str(out_dir / f"report{sfx}.html"),
                asset_urls="inline")
            print(f"    report -> {out_dir/('report'+sfx+'.html')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    report FAILED: {e}", flush=True)


def run_cohort(cohort_key, subdir, mask_fn, label_df, out_root, condition,
               reject=True, mad_z=MAD_Z, outlier_frac=OUTLIER_FRAC, clip=CLIP):
    cohort_df = label_df[mask_fn(label_df)].copy()
    acfg = {"data_path": FEATURE_CSV, "target_col": "Epilepsy", "condition": condition}
    out_dir = Path(out_root) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, y, groups, feats = ra.load_dim_reduction_data(acfg, cohort_df)
    except Exception as e:  # noqa: BLE001
        print(f"[{cohort_key}] load failed: {e}", flush=True)
        (out_dir / "SKIPPED.txt").write_text(f"load failed: {e}\n")
        return
    if X.shape[0] < 20:
        print(f"[{cohort_key}] too few samples ({X.shape[0]}); skipping", flush=True)
        (out_dir / "SKIPPED.txt").write_text(f"only {X.shape[0]} samples\n")
        return

    if reject:
        X, y, groups, _ = reject_outlier_rows(
            X, y, groups, feats, mad_z, outlier_frac, out_dir)
        if X.shape[0] < 20:
            print(f"[{cohort_key}] too few samples after MAD rejection "
                  f"({X.shape[0]}); skipping", flush=True)
            (out_dir / "SKIPPED.txt").write_text(
                f"only {X.shape[0]} samples after MAD rejection\n")
            return

    if clip is not None:
        X = scale_features(X, clip=clip)
        print(f"    robust-scaled + clipped features to +/-{clip}", flush=True)

    meta = _cohort_metadata(groups, cohort_df)
    print(f"[{cohort_key}] {X.shape[0]} samples x {X.shape[1]} feats "
          f"(epi {np.unique(y, return_counts=True)[1].tolist()})", flush=True)

    cond_tag = condition.replace("_baseline", "")
    reduce_and_report(X, y, groups, meta, out_dir,
                      f"Dim-reduction — {cohort_key} ({cond_tag})", cond=cond_tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-group", required=True, choices=list(COHORT_GROUPS))
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--source", default=None, help="filter to a source_dataset (e.g. adhd)")
    ap.add_argument("--label-csv", default=LABEL_CSV,
                    help="metadata CSV to use (default: patients_metadata_clean.csv).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--no-reject", action="store_true",
                    help="disable MAD-based outlier-row rejection before reduction.")
    ap.add_argument("--mad-z", type=float, default=MAD_Z,
                    help="robust z-score threshold for a feature value to count as outlier.")
    ap.add_argument("--outlier-frac", type=float, default=OUTLIER_FRAC,
                    help="drop a row when its fraction of outlier features exceeds this.")
    ap.add_argument("--clip", type=float, default=CLIP,
                    help="robust-scale features and clip to +/- this many IQRs before "
                         "reduction (0 disables; needed so PCA/Isomap aren't dominated "
                         "by a few extreme feature values).")
    args = ap.parse_args()

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if args.source:
        label_df = label_df[label_df.source_dataset == args.source].copy()
        print(f"source filter '{args.source}': {len(label_df)} subjects", flush=True)

    cohorts = COHORT_GROUPS[args.cohort_group]
    keys = [args.cohort] if args.cohort else list(cohorts)
    for key in keys:
        subdir, mask_fn = cohorts[key]
        run_cohort(key, subdir, mask_fn, label_df, args.out_dir, args.condition,
                   reject=not args.no_reject, mad_z=args.mad_z,
                   outlier_frac=args.outlier_frac,
                   clip=(args.clip if args.clip and args.clip > 0 else None))


if __name__ == "__main__":
    main()
