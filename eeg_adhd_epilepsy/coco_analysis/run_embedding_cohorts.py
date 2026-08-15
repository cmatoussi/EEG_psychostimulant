"""
Cohort epilepsy decoding from pre-extracted FM embeddings + classical heads.

Generalized, CLI-driven version of the comorbidity sweep so a single cohort can
run as one SLURM job. Uses the ALREADY-EXTRACTED embeddings (no extraction).

Usage:
    python run_embedding_cohorts.py --cohort-group sex --cohort F --out-dir <dir>
    python run_embedding_cohorts.py --cohort-group comorbidity            # all cohorts
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402
import cohort_balance  # noqa: E402

LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
LEVEL     = "epoch_level"
N_SPLITS  = 5
# All models with extracted embeddings; lighter dims first, eegpt (2048-d) last.
MODELS     = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
CONDITIONS = ["EO_baseline", "EC_baseline"]

HEADS = {
    "logreg": {"method": "LogisticRegression", "max_iter": 500, "class_weight": "balanced"},
    "rf":     {"method": "RandomForestClassifier", "n_estimators": 200, "class_weight": "balanced"},
    "svm":    {"method": "SVC", "kernel": "rbf", "probability": True, "class_weight": "balanced"},
    "histgb": {"method": "HistGradientBoostingClassifier", "class_weight": "balanced"},
    "dummy":  {"method": "DummyClassifier", "strategy": "stratified", "random_state": 42},
}

# cohort-group -> {cohort key: (csv name, mask fn on normalized label_df)}
COHORT_GROUPS = {
    "comorbidity": {
        "no_comorbidity":    ("results_embedding_no_comorbidity.csv",    lambda d: (d.autism == 0) & (d.adhd == 0)),
        "with_asd":          ("results_embedding_with_asd.csv",          lambda d: (d.autism == 1) & (d.adhd == 0)),
        "with_adhd":         ("results_embedding_with_adhd.csv",         lambda d: (d.autism == 0) & (d.adhd == 1)),
        "with_adhd_and_asd": ("results_embedding_with_adhd_and_asd.csv", lambda d: (d.autism == 1) & (d.adhd == 1)),
    },
    "sex": {
        "F":   ("results_embedding_female.csv", lambda d: d.sex == "F"),
        "M":   ("results_embedding_male.csv",   lambda d: d.sex == "M"),
        "ALL": ("results_embedding_all.csv",    lambda d: pd.Series(True, index=d.index)),
    },
    "age": {  # uses the pre-binned age_group column
        "0-4":   ("results_embedding_age_0_4.csv",   lambda d: d.age_group == "0-4"),
        "5-8":   ("results_embedding_age_5_8.csv",   lambda d: d.age_group == "5-8"),
        "9-12":  ("results_embedding_age_9_12.csv",  lambda d: d.age_group == "9-12"),
        "13-18": ("results_embedding_age_13_18.csv", lambda d: d.age_group == "13-18"),
    },
}

COLUMNS = [
    "fm_model", "condition", "aggregation", "head", "status", "n_subjects", "n_windows",
    "adhd", "epilepsy", "autism",
    "accuracy_mean", "accuracy_std",
    "balanced_accuracy_mean", "balanced_accuracy_std",
    "balanced_accuracy_optimal_mean", "balanced_accuracy_optimal_std",
    "f1_mean", "f1_std",
    "roc_auc_mean", "roc_auc_std",
]
_METRICS = ["accuracy", "balanced_accuracy", "balanced_accuracy_optimal", "f1", "roc_auc"]

# Subject-level aggregation modes:
#   averaged_predictions - score each epoch, then average the per-epoch
#       probabilities into one prediction per subject (late / output-level fusion).
#   averaged_epochs - average the epoch EMBEDDINGS into one vector per subject
#       (the pre-computed *_subject_embeddings.csv), then predict once (early /
#       input-level fusion, in embedding space so oscillations don't cancel).
# They coincide only for a linear head; differ for any nonlinear head (Jensen).
AGG = {
    "averaged_predictions": {"embedding_level": "epoch",   "metric_level": "subject_level"},
    "averaged_epochs":      {"embedding_level": "subject", "metric_level": "epoch_level"},
}


def _cond_short(c):
    return c.replace("_baseline", "")


def _empty_metrics():
    return {f"{m}_{s}": np.nan for m in _METRICS for s in ("mean", "std")}


def run_one_cohort(cohort_key, csv_name, mask_fn, label_df, out_dir,
                   cv_strategy="stratified_group_kfold",
                   aggregations=("averaged_predictions",), epoch_out_dir=None,
                   target_col="epilepsy", cohort_group=None, balanced=False):
    cohort_df = label_df[mask_fn(label_df)].copy()
    if balanced:
        cohort_df, n_bal, drop_reason = cohort_balance.build_balanced(
            cohort_df, target_col, cohort_group)
        if drop_reason:
            print(f"=== cohort {cohort_key}: BALANCED SKIP ({drop_reason}) ===", flush=True)
            return
        print(f"=== cohort {cohort_key}: balanced to {n_bal} subjects "
              f"(sex+age matched, target={target_col}) ===", flush=True)
    run_root = Path(out_dir) / "runs" / cohort_key
    out_csv = Path(out_dir) / csv_name
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    # Optional epoch-level table: the SAME runs already carry an `epoch_level`
    # posthoc block (score each epoch, no per-subject pooling). We surface it only
    # for averaged_predictions (which fits on epochs); averaged_epochs pre-averages
    # so its "epoch_level" is really subject-level and is skipped.
    erows = [] if epoch_out_dir else None
    eout_csv = Path(epoch_out_dir) / csv_name if epoch_out_dir else None
    if epoch_out_dir:
        Path(epoch_out_dir).mkdir(parents=True, exist_ok=True)
    print(f"=== cohort {cohort_key}: {cohort_df['study_id'].nunique()} subjects ===", flush=True)

    def _flush():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)
        if erows is not None:
            pd.DataFrame(erows, columns=COLUMNS).to_csv(eout_csv, index=False)

    for model in MODELS:
        for cond in CONDITIONS:
            cond_s = _cond_short(cond)
            for agg in aggregations:
                base = {"fm_model": model, "condition": cond_s, "aggregation": agg}
                emb_level = AGG[agg]["embedding_level"]
                metric_level = AGG[agg]["metric_level"]
                acfg = {
                    "model_key": model, "target_col": target_col, "embedding_level": emb_level,
                    "models": HEADS,
                    "cv": {"strategy": cv_strategy, "n_splits": N_SPLITS},
                    "metrics": ["accuracy", "roc_auc", "balanced_accuracy", "f1"],
                }
                try:
                    X, y, groups = ra.load_precomputed_embeddings(
                        acfg, {"paths": {}}, cohort_df, {"conditions": [cond]}
                    )
                except FileNotFoundError as e:
                    print(f"  {model}/{cond_s}/{agg}: MISSING ({e})", flush=True)
                    for h in HEADS:
                        rows.append({**base, "head": h, "status": "missing_embeddings",
                                     "n_subjects": 0, "n_windows": 0, "adhd": 0, "epilepsy": 0,
                                     "autism": 0, **_empty_metrics()})
                    _flush()
                    continue

                present = set(groups.tolist())
                sub = cohort_df[cohort_df["study_id"].astype(str).isin(present)].drop_duplicates("study_id")
                counts = {
                    "n_subjects": len(present), "n_windows": int(X.shape[0]),
                    "adhd": int((sub.adhd == 1).sum()),
                    "epilepsy": int((sub.epilepsy == 1).sum()),
                    "autism": int((sub.autism == 1).sum()),
                }
                if len(np.unique(y)) < 2 or len(present) < N_SPLITS:
                    print(f"  {model}/{cond_s}/{agg}: SKIP (subjects={len(present)})", flush=True)
                    for h in HEADS:
                        rows.append({**base, "head": h, "status": "skipped_insufficient",
                                     **counts, **_empty_metrics()})
                    _flush()
                    continue

                run_root.mkdir(parents=True, exist_ok=True)
                try:
                    ra.run_embed_head(acfg, X, y, groups, run_root,
                                      result_name=f"{model}_{cond_s}_{agg}")
                    posthoc = json.loads(
                        (run_root / f"{model}_{cond_s}_{agg}_posthoc_metrics.json").read_text()
                    )["metrics"]
                except Exception as e:  # noqa: BLE001
                    print(f"  {model}/{cond_s}/{agg}: ERROR {e}", flush=True)
                    for h in HEADS:
                        rows.append({**base, "head": h, "status": "error", **counts, **_empty_metrics()})
                    _flush()
                    continue

                for h in HEADS:
                    lvl = posthoc.get(h, {}).get(metric_level, {})
                    mcols = {}
                    for m in _METRICS:
                        mean = lvl.get(m, {}).get("mean", np.nan) if lvl else np.nan
                        std = lvl.get(m, {}).get("std", np.nan) if lvl else np.nan
                        mcols[f"{m}_mean"] = round(mean, 4) if mean == mean else np.nan
                        mcols[f"{m}_std"] = round(std, 4) if std == std else np.nan
                    status = "success" if mcols["roc_auc_mean"] == mcols["roc_auc_mean"] else "degenerate"
                    rows.append({**base, "head": h, "status": status, **counts, **mcols})
                    # epoch-level row (per-epoch scoring) from the same run's posthoc
                    if erows is not None and agg == "averaged_predictions":
                        elvl = posthoc.get(h, {}).get("epoch_level", {})
                        emcols = {}
                        for m in _METRICS:
                            em = elvl.get(m, {}).get("mean", np.nan) if elvl else np.nan
                            es = elvl.get(m, {}).get("std", np.nan) if elvl else np.nan
                            emcols[f"{m}_mean"] = round(em, 4) if em == em else np.nan
                            emcols[f"{m}_std"] = round(es, 4) if es == es else np.nan
                        estatus = "success" if emcols["roc_auc_mean"] == emcols["roc_auc_mean"] else "degenerate"
                        erows.append({**base, "aggregation": "per_epoch", "head": h,
                                      "status": estatus, **counts, **emcols})
                    print(f"  {model}/{cond_s}/{agg}/{h}: roc_auc={mcols['roc_auc_mean']} "
                          f"bal_acc_opt={mcols['balanced_accuracy_optimal_mean']} "
                          f"(subj={counts['n_subjects']}, win={counts['n_windows']})", flush=True)
                _flush()  # incremental save after each model x condition x aggregation

    _flush()
    print(f"--> wrote {len(rows)} rows to {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-group", required=True, choices=list(COHORT_GROUPS))
    ap.add_argument("--cohort", default=None, help="single cohort key; omit for all in the group")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cv-strategy", default="stratified_group_kfold",
                    help="CV strategy passed to coco_pipe (e.g. kfold, group_kfold, "
                         "stratified_group_kfold). Non-grouped strategies avoid "
                         "single-class test folds but leak subjects across folds.")
    ap.add_argument("--source", default=None,
                    help="restrict to a single source_dataset (e.g. 'adhd') to remove "
                         "the study/source confound where epilepsy is entangled with provenance.")
    ap.add_argument("--label-csv", default=LABEL_CSV,
                    help="metadata CSV to use (default: patients_metadata_clean.csv).")
    ap.add_argument("--epoch-out-dir", default=None,
                    help="if set, also write an epoch-level (per-epoch, no subject pooling) "
                         "table here, from the same runs' epoch_level posthoc block.")
    ap.add_argument("--skip", nargs="*", default=["age_0_4", "with_asd"],
                    help="cohort keys to skip (small/undecodable). Default omits age_0_4 and with_asd.")
    ap.add_argument("--target-col", default="epilepsy",
                    help="binary label column to decode (e.g. epilepsy, asm_resistant).")
    ap.add_argument("--aggregation", default="both",
                    choices=["averaged_predictions", "averaged_epochs", "both"],
                    help="subject-level aggregation: averaged_predictions (score epochs, "
                         "average probabilities), averaged_epochs (average the embeddings, "
                         "predict once), or both for a side-by-side comparison.")
    ap.add_argument("--balanced", action="store_true",
                    help="uniform sex x age (x comorbidity, where free) matched "
                         "case/control cohort instead of the natural baseline population; "
                         "cohorts that can't reach 30 matched subjects are skipped.")
    args = ap.parse_args()
    aggregations = list(AGG) if args.aggregation == "both" else [args.aggregation]

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if "source_dataset" not in label_df.columns:
        print("note: label CSV has no source_dataset column (dropped upstream).", flush=True)
    if args.source:
        if "source_dataset" not in label_df.columns:
            raise SystemExit("--source given but no 'source_dataset' column in label CSV.")
        n0 = len(label_df)
        label_df = label_df[label_df.source_dataset == args.source].copy()
        print(f"source filter '{args.source}': {n0} -> {len(label_df)} subjects", flush=True)
    if args.target_col == "asm_resistant":
        label_df = label_df[label_df.epilepsy == 1].copy()
        print(f"asm_resistant target: restricted to epilepsy==1 -> {len(label_df)} subjects", flush=True)
    cohorts = COHORT_GROUPS[args.cohort_group]
    keys = [args.cohort] if args.cohort else list(cohorts)
    for key in keys:
        if key not in cohorts:
            raise SystemExit(f"Unknown cohort {key!r}; options: {list(cohorts)}")
        if key in (args.skip or []):
            print(f"skip cohort {key!r} (in --skip)", flush=True)
            continue
        csv_name, mask_fn = cohorts[key]
        run_one_cohort(key, csv_name, mask_fn, label_df, args.out_dir,
                       cv_strategy=args.cv_strategy, aggregations=aggregations,
                       epoch_out_dir=args.epoch_out_dir, target_col=args.target_col,
                       cohort_group=args.cohort_group, balanced=args.balanced)


if __name__ == "__main__":
    main()
