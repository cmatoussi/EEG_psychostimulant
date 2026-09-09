"""
Predict DRUG RESISTANCE (asm_resistant) from the pre-extracted FM embeddings,
averaged-epoch level (one point per subject = the *_subject_embeddings.csv),
with classical heads. Population = epilepsy+ subjects only (resistance is only
defined there: resistant vs drug-responsive epilepsy).

Two source sets, saved in separate subfolders:
  both_sources     - all epilepsy+ (972: 406 resistant, of which 314 are the
                     drug_resistant STUDY). asm_resistant is ~77% that study, so
                     this AUC is inflated by the source confound (the model can
                     read the study from the embeddings, AUC~0.75).
  adhd_only_source - epilepsy+ within the adhd study only (658: 92 resistant vs
                     566 responsive). The honest within-study resistance signal.

Usage:
    python run_drug_resistance.py --out-dir /home/mat/scratch/results/embedding_performance/drug_resistance
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402

LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
LEVEL = "epoch_level"  # subject embeddings are 1 row/subject, so this IS per-subject
N_SPLITS = 5
MODELS = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
CONDITIONS = ["EO_baseline", "EC_baseline"]
METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "f1"]

HEADS = {
    "logreg": {"method": "LogisticRegression", "max_iter": 500, "class_weight": "balanced"},
    "rf":     {"method": "RandomForestClassifier", "n_estimators": 200, "class_weight": "balanced"},
    "svm":    {"method": "SVC", "kernel": "rbf", "probability": True, "class_weight": "balanced"},
    "histgb": {"method": "HistGradientBoostingClassifier", "class_weight": "balanced"},
    "dummy":  {"method": "DummyClassifier", "strategy": "stratified", "random_state": 42},
}

# source set -> source_dataset filter (None = keep both studies)
SETS = {"both_sources": None, "adhd_only_source": "adhd"}

COLUMNS = [
    "source_set", "fm_model", "condition", "head", "status",
    "n_subjects", "n_resistant", "n_responsive",
    "accuracy_mean", "accuracy_std",
    "balanced_accuracy_mean", "balanced_accuracy_std",
    "roc_auc_mean", "roc_auc_std",
    "f1_mean", "f1_std",
]


def _cond_short(c):
    return c.replace("_baseline", "")


def _empty_metrics():
    return {f"{m}_{s}": np.nan for m in METRICS for s in ("mean", "std")}


def run_set(set_name, src_filter, epi_df, out_root):
    set_df = epi_df if src_filter is None else epi_df[epi_df.source_dataset == src_filter].copy()
    out_dir = Path(out_root) / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    run_root = out_dir / "runs"
    rows = []
    n_r = int((set_df.asm_resistant == 1).sum())
    print(f"=== {set_name}: {set_df['study_id'].nunique()} epilepsy+ subjects "
          f"({n_r} resistant, {len(set_df) - n_r} responsive) ===", flush=True)

    def _flush():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

    for model in MODELS:
        for cond in CONDITIONS:
            cs = _cond_short(cond)
            base = {"source_set": set_name, "fm_model": model, "condition": cs, "head": None}
            acfg = {
                "model_key": model, "target_col": "asm_resistant", "embedding_level": "subject",
                "models": HEADS,
                "cv": {"strategy": "stratified_group_kfold", "n_splits": N_SPLITS},
                "metrics": METRICS,
            }
            try:
                X, y, groups = ra.load_precomputed_embeddings(
                    acfg, {"paths": {}}, set_df, {"conditions": [cond]})
            except FileNotFoundError as e:
                print(f"  {model}/{cs}: MISSING ({e})", flush=True)
                for h in HEADS:
                    rows.append({**base, "head": h, "status": "missing_embeddings",
                                 "n_subjects": 0, "n_resistant": 0, "n_responsive": 0,
                                 **_empty_metrics()})
                _flush(); continue

            counts = {"n_subjects": int(len(np.unique(groups))),
                      "n_resistant": int((y == 1).sum()), "n_responsive": int((y == 0).sum())}
            if len(np.unique(y)) < 2 or counts["n_subjects"] < N_SPLITS:
                print(f"  {model}/{cs}: SKIP (subjects={counts['n_subjects']})", flush=True)
                for h in HEADS:
                    rows.append({**base, "head": h, "status": "skipped_insufficient",
                                 **counts, **_empty_metrics()})
                _flush(); continue

            run_root.mkdir(parents=True, exist_ok=True)
            try:
                ra.run_embed_head(acfg, X, y, groups, run_root, result_name=f"{model}_{cs}")
                posthoc = json.loads(
                    (run_root / f"{model}_{cs}_posthoc_metrics.json").read_text())["metrics"]
            except Exception as e:  # noqa: BLE001
                print(f"  {model}/{cs}: ERROR {e}", flush=True)
                for h in HEADS:
                    rows.append({**base, "head": h, "status": "error", **counts, **_empty_metrics()})
                _flush(); continue

            for h in HEADS:
                lvl = posthoc.get(h, {}).get(LEVEL, {})
                mc = {}
                for m in METRICS:
                    mean = lvl.get(m, {}).get("mean", np.nan) if lvl else np.nan
                    std = lvl.get(m, {}).get("std", np.nan) if lvl else np.nan
                    mc[f"{m}_mean"] = round(mean, 4) if mean == mean else np.nan
                    mc[f"{m}_std"] = round(std, 4) if std == std else np.nan
                status = "success" if mc["roc_auc_mean"] == mc["roc_auc_mean"] else "degenerate"
                rows.append({**base, "head": h, "status": status, **counts, **mc})
                print(f"  {model}/{cs}/{h}: roc_auc={mc['roc_auc_mean']} "
                      f"bal_acc={mc['balanced_accuracy_mean']} (n={counts['n_subjects']})", flush=True)
            _flush()

    _flush()
    print(f"--> wrote {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if "source_dataset" not in label_df.columns:
        raise SystemExit("label CSV has no 'source_dataset' column; use patients_metadata_clean.csv.")
    label_df["asm_resistant"] = pd.to_numeric(
        label_df["asm_resistant"], errors="coerce").fillna(0).astype(int)
    epi_df = label_df[label_df.epilepsy == 1].copy()  # resistance only defined for epilepsy+

    for set_name, src in SETS.items():
        run_set(set_name, src, epi_df, args.out_dir)


if __name__ == "__main__":
    main()
