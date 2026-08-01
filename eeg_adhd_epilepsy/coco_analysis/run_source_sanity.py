"""
Source-confound sanity check: predict the SOURCE STUDY (adhd vs drug_resistant)
from the pre-extracted FM embeddings, with a logistic-regression head.

This is the same load -> head -> grouped CV machinery as the epilepsy embedding
sweep (run_embedding_cohorts.py); only the predicted variable changes to
`source_dataset`. A high AUC means the embeddings carry a study/site fingerprint
(the confound); ~0.5 means they don't.

Two scopes:
  epilepsy_only  - only epilepsy+ subjects (658 adhd + 314 drug_resistant). The
                   honest measure: both studies contribute the same kind of
                   patient, so a high score is a genuine site signal.
  all_subjects   - everyone; inflated because every non-epilepsy control is
                   adhd-source, so the model can partly just re-detect epilepsy.

Usage:
    python run_source_sanity.py --out-csv /home/mat/scratch/results/source_sanity_check.csv
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
LEVEL = "epoch_level"
N_SPLITS = 5
MODELS = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
CONDITIONS = ["EO_baseline", "EC_baseline"]
METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "f1"]

HEADS = {"logreg": {"method": "LogisticRegression", "max_iter": 500,
                    "class_weight": "balanced"}}

# scope -> mask on the normalized label_df
SCOPES = {
    "epilepsy_only": lambda d: d.Epilepsy == 1,
    "all_subjects":  lambda d: pd.Series(True, index=d.index),
}

COLUMNS = [
    "scope", "fm_model", "condition", "head", "status",
    "n_subjects", "n_windows", "n_adhd", "n_drugres",
    "accuracy_mean", "accuracy_std",
    "balanced_accuracy_mean", "balanced_accuracy_std",
    "roc_auc_mean", "roc_auc_std",
    "f1_mean", "f1_std",
]


def _cond_short(c):
    return c.replace("_baseline", "")


def _empty_metrics():
    return {f"{m}_{s}": np.nan for m in METRICS for s in ("mean", "std")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if "source_dataset" not in label_df.columns:
        raise SystemExit("label CSV has no 'source_dataset' column; use "
                         "patients_metadata_clean.csv (not the without_source copy).")
    # binary target: adhd = 0, drug_resistant = 1
    label_df["SourceBin"] = (label_df.source_dataset == "drug_resistant").astype(int)
    src_lut = (label_df.drop_duplicates("Study ID")
               .set_index(label_df["Study ID"].astype(str))["SourceBin"])

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def _flush():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)

    for scope, mask_fn in SCOPES.items():
        scope_df = label_df[mask_fn(label_df)].copy()
        print(f"=== scope {scope}: {scope_df['Study ID'].nunique()} subjects "
              f"({int((scope_df.SourceBin==0).sum())} adhd, "
              f"{int((scope_df.SourceBin==1).sum())} drug_resistant) ===", flush=True)
        for model in MODELS:
            for cond in CONDITIONS:
                cs = _cond_short(cond)
                base = {"scope": scope, "fm_model": model, "condition": cs, "head": "logreg"}
                acfg = {
                    "model_key": model, "target_col": "SourceBin", "embedding_level": "epoch",
                    "models": HEADS,
                    "cv": {"strategy": "stratified_group_kfold", "n_splits": N_SPLITS},
                    "metrics": METRICS,
                }
                try:
                    X, y, groups = ra.load_precomputed_embeddings(
                        acfg, {"paths": {}}, scope_df, {"conditions": [cond]})
                except FileNotFoundError as e:
                    print(f"  {model}/{cs}: MISSING ({e})", flush=True)
                    rows.append({**base, "status": "missing_embeddings", "n_subjects": 0,
                                 "n_windows": 0, "n_adhd": 0, "n_drugres": 0, **_empty_metrics()})
                    _flush(); continue

                present = np.unique(groups)
                n_adhd = int(sum(src_lut.get(str(g), -1) == 0 for g in present))
                n_dr = int(sum(src_lut.get(str(g), -1) == 1 for g in present))
                counts = {"n_subjects": len(present), "n_windows": int(X.shape[0]),
                          "n_adhd": n_adhd, "n_drugres": n_dr}
                if len(np.unique(y)) < 2 or len(present) < N_SPLITS:
                    print(f"  {model}/{cs}: SKIP (subjects={len(present)}, "
                          f"classes={np.unique(y).tolist()})", flush=True)
                    rows.append({**base, "status": "skipped_insufficient", **counts,
                                 **_empty_metrics()})
                    _flush(); continue

                run_root = out.parent / "source_sanity_runs" / scope
                run_root.mkdir(parents=True, exist_ok=True)
                try:
                    ra.run_embed_head(acfg, X, y, groups, run_root,
                                      result_name=f"{model}_{cs}")
                    posthoc = json.loads(
                        (run_root / f"{model}_{cs}_posthoc_metrics.json").read_text())["metrics"]
                except Exception as e:  # noqa: BLE001
                    print(f"  {model}/{cs}: ERROR {e}", flush=True)
                    rows.append({**base, "status": "error", **counts, **_empty_metrics()})
                    _flush(); continue

                lvl = posthoc.get("logreg", {}).get(LEVEL, {})
                mc = {}
                for m in METRICS:
                    mean = lvl.get(m, {}).get("mean", np.nan) if lvl else np.nan
                    std = lvl.get(m, {}).get("std", np.nan) if lvl else np.nan
                    mc[f"{m}_mean"] = round(mean, 4) if mean == mean else np.nan
                    mc[f"{m}_std"] = round(std, 4) if std == std else np.nan
                status = "success" if mc["roc_auc_mean"] == mc["roc_auc_mean"] else "degenerate"
                rows.append({**base, "status": status, **counts, **mc})
                print(f"  {model}/{cs}: roc_auc={mc['roc_auc_mean']} "
                      f"bal_acc={mc['balanced_accuracy_mean']} "
                      f"(subj={counts['n_subjects']}, win={counts['n_windows']})", flush=True)
                _flush()

    _flush()
    print(f"--> wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
