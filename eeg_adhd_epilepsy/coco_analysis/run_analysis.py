import argparse
import csv
import fcntl
import json
import logging
import yaml
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from coco_pipe.decoding.configs import (
    CVConfig,
    ClassicalModelConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    FoundationEmbeddingModelConfig,
    FrozenBackboneDecoderConfig,
    LoRAConfig,
    NeuralFineTuneConfig,
    TrainerConfig,
)
from coco_pipe.decoding.experiment import Experiment
from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models import FoundationEmbeddingExtractor
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.io.embeddings import (
    load_embedding_derivatives,
    save_embedding_derivative,
    write_embedding_dataset_description,
    write_embedding_manifest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_FIELDNAMES = [
    "model", "condition", "sex", "age", "comorbidities", "medication",
    "status", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean",
    "balanced_accuracy_std", "balanced_accuracy_optimal_mean", "balanced_accuracy_optimal_std",
    "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
]


EMB_SUMMARY_FIELDNAMES = [
    "fm_model", "condition", "head", "status", "n_subjects", "n_windows",
    "accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "balanced_accuracy_std",
    "balanced_accuracy_optimal_mean", "balanced_accuracy_optimal_std",
    "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
]


def _parse_group_id(group_id: str) -> dict:
    """Parse 'sex-ALL_age-ALL_com-ALL_med-ALL_cond-EO' into cohort fields."""
    raw = {}
    for chunk in group_id.split("_"):
        key, val = chunk.split("-", 1)
        raw[key] = val
    return {
        "sex": raw.get("sex", ""),
        "age": raw.get("age", ""),
        "comorbidities": raw.get("com", ""),
        "medication": raw.get("med", ""),
        "condition": raw.get("cond", ""),
    }


def _rows_from_result_json(json_path: Path, cohort_fields: dict) -> list:
    """Build summary rows from a saved results.json file."""
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for model_key, model_data in data.get("results", {}).items():
        if model_data.get("status") == "failed":
            continue
        row = {"model": model_key, "status": model_data.get("status", "unknown"), **cohort_fields}
        for metric_name, metric_vals in model_data.get("metrics", {}).items():
            row[f"{metric_name}_mean"] = metric_vals.get("mean")
            row[f"{metric_name}_std"] = metric_vals.get("std")
        rows.append(row)
    return rows


def _append_to_summary(summary_csv: Path, rows: list, extra_fields: list = None) -> None:
    """Append result rows to the shared summary CSV with an exclusive file lock."""
    fieldnames = SUMMARY_FIELDNAMES + (extra_fields or [])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)  # re-check actual end after acquiring lock (race-condition guard)
            write_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _append_embeddings_summary(
    summary_csv: Path,
    json_path: Path,
    fm_model: str,
    condition: str,
    n_subjects: int,
    n_windows: int,
) -> int:
    """Append one row per CV head from a head-results JSON to the embeddings summary."""
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for head_name, head_data in data.get("results", {}).items():
        if head_data.get("status") == "failed":
            continue
        row = {
            "fm_model": fm_model,
            "condition": condition,
            "head": head_name,
            "status": head_data.get("status", "unknown"),
            "n_subjects": n_subjects,
            "n_windows": n_windows,
        }
        for metric_name, metric_vals in head_data.get("metrics", {}).items():
            row[f"{metric_name}_mean"] = metric_vals.get("mean")
            row[f"{metric_name}_std"] = metric_vals.get("std")
        rows.append(row)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            write_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=EMB_SUMMARY_FIELDNAMES, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return len(rows)


REGION_MAP = {
    "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Temporal": ["T3", "T4", "T5", "T6"],
    "Parietal": ["P3", "Pz", "P4"],
    "Occipital": ["O1", "O2"],
}


def map_sensor_to_region(X_df: pd.DataFrame, region: str) -> pd.DataFrame:
    if region not in REGION_MAP:
        raise ValueError(f"Unknown region '{region}'. Valid: {list(REGION_MAP)}")
    sensors = REGION_MAP[region]
    keep = [c for c in X_df.columns if any(c.startswith(f"{s}_") for s in sensors)]
    return X_df[keep]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _undersample_majority(X, y, groups, seed=42):
    """Random undersampling: trim the majority class down to minority class size."""
    classes, counts = np.unique(y, return_counts=True)
    minority_n = counts.min()
    rng = np.random.default_rng(seed)
    keep = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        keep.append(rng.choice(idx, size=minority_n, replace=False))
    keep = np.sort(np.concatenate(keep))
    g = groups[keep] if groups is not None else None
    return X[keep], y[keep], g


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

# Map the shared patients_metadata_clean.csv (lowercase) schema onto the
# canonical column names the loaders and cohort logic expect.
# Canonical schema is the lowercase BIDS naming. These map any legacy-schema
# columns to the canonical names, so old CSVs still load; lowercase CSVs pass
# through untouched.
_LABEL_COL_ALIASES = {
    "Study ID": "study_id",
    "Epilepsy": "epilepsy",
    "TSA": "autism",             # ASD
    "TDAH": "adhd",              # ADHD
    "Sex": "sex",
    "Age": "age",
    "Psychostimulant (y/n)": "psychostimulant",
}


def normalize_label_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename metadata columns to canonical names, only where the alias exists
    and the canonical name isn't already present. Leaves legacy CSVs untouched."""
    rename = {
        src: dst for src, dst in _LABEL_COL_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    return df.rename(columns=rename) if rename else df


def load_handcrafted_data(
    analysis_cfg: dict, label_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature CSV and align it to the cohort subjects in label_df."""
    data_path = analysis_cfg["data_path"]
    target_col = analysis_cfg.get("target_col", "epilepsy")
    subject_col = "study_id"

    feat_df = pd.read_csv(data_path)
    feat_df[subject_col] = feat_df[subject_col].astype(str)
    label_df = label_df.copy()
    label_df[subject_col] = label_df[subject_col].astype(str)

    # Merge to get cohort-filtered rows with labels
    label_sub = label_df[[subject_col, target_col]].drop_duplicates(subset=[subject_col])
    feat_df = feat_df.merge(label_sub, on=subject_col, how="inner", suffixes=("", "_lbl"))
    if f"{target_col}_lbl" in feat_df.columns:
        feat_df[target_col] = feat_df[f"{target_col}_lbl"]
        feat_df = feat_df.drop(columns=[f"{target_col}_lbl"])

    y = feat_df[target_col].astype(int).values
    groups = feat_df[subject_col].values

    # Drop non-feature columns
    drop_cols = {subject_col, target_col, "sex", "age", "autism", "adhd"}
    X_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])

    # Optional spatial / region slicing
    analysis_unit = analysis_cfg.get("analysis_unit", "all")
    spatial_units = analysis_cfg.get("spatial_units", "all")
    if analysis_unit == "region" and spatial_units != "all":
        X_df = map_sensor_to_region(X_df, spatial_units)
    elif analysis_unit == "sensor" and isinstance(spatial_units, list):
        X_df = X_df[[c for c in X_df.columns if any(c.startswith(f"{s}_") for s in spatial_units)]]

    return X_df.values.astype(np.float32), y, groups


# Identifier/metadata columns to exclude when no feature-column sidecar exists.
_DIMRED_META_COLS = {
    "epoch_count", "subject", "session", "run", "recording_id", "condition",
    "source_dataset", "study_id", "patient_id", "patient_group_id", "eeg_date",
    "first_eeg", "age", "age_group", "sex", "adhd", "autism", "epilepsy",
    "combined_diagnosis", "psychostimulant",
    "psychostimulant_description", "psychostimulant_category",
}
# Candidate subject-id column names across schema versions.
_DIMRED_SUBJECT_COLS = ["study_id", "subject"]


def _dimred_feature_columns(data_path: str, present_cols: list[str]) -> list[str] | None:
    """Return the feature-column list from the `*_feature_columns.json` sidecar
    next to the CSV, intersected with columns actually present. None if absent."""
    sidecar = Path(str(data_path).rsplit(".", 1)[0] + "_feature_columns.json")
    if not sidecar.exists():
        return None
    names = json.load(open(sidecar))
    present = set(present_cols)
    return [c for c in names if c in present]


def load_dim_reduction_data(
    analysis_cfg: dict, label_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the feature CSV for dim reduction: filter to one condition, keep only
    feature columns. Returns (X, y, groups, feature_names).

    Feature columns come from the ``*_feature_columns.json`` sidecar when present
    (robust to schema changes); otherwise from a numeric-dtype selection minus a
    metadata blocklist. The subject feature CSVs hold one row per subject *per
    recording condition*, so ``condition`` filters to a single recording.
    """
    data_path = analysis_cfg["data_path"]
    target_col = analysis_cfg.get("target_col", "epilepsy")
    condition = analysis_cfg.get("condition")  # e.g. "EO_baseline"; None = all

    feat_df = pd.read_csv(data_path, low_memory=False)

    # Resolve the subject-id column across schema versions.
    subject_col = next((c for c in _DIMRED_SUBJECT_COLS if c in feat_df.columns), None)
    if subject_col is None:
        raise ValueError(
            f"No subject-id column found in {data_path}. "
            f"Looked for {_DIMRED_SUBJECT_COLS}; have {list(feat_df.columns[:15])}..."
        )
    feat_df[subject_col] = feat_df[subject_col].astype(str)

    # Filter to a single recording condition if requested and present.
    if condition is not None and "condition" in feat_df.columns:
        avail = sorted(feat_df["condition"].unique())
        feat_df = feat_df[feat_df["condition"] == condition]
        if feat_df.empty:
            raise ValueError(f"No rows for condition={condition!r}. Available: {avail}")

    # Attach labels from the cohort table (label_df uses 'study_id').
    label_df = label_df.copy()
    label_df["study_id"] = label_df["study_id"].astype(str)
    label_sub = label_df[["study_id", target_col]].drop_duplicates(subset=["study_id"])
    feat_df = feat_df.merge(
        label_sub, left_on=subject_col, right_on="study_id", how="inner",
        suffixes=("", "_lbl"),
    )
    if f"{target_col}_lbl" in feat_df.columns:
        feat_df[target_col] = feat_df[f"{target_col}_lbl"]
        feat_df = feat_df.drop(columns=[f"{target_col}_lbl"])

    y = feat_df[target_col].astype(int).values
    groups = feat_df[subject_col].values

    # Select feature columns: prefer the sidecar, else blocklist + numeric dtype.
    feature_cols = _dimred_feature_columns(data_path, list(feat_df.columns))
    if feature_cols:
        X_df = feat_df[feature_cols]
    else:
        X_df = feat_df.drop(columns=[c for c in _DIMRED_META_COLS if c in feat_df.columns])

    # Optional spatial / region slicing (same semantics as handcrafted mode).
    analysis_unit = analysis_cfg.get("analysis_unit", "all")
    spatial_units = analysis_cfg.get("spatial_units", "all")
    if analysis_unit == "region" and spatial_units != "all":
        X_df = map_sensor_to_region(X_df, spatial_units)
    elif analysis_unit == "sensor" and isinstance(spatial_units, list):
        X_df = X_df[[c for c in X_df.columns if any(c.startswith(f"{s}_") for s in spatial_units)]]

    # Keep only numeric columns; drop all-NaN columns.
    X_df = X_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    X = X_df.values.astype(np.float32)
    # Impute residual NaNs with per-column means (reducers reject NaNs).
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        X = np.where(np.isnan(X), col_mean, X)

    logger.info(
        f"dim_reduction input: {X.shape[0]} samples x {X.shape[1]} features "
        f"(subject_col={subject_col}, condition={condition}, "
        f"classes={np.unique(y, return_counts=True)})"
    )
    return X, y, groups, list(X_df.columns)


# Default location of the pre-extracted foundation-model embeddings.
_EMBEDDINGS_ROOT = (
    "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
    "signal_features/eeg_foundation_embeddings/combined"
)


def load_precomputed_embeddings(
    analysis_cfg: dict, config: dict, label_df: pd.DataFrame, signal_cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted FM embeddings for the chosen model + condition.

    File layout: ``{embeddings_root}/{model}_{condition}_{level}_embeddings.csv``
    with ``level`` in {epoch, recording, subject}. Columns are meta
    (subject/session/run/condition/recording_id/model_key/window_index) plus
    ``embedding_0000..N``. Returns (X, y, groups) ready for a classical head;
    no extraction is performed.
    """
    model_key = analysis_cfg["model_key"]
    target_col = analysis_cfg.get("target_col", "epilepsy")
    level = analysis_cfg.get("embedding_level", "epoch")  # epoch | recording | subject
    emb_root = Path(config.get("paths", {}).get("embeddings_root", _EMBEDDINGS_ROOT))

    conditions = signal_cfg.get("conditions") or ["EO_baseline"]
    condition = conditions[0]  # one condition per job

    emb_path = emb_root / f"{model_key}_{condition}_{level}_embeddings.csv"
    if not emb_path.exists():
        avail = sorted(p.name for p in emb_root.glob(f"{model_key}_*_{level}_embeddings.csv"))
        raise FileNotFoundError(
            f"No embeddings at {emb_path}. Available for {model_key}/{level}: {avail}"
        )

    df = pd.read_csv(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    if not emb_cols:
        raise ValueError(f"No 'embedding_*' columns in {emb_path}; got {list(df.columns)[:10]}")

    # Attach labels: embeddings 'subject' <-> cohort 'study_id'.
    df["subject"] = df["subject"].astype(str)
    label_df = label_df.copy()
    label_df["study_id"] = label_df["study_id"].astype(str)
    label_sub = label_df[["study_id", target_col]].drop_duplicates(subset=["study_id"])
    df = df.merge(label_sub, left_on="subject", right_on="study_id", how="inner")

    y = df[target_col].astype(int).values
    groups = df["subject"].values
    X = df[emb_cols].values.astype(np.float32)

    logger.info(
        f"precomputed embeddings: {model_key}/{condition}/{level} -> "
        f"{X.shape[0]} samples x {X.shape[1]} dims, "
        f"{len(np.unique(groups))} subjects, classes={np.unique(y, return_counts=True)}"
    )
    return X, y, groups


def load_eeg_epochs(
    config: dict, label_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed EEG epoch .fif files for every subject in label_df."""
    data_root = Path(config["paths"]["data_root"])
    signal_cfg = config.get("signal", {})
    epoch_desc = signal_cfg.get("epoch_desc", "base")
    ch_names = signal_cfg.get("ch_names")
    conditions = signal_cfg.get("conditions")
    target_col = "epilepsy"

    all_X, all_y, all_groups = [], [], []

    for _, row in label_df.iterrows():
        raw_id = str(row["study_id"]).strip()
        try:
            subject_id = f"{int(raw_id):04d}"
        except ValueError:
            subject_id = raw_id

        label = int(row[target_col])
        # Recursive glob handles both flat (sub-XXXX/eeg/) and BIDS-session
        # (sub-XXXX/ses-01/eeg/) layouts.
        fif_files = sorted(
            data_root.glob(f"sub-{subject_id}/**/sub-{subject_id}*desc-{epoch_desc}*_epo.fif")
        )
        if not fif_files:
            logger.warning(f"No epoch file for sub-{subject_id}, skipping.")
            continue

        epochs = mne.read_epochs(fif_files[0], preload=True, verbose=False)
        if ch_names:
            available = [c for c in ch_names if c in epochs.ch_names]
            epochs = epochs.pick_channels(available, ordered=True)
        if conditions:
            try:
                epochs = epochs[conditions]
            except KeyError:
                # Subject has no epochs for the requested condition — skip rather
                # than loading all conditions (which would introduce off-condition
                # data and inflate RAM by orders of magnitude for sparse conditions
                # like EC_baseline where many subjects lack it).
                logger.warning(
                    f"No matching conditions for sub-{subject_id}, skipping "
                    f"(conditions={conditions})."
                )
                continue

        if len(epochs) == 0:
            logger.warning(f"Zero epochs after filtering for sub-{subject_id}, skipping.")
            continue

        X_sub = epochs.get_data(units="uV").astype(np.float32)
        n_times = X_sub.shape[-1]
        X_sub = X_sub[..., : n_times // 200 * 200]
        all_X.append(X_sub)
        all_y.extend([label] * len(X_sub))
        all_groups.extend([subject_id] * len(X_sub))

    if not all_X:
        raise RuntimeError("No EEG data loaded. Check data_root and subject IDs.")

    # Pre-allocate a single contiguous output array and fill it subject by subject.
    # np.concatenate(all_X) would hold the accumulated list AND the output array in
    # RAM simultaneously (2× peak); writing into a pre-allocated slice avoids that.
    n_ch, n_times = all_X[0].shape[1], all_X[0].shape[2]
    total = sum(x.shape[0] for x in all_X)
    X_out = np.empty((total, n_ch, n_times), dtype=np.float32)
    offset = 0
    for x in all_X:
        n = x.shape[0]
        X_out[offset : offset + n] = x
        offset += n
    del all_X  # release individual subject arrays immediately

    return X_out, np.array(all_y, dtype=int), np.array(all_groups)


def iter_subject_epochs(config: dict, label_df: pd.DataFrame):
    """Yield ``(subject_id, label, X_sub, ch_names)`` one subject at a time.

    Mirrors ``load_eeg_epochs`` but keeps each subject separate so a per-recording
    embedding can be pooled correctly and saved as its own BIDS derivative.
    """
    data_root = Path(config["paths"]["data_root"])
    signal_cfg = config.get("signal", {})
    epoch_desc = signal_cfg.get("epoch_desc", "base")
    ch_names = signal_cfg.get("ch_names")
    conditions = signal_cfg.get("conditions")
    target_col = "epilepsy"

    for _, row in label_df.iterrows():
        raw_id = str(row["study_id"]).strip()
        try:
            subject_id = f"{int(raw_id):04d}"
        except ValueError:
            subject_id = raw_id

        label = int(row[target_col])
        # Recursive glob handles both flat (sub-XXXX/eeg/) and BIDS-session
        # (sub-XXXX/ses-01/eeg/) layouts.
        fif_files = sorted(
            data_root.glob(f"sub-{subject_id}/**/sub-{subject_id}*desc-{epoch_desc}*_epo.fif")
        )
        if not fif_files:
            logger.warning(f"No epoch file for sub-{subject_id}, skipping.")
            continue

        epochs = mne.read_epochs(fif_files[0], preload=True, verbose=False)
        if ch_names:
            available = [c for c in ch_names if c in epochs.ch_names]
            epochs = epochs.pick_channels(available, ordered=True)
        if conditions:
            try:
                epochs = epochs[conditions]
            except KeyError:
                logger.warning(
                    f"No matching conditions for sub-{subject_id}, skipping "
                    f"(conditions={conditions})."
                )
                continue
        if len(epochs) == 0:
            logger.warning(f"Zero epochs after filtering for sub-{subject_id}, skipping.")
            continue

        X_sub = epochs.get_data(units="uV").astype(np.float32)
        n_times = X_sub.shape[-1]
        X_sub = X_sub[..., : n_times // 200 * 200]
        yield subject_id, label, X_sub, list(epochs.ch_names)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cv(cv_raw: dict) -> CVConfig:
    kwargs = {
        "strategy": cv_raw.get("strategy", "group_kfold"),
        "n_splits": cv_raw.get("n_splits", 5),
    }
    # subject_level_metrics only exists on the epilepsy branch's CVConfig; pass it
    # only when supported so run_analysis stays compatible with other branches.
    if "subject_level_metrics" in getattr(CVConfig, "model_fields", {}):
        kwargs["subject_level_metrics"] = cv_raw.get("subject_level_metrics", False)
    return CVConfig(**kwargs)


def _build_classical_models(models_raw: dict) -> dict:
    models = {}
    for name, mcfg in models_raw.items():
        method = mcfg.get("method", "LogisticRegression")
        params = {k: v for k, v in mcfg.items() if k != "method"}
        models[name] = ClassicalModelConfig(estimator=method, params=params)
    return models


# ---------------------------------------------------------------------------
# Post-hoc scoring (branch-independent; computed from saved fold predictions)
# ---------------------------------------------------------------------------
# These reproduce two epilepsy-branch features without touching coco_pipe:
#   - balanced_accuracy_optimal: balanced accuracy at the threshold that maximises it
#   - subject-level aggregation: mean probability per subject, then re-score
# Both are pure re-scorings of the exact stored predictions, so results are
# identical to computing them inside the CV loop (per fold, then averaged).

def _balanced_accuracy_optimal(y_true: np.ndarray, proba1: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    best = 0.0
    for t in np.unique(proba1):
        ba = balanced_accuracy_score(y_true, (proba1 >= t).astype(int))
        if ba > best:
            best = ba
    return float(best)


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, proba1: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    )
    two_class = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba1)) if two_class else float("nan"),
        "balanced_accuracy_optimal": _balanced_accuracy_optimal(y_true, proba1) if two_class else float("nan"),
    }


def _aggregate_subject(y_true: np.ndarray, proba1: np.ndarray, groups: np.ndarray):
    """Mean probability per subject -> one prediction per subject (threshold 0.5)."""
    uniq = np.unique(groups)
    sy = np.array([int(round(float(y_true[groups == g].mean()))) for g in uniq])
    sp = np.array([float(proba1[groups == g].mean()) for g in uniq])
    return sy, (sp >= 0.5).astype(int), sp


def augment_posthoc_metrics(json_path: Path, analysis_level: str = "epoch_level"):
    """Write a ``*_posthoc_metrics.json`` sidecar with epoch- and subject-level
    metrics (incl. balanced_accuracy_optimal), computed per fold then averaged."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    summary = {}
    for model, node in data.get("results", {}).items():
        per_level = {"epoch_level": [], "subject_level": []}
        for fold in node.get("predictions", []):
            yt = np.asarray(fold["y_true"])
            yp = np.asarray(fold["y_pred"])
            proba = np.asarray(fold["y_proba"])
            p1 = proba[:, 1] if proba.ndim == 2 else proba
            per_level["epoch_level"].append(_score_predictions(yt, yp, p1))
            grp = fold.get("group")
            if grp is not None:
                grp = np.asarray(grp)
                if grp.size == yt.size and len(np.unique(grp)) < yt.size:
                    sy, spred, sp = _aggregate_subject(yt, p1, grp)
                    per_level["subject_level"].append(_score_predictions(sy, spred, sp))
        out = {}
        for lvl, folds in per_level.items():
            if not folds:
                continue
            out[lvl] = {
                k: {
                    "mean": float(np.nanmean([f[k] for f in folds])),
                    "std": float(np.nanstd([f[k] for f in folds])),
                }
                for k in folds[0]
            }
        summary[model] = out
    sidecar = json_path.with_name(json_path.stem + "_posthoc_metrics.json")
    sidecar.write_text(
        json.dumps({"analysis_level": analysis_level, "metrics": summary}, indent=2)
    )
    return summary, sidecar


# ---------------------------------------------------------------------------
# Mode Handlers
# ---------------------------------------------------------------------------

def run_fm_embed(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name="results"):
    """Mode 1: Frozen backbone → embed → classical CV head."""
    model_key = analysis_cfg["model_key"]
    sfreq = float(signal_cfg.get("sfreq", 200.0))
    ch_names = signal_cfg.get("ch_names")
    models_raw = analysis_cfg.get("models", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "balanced_accuracy_optimal", "f1"])

    models = {}
    for name, mcfg in models_raw.items():
        method = mcfg.get("method", "LogisticRegression")
        params = {k: v for k, v in mcfg.items() if k != "method"}
        models[name] = FrozenBackboneDecoderConfig(
            backbone=FoundationEmbeddingModelConfig(
                model_key=model_key,
                sfreq=sfreq,
                ch_names=ch_names,
                train_mode="frozen",
            ),
            head=ClassicalModelConfig(estimator=method, params=params),
        )

    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models=models,
            cv=_build_cv(cv_raw),
            metrics=metrics,
            output_dir=output_dir,
            tag=f"fm_embed_{model_key}",
        )
    )
    result = exp.run(X, y, groups=groups)
    result.save(Path(output_dir) / f"{result_name}.json")
    return result


def run_embed_head(analysis_cfg, X, y, groups, output_dir, result_name="results"):
    """Attach a classical head to pre-extracted FM embeddings (no extraction).

    Same CV/head machinery as the handcrafted mode: X is the embedding matrix.
    """
    models_raw = analysis_cfg.get("models", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get(
        "metrics", ["accuracy", "roc_auc", "balanced_accuracy", "balanced_accuracy_optimal", "f1"]
    )
    model_key = analysis_cfg.get("model_key", "fm")
    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models=_build_classical_models(models_raw),
            cv=_build_cv(cv_raw),
            metrics=metrics,
            output_dir=output_dir,
            tag=f"fm_embed_precomputed_{model_key}",
        )
    )
    result = exp.run(X, y, groups=groups)
    json_path = Path(output_dir) / f"{result_name}.json"
    result.save(json_path)

    # Post-hoc metrics (balanced_accuracy_optimal + subject-level aggregation),
    # computed from the saved fold predictions so results match the CV loop.
    level = "subject_level" if cv_raw.get("subject_level_metrics") else "epoch_level"
    summary, sidecar = augment_posthoc_metrics(json_path, analysis_level=level)
    for m, lv in summary.items():
        sel = lv.get(level) or lv.get("epoch_level", {})
        logger.info(f"[{m}] post-hoc {level}: " +
                    ", ".join(f"{k}={v['mean']:.3f}" for k, v in sel.items()))
    logger.info(f"post-hoc metrics -> {sidecar}")
    return result


def _condition_short(signal_cfg: dict) -> str:
    """Derive a short condition tag (e.g. 'EO') for BIDS entities and summaries."""
    conds = signal_cfg.get("conditions") or []
    if conds:
        return str(conds[0]).split("_")[0]
    return "ALL"


def _window_signaljepa(X_sub: np.ndarray, window: int = 400) -> np.ndarray:
    """Re-slice (n_ep, n_ch, T) epochs into non-overlapping `window`-sample windows."""
    n_ep, n_ch, n_times = X_sub.shape
    n_wins = n_times // window
    return (
        X_sub[:, :, : n_wins * window]
        .reshape(n_ep, n_ch, n_wins, window)
        .transpose(0, 2, 1, 3)
        .reshape(n_ep * n_wins, n_ch, window)
    )


def _concat_subject_to_window(X_sub: np.ndarray, window: int) -> np.ndarray:
    """Concatenate one subject's epochs along time, then re-slice at `window`.

    (n_ep, n_ch, n_times) → (n_wins, n_ch, window). Avoids zero-padding when each
    epoch is shorter than `window` (e.g. LaBraM's 3000-sample requirement). Returns
    an empty (0, n_ch, window) array when the subject has insufficient total data.
    """
    n_ep, n_ch, n_times = X_sub.shape
    X_cat = X_sub.transpose(1, 0, 2).reshape(n_ch, n_ep * n_times)
    n_wins = X_cat.shape[-1] // window
    if n_wins == 0:
        return np.empty((0, n_ch, window), dtype=X_sub.dtype)
    return (
        X_cat[:, : n_wins * window]
        .reshape(n_ch, n_wins, window)
        .transpose(1, 0, 2)
    )


def run_fm_extract(
    analysis_cfg,
    config,
    label_df,
    signal_cfg,
    embeddings_root,
    cohort,
    result_name="results",
    overwrite=False,
):
    """Option B: per-subject frozen-FM embedding extraction → BIDS save → CV heads.

    For every subject in ``label_df`` extract one embedding per window plus a pooled
    recording vector, save them as a BIDS derivative under
    ``<embeddings_root>/<model_key>/sub-XXXX/eeg/...``, then run the configured
    classical heads (logreg/rf/hgb/dummy/svm) on the window-level embeddings and
    append per-head metrics to ``embeddings_performance_summary.csv``.
    """
    model_key = analysis_cfg["model_key"]
    sfreq = float(signal_cfg.get("sfreq", 200.0))
    cond = _condition_short(signal_cfg)
    desc = f"{model_key}Mean"
    model_root = Path(embeddings_root) / model_key

    backend_kwargs = {}
    if model_key in {"labram", "bendr"}:
        backend_kwargs["interpolate_channels"] = True

    extractor = FoundationEmbeddingExtractor(
        model_key,
        pooling="mean",
        recording_pooling="mean",
        normalize_embeddings=True,
        backend_kwargs=backend_kwargs,
    )

    records = []
    saved_paths = []
    for subject_id, label, X_sub, ch_names in iter_subject_epochs(config, label_df):
        cur_ch = ch_names
        if model_key == "biot":
            # Defer BIOT's bipolar montage to coco-pipe; supply modern names.
            cur_ch = _to_modern_nomenclature(ch_names)
        if model_key == "signaljepa" and X_sub.shape[-1] > 400:
            X_sub = _window_signaljepa(X_sub, window=400)
        if model_key == "labram" and X_sub.shape[-1] != 3000:
            # LaBraM requires exactly 3000 samples (15 s @ 200 Hz). Concatenate this
            # subject's epochs along time and re-slice into 3000-sample windows (no
            # implicit padding/cropping is allowed by the backend).
            X_sub = _concat_subject_to_window(X_sub, window=3000)
            if X_sub.shape[0] == 0:
                logger.warning(f"sub-{subject_id}: <3000 samples total for LaBraM, skipping.")
                records.append({"subject": subject_id, "status": "failed", "reason": "insufficient_samples_for_labram"})
                continue

        rec_id = f"sub-{subject_id}_task-{cond}"
        path = (
            model_root / f"sub-{subject_id}" / "eeg"
            / f"sub-{subject_id}_task-{cond}_desc-{desc}_embedding.npz"
        )
        if path.exists() and not overwrite:
            logger.info(f"sub-{subject_id}: embedding exists, reusing ({path.name}).")
            saved_paths.append(path)
            records.append({"subject": subject_id, "status": "success", "reason": "exists",
                            "artifact_path": str(path)})
            continue

        try:
            result = extractor.extract(
                X_sub,
                signal_metadata=SignalMetadata(sfreq=sfreq, ch_names=cur_ch),
                metadata={
                    "subject": subject_id,
                    "recording_id": rec_id,
                    "condition": cond,
                    "label": int(label),
                    "patient_group_id": subject_id,
                    **{f"cohort_{k}": v for k, v in (cohort or {}).items()},
                },
            )
            save_embedding_derivative(result, path, overwrite=overwrite)
            saved_paths.append(path)
            records.append({"subject": subject_id, "status": "success",
                            "artifact_path": str(path),
                            "n_windows": int(result.window_embeddings.shape[0])})
            logger.info(
                f"sub-{subject_id}: saved {result.window_embeddings.shape[0]} window "
                f"embeddings (dim={result.window_embeddings.shape[1]})."
            )
        except Exception as exc:  # noqa: BLE001 — record and continue to next subject
            logger.exception(f"sub-{subject_id}: extraction failed: {exc}")
            records.append({"subject": subject_id, "status": "failed", "reason": str(exc)})

    # Provenance artifacts (dataset description is idempotent; manifest is per-condition
    # so EO and EC jobs writing to the same model root do not clobber each other).
    write_embedding_dataset_description(
        model_root,
        name=f"{model_key} frozen foundation-model embeddings",
        bids_version="1.9.0",
        generated_by=[{
            "Name": "coco_pipe.decoding.FoundationEmbeddingExtractor",
            "Model": model_key,
            "Pooling": "mean",
        }],
    )
    write_embedding_manifest(model_root / f"task-{cond}", records)

    n_success = sum(1 for r in records if r["status"] == "success")
    logger.info(f"{model_key}/{cond}: {n_success}/{len(records)} subjects extracted.")
    if not saved_paths:
        logger.error(f"{model_key}/{cond}: no embeddings available, skipping head CV.")
        return None

    # ── Classical heads on the saved window-level embeddings ──────────────────
    container = load_embedding_derivatives(
        saved_paths, representation="window", model_key=model_key
    )
    X = np.asarray(container.X, dtype=np.float32)
    y = np.asarray(container.coords["label"]).astype(int)
    groups = np.asarray(container.coords["subject"])

    models = _build_classical_models(analysis_cfg.get("models", {}))
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get(
        "metrics",
        ["accuracy", "roc_auc", "balanced_accuracy", "balanced_accuracy_optimal", "f1"],
    )
    head_out_dir = model_root / f"task-{cond}" / "head_cv"
    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models=models,
            cv=_build_cv(cv_raw),
            metrics=metrics,
            output_dir=str(head_out_dir),
            tag=f"fm_embed_{model_key}_{cond}",
        )
    )
    head_result = exp.run(X, y, groups=groups)
    json_path = head_out_dir / f"{result_name}.json"
    head_result.save(json_path)

    summary_csv = Path(embeddings_root) / "embeddings_performance_summary.csv"
    n_rows = _append_embeddings_summary(
        summary_csv, json_path, model_key, cond,
        n_subjects=int(len(np.unique(groups))), n_windows=int(len(y)),
    )
    logger.info(f"Appended {n_rows} head row(s) to {summary_csv}")
    return head_result


def run_fm_frozen(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name="results"):
    """Mode: frozen backbone — only the linear head trains."""
    model_key = analysis_cfg["model_key"]
    sfreq = float(signal_cfg.get("sfreq", 200.0))
    ch_names = signal_cfg.get("ch_names")
    trainer_raw = analysis_cfg.get("trainer", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "balanced_accuracy_optimal", "f1"])

    backend_kwargs = {}
    if model_key in {"labram", "bendr"}:
        backend_kwargs["interpolate_channels"] = True

    models = {
        model_key: NeuralFineTuneConfig(
            model_key=model_key,
            train_mode="frozen",
            sfreq=sfreq,
            ch_names=ch_names,
            trainer=TrainerConfig(
                max_epochs=trainer_raw.get("max_epochs", 15),
                batch_size=trainer_raw.get("batch_size", 64),
                early_stopping_patience=trainer_raw.get("early_stopping_patience"),
                lr=trainer_raw.get("lr", 1e-3),
                weight_decay=trainer_raw.get("weight_decay", 0.01),
                accumulate_grad_batches=trainer_raw.get("accumulate_grad_batches", 1),
                lr_warmup_epochs=trainer_raw.get("lr_warmup_epochs", 0),
            ),
            class_weight=analysis_cfg.get("class_weight", "balanced"),
            backend_kwargs=backend_kwargs,
        )
    }

    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models=models,
            cv=_build_cv(cv_raw),
            metrics=metrics,
            output_dir=output_dir,
            tag=f"fm_frozen_{model_key}",
        )
    )
    result = exp.run(X, y, groups=groups)
    result.save(Path(output_dir) / f"{result_name}.json")
    return result


def run_fm_lora(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name="results"):
    """Mode 2: LoRA fine-tuning of a foundation model."""
    model_key = analysis_cfg["model_key"]
    sfreq = float(signal_cfg.get("sfreq", 200.0))
    ch_names = signal_cfg.get("ch_names")
    lora_raw = analysis_cfg.get("lora", {})
    trainer_raw = analysis_cfg.get("trainer", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "balanced_accuracy_optimal", "f1"])

    backend_kwargs = {}
    if model_key in {"labram", "bendr"}:
        backend_kwargs["interpolate_channels"] = True

    # REVE ships a pretrained single-query attention read-out; use it by default.
    pooling = analysis_cfg.get("pooling", "attention" if model_key == "reve" else "mean")

    models = {
        model_key: NeuralFineTuneConfig(
            model_key=model_key,
            train_mode="lora",
            sfreq=sfreq,
            ch_names=ch_names,
            pooling=pooling,
            lora=LoRAConfig(
                r=lora_raw.get("r", 8),
                alpha=lora_raw.get("alpha", 16),
                dropout=lora_raw.get("dropout", 0.05),
                target_modules=lora_raw.get("target_modules", "all-linear"),
            ),
            trainer=TrainerConfig(
                max_epochs=trainer_raw.get("max_epochs", 15),
                batch_size=trainer_raw.get("batch_size", 64),
                early_stopping_patience=trainer_raw.get("early_stopping_patience"),
                lr=trainer_raw.get("lr", 1e-3),
                weight_decay=trainer_raw.get("weight_decay", 0.01),
                accumulate_grad_batches=trainer_raw.get("accumulate_grad_batches", 1),
                lr_warmup_epochs=trainer_raw.get("lr_warmup_epochs", 0),
                training_strategy=trainer_raw.get("training_strategy", "ft_only"),
                lp_epochs=trainer_raw.get("lp_epochs", 5),
                lp_lr=trainer_raw.get("lp_lr"),
                use_focal_loss=analysis_cfg.get("use_focal_loss", False),
                focal_gamma=analysis_cfg.get("focal_gamma", 2.0),
            ),
            class_weight=analysis_cfg.get("class_weight", "balanced"),
            backend_kwargs=backend_kwargs,
        )
    }

    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models=models,
            cv=_build_cv(cv_raw),
            metrics=metrics,
            output_dir=output_dir,
            tag=f"fm_lora_{model_key}",
        )
    )
    result = exp.run(X, y, groups=groups)
    result.save(Path(output_dir) / f"{result_name}.json")
    return result


def run_handcrafted(analysis_cfg, X, y, groups, output_dir, result_name="results"):
    """Mode 3: Classical ML on handcrafted features."""
    models_raw = analysis_cfg.get("models", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "balanced_accuracy_optimal", "f1"])
    outer_cv = _build_cv(cv_raw)

    fs_raw = analysis_cfg.get("feature_selection", {})
    if fs_raw.get("enabled", False):
        inner_cv_raw = fs_raw.get("cv", cv_raw)
        fs_cfg = FeatureSelectionConfig(
            enabled=True,
            method=fs_raw.get("method", "sfs"),
            n_features=fs_raw.get("n_features"),
            direction=fs_raw.get("direction", "forward"),
            cv=_build_cv(inner_cv_raw),
        )
    else:
        fs_cfg = FeatureSelectionConfig()

    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models=_build_classical_models(models_raw),
            cv=outer_cv,
            metrics=metrics,
            output_dir=output_dir,
            feature_selection=fs_cfg,
            tag="handcrafted",
        )
    )
    result = exp.run(X, y, groups=groups)
    result.save(Path(output_dir) / f"{result_name}.json")
    return result


def run_dim_reduction(analysis_cfg, X, y, ids, output_dir, feature_names=None):
    """Mode 4: Dimensionality reduction via coco_pipe.dim_reduction's pipeline.

    Uses coco_pipe's container-native ``run_fit`` / ``run_eval`` (the updated
    dim_reduction pipeline):
      - ``run_fit``  fits each reducer, saves the embedding + fit artifact, and
        scores the structure-preservation metrics (trustworthiness/continuity/
        lcmc/mrre/shepard) at the library's default neighborhood.
      - ``run_eval`` computes the supervised separation
        (``separation_logreg_balanced_accuracy``) of the label within the
        embedding, grouped by subject.
    Writes coco_pipe run inventories (fit_runs.csv / eval_runs.csv) plus a
    friendly ``dim_reduction_summary.csv``.

    Note: the pipeline builds each reducer from (method, n_components) only;
    extra reducer hyperparameters (e.g. UMAP n_neighbors) use library defaults.
    """
    from coco_pipe.io import DataContainer
    from coco_pipe.dim_reduction import (
        run_fit, run_eval, update_runs, load_fit_artifact,
        FIT_METRIC_COLUMNS, EVAL_METRIC_COLUMNS,
        FIT_RUN_KEY_FIELDS, EVAL_RUN_KEY_FIELDS,
    )

    models_cfg = analysis_cfg.get("models", {})
    selection_metric = (analysis_cfg.get("evaluation", {}) or {}).get(
        "selection_metric", "trustworthiness"
    )
    target_name = analysis_cfg.get("target_col", "epilepsy")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    ids_arr = np.asarray(ids, dtype=object).astype(str)
    y_arr = np.asarray(y)
    feats = list(feature_names) if feature_names is not None else [
        f"feat_{i}" for i in range(X.shape[1])
    ]

    # Container carries the label (y) and a per-obs 'subject' coord so run_eval
    # can compute grouped supervised separation.
    container = DataContainer(
        X=np.asarray(X, dtype=float),
        dims=("obs", "feature"),
        coords={"feature": feats, "subject": ids_arr},
        y=y_arr,
        ids=ids_arr,
    )
    eval_spec = {
        "name": target_name, "target_col": "y", "group_col": "subject",
        "filters": [], "label_map": {},
    }

    rows = []
    for name, cfg in models_cfg.items():
        method = cfg["method"]
        n_components = cfg.get("n_components", 2)
        out_path = output_root / f"reducer-{method}_comp-{n_components}"
        fit_payload = {
            "fit_id": f"{method}_c{n_components}", "reducer": method,
            "n_components": n_components, "unit_name": name,
            # provenance fields used for artifact naming / run records
            "scope": "dim_reduction",
            "condition": analysis_cfg.get("condition", "all"),
            "unit_key": "all",
        }
        logger.info(f"[{name}] run_fit {method} (n_components={n_components}) on {X.shape}")
        fit_rec = run_fit(fit_payload, container, out_path, output_root,
                          overwrite=True, errors="record")
        update_runs(output_root / "fit_runs.csv", fit_rec, FIT_RUN_KEY_FIELDS)

        eval_rec = {}
        if fit_rec.get("status") != "failed":
            fit_artifact = load_fit_artifact(out_path)
            eval_rec = run_eval(fit_artifact, container, eval_spec,
                                out_path / f"eval-{target_name}", output_root,
                                overwrite=True, errors="record")
            update_runs(output_root / "eval_runs.csv", eval_rec, EVAL_RUN_KEY_FIELDS)

        row = {"unit_name": name, "reducer": method, "n_components": n_components,
               "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
               "status": fit_rec.get("status", "success")}
        for m in FIT_METRIC_COLUMNS:
            row[m] = fit_rec.get(m)
        for m in EVAL_METRIC_COLUMNS:
            row[m] = eval_rec.get(m)
        rows.append(row)
        logger.info(f"[{name}] {method}: trust={row.get('trustworthiness')} "
                    f"cont={row.get('continuity')} "
                    f"sep={row.get('separation_logreg_balanced_accuracy')}")

    if rows:
        df = pd.DataFrame(rows)
        if selection_metric in df.columns:
            df = df.sort_values(selection_metric, ascending=False, na_position="last")
        summary_path = output_root / "dim_reduction_summary.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"Saved comparison table -> {summary_path}")


# ---------------------------------------------------------------------------
# Channel nomenclature
# ---------------------------------------------------------------------------

# Old (T3/T4/T5/T6) -> modern 10-20 (T7/T8/P7/P8). coco-pipe's montage system
# (_montages) recognises the modern names, so we rename rather than deriving
# BIOT's bipolar montage ourselves — the backend builds it internally.
_MODERN_NOMENCLATURE = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _to_modern_nomenclature(ch_names: list[str]) -> list[str]:
    return [_MODERN_NOMENCLATURE.get(c, c) for c in ch_names]


# ---------------------------------------------------------------------------
# BIOT bipolar derivation (legacy: kept for reference; coco-pipe now derives
# the montage internally, so run_analysis only renames channels for BIOT)
# ---------------------------------------------------------------------------

# 16 TCP bipolar pairs that BIOT was pretrained on, expressed in 10-20 names.
# Every electrode here is present in our 19-channel unipolar set.
_BIOT_BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),   # left temporal
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),   # right temporal
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),   # left parasagittal
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),   # right parasagittal
]
_BIOT_BIPOLAR_NAMES = [f"{a}-{b}" for a, b in _BIOT_BIPOLAR_PAIRS]


def _to_biot_bipolar(
    X: np.ndarray,
    ch_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Convert (N, 19, T) unipolar data to (N, 16, T) TCP bipolar derivations."""
    idx = {name: i for i, name in enumerate(ch_names)}
    X_bip = np.stack(
        [X[:, idx[a], :] - X[:, idx[b], :] for a, b in _BIOT_BIPOLAR_PAIRS],
        axis=1,
    )
    return X_bip, _BIOT_BIPOLAR_NAMES


# ---------------------------------------------------------------------------
# Epoch windowing helpers
# ---------------------------------------------------------------------------

def _slice_epochs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each epoch into non-overlapping windows of length `window`.

    (N, C, T) → (N * (T // window), C, window), repeating y and groups.
    Trailing samples that don't fill a full window are discarded.
    """
    n_epochs, n_ch, n_times = X.shape
    n_wins = n_times // window
    X_sliced = (
        X[:, :, : n_wins * window]
        .reshape(n_epochs, n_ch, n_wins, window)
        .transpose(0, 2, 1, 3)
        .reshape(n_epochs * n_wins, n_ch, window)
    )
    y_sliced = np.repeat(y, n_wins)
    groups_sliced = np.repeat(groups, n_wins)
    return X_sliced, y_sliced, groups_sliced


def _concat_and_slice_epochs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per subject: concatenate all epochs along time then re-slice at `window`.

    Avoids zero-padding when each epoch is shorter than `window`.
    Subjects with insufficient total data for one window are dropped.
    """
    result_X, result_y, result_groups = [], [], []
    for grp in np.unique(groups):
        mask = groups == grp
        X_grp = X[mask]          # (n_ep, n_ch, n_times)
        label = y[mask][0]
        n_ep, n_ch, n_times = X_grp.shape
        X_cat = X_grp.transpose(1, 0, 2).reshape(n_ch, n_ep * n_times)
        n_wins = X_cat.shape[-1] // window
        if n_wins == 0:
            continue
        X_wins = (
            X_cat[:, : n_wins * window]
            .reshape(n_ch, n_wins, window)
            .transpose(1, 0, 2)
        )
        result_X.append(X_wins)
        result_y.extend([label] * n_wins)
        result_groups.extend([grp] * n_wins)
    return (
        np.concatenate(result_X, axis=0),
        np.array(result_y),
        np.array(result_groups),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified EEG Epilepsy pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--analysis-id", required=True, help="ID of the analysis block")
    parser.add_argument("--group-id", required=True, help="Demographic group identifier")
    parser.add_argument("--label-csv", default=None, help="Filtered cohort label CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--result-name", default="results", help="Stem of the output JSON file")
    parser.add_argument("--summary-csv", default=None, help="Shared summary CSV to append results to")
    parser.add_argument("--extra-col", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra column(s) to add to the summary CSV (e.g. --extra-col lr=0.0001)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    analyses = config.get("analyses", [])
    analysis_cfg = next((a for a in analyses if a["id"] == args.analysis_id), None)
    if analysis_cfg is None:
        raise ValueError(f"Analysis ID '{args.analysis_id}' not found in config.")
    if not analysis_cfg.get("enabled", True):
        logger.info(f"Analysis '{args.analysis_id}' is disabled, skipping.")
        return

    # Global toggle: add or strip balanced_accuracy_optimal from the metrics list
    _bao = config.get("balanced_accuracy_optimal", True)
    _metrics = analysis_cfg.setdefault("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "f1"])
    if _bao and "balanced_accuracy_optimal" not in _metrics:
        _metrics.append("balanced_accuracy_optimal")
    elif not _bao and "balanced_accuracy_optimal" in _metrics:
        _metrics.remove("balanced_accuracy_optimal")

    _balanced_sample = config.get("balanced_sample", False)

    # Global analysis level: propagate to every analysis block's cv config.
    # "subject_level" → aggregate epoch predictions per subject before scoring.
    # "epoch_level"   → score each epoch independently (default behaviour).
    _analysis_level = config.get("analysis_level", "epoch_level")
    analysis_cfg.setdefault("cv", {})["subject_level_metrics"] = (
        _analysis_level == "subject_level"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = config.get("paths", {})
    signal_cfg = config.get("signal", {})

    label_csv = args.label_csv or paths.get("label_csv")
    if not label_csv:
        raise ValueError("Provide --label-csv or set paths.label_csv in the config.")
    label_df = normalize_label_df(pd.read_csv(label_csv))
    logger.info(f"Loaded {len(label_df)} subjects from {label_csv}")

    mode = analysis_cfg.get("mode")
    summary_csv = Path(args.summary_csv) if args.summary_csv else None

    result_name = args.result_name

    if mode == "handcrafted":
        X, y, groups = load_handcrafted_data(analysis_cfg, label_df)
        if _balanced_sample:
            X, y, groups = _undersample_majority(X, y, groups)
            logger.info(f"After undersampling: {len(y)} samples, classes={np.unique(y, return_counts=True)}")
        run_handcrafted(analysis_cfg, X, y, groups, output_dir, result_name=result_name)

    elif mode == "fm_embed_precomputed":
        # Use pre-extracted embeddings for the chosen model; attach a classical head.
        X, y, groups = load_precomputed_embeddings(analysis_cfg, config, label_df, signal_cfg)
        if _balanced_sample:
            X, y, groups = _undersample_majority(X, y, groups)
            logger.info(f"After undersampling: {len(y)} samples, classes={np.unique(y, return_counts=True)}")
        run_embed_head(analysis_cfg, X, y, groups, output_dir, result_name=result_name)

    elif mode == "fm_embed":
        X, y, groups = load_eeg_epochs(config, label_df)
        model_key = analysis_cfg.get("model_key", "")
        if model_key == "biot":
            signal_cfg = {**signal_cfg, "ch_names": _to_modern_nomenclature(signal_cfg.get("ch_names", []))}
        if model_key == "signaljepa" and X.shape[-1] > 400:
            X, y, groups = _slice_epochs(X, y, groups, window=400)
        if _balanced_sample:
            X, y, groups = _undersample_majority(X, y, groups)
            logger.info(f"After undersampling: {len(y)} samples, classes={np.unique(y, return_counts=True)}")
        run_fm_embed(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name=result_name)

    elif mode == "fm_extract":
        cohort = _parse_group_id(args.group_id)
        run_fm_extract(
            analysis_cfg,
            config,
            label_df,
            signal_cfg,
            embeddings_root=output_dir,
            cohort=cohort,
            result_name=result_name,
            overwrite=config.get("overwrite_embeddings", False),
        )

    elif mode == "fm_frozen":
        X, y, groups = load_eeg_epochs(config, label_df)
        model_key = analysis_cfg.get("model_key", "")
        if model_key == "signaljepa" and X.shape[-1] > 400:
            X, y, groups = _slice_epochs(X, y, groups, window=400)
        if _balanced_sample:
            X, y, groups = _undersample_majority(X, y, groups)
            logger.info(f"After undersampling: {len(y)} samples, classes={np.unique(y, return_counts=True)}")
        run_fm_frozen(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name=result_name)

    elif mode == "fm_lora":
        X, y, groups = load_eeg_epochs(config, label_df)
        model_key = analysis_cfg.get("model_key", "")
        if model_key == "biot":
            # Defer BIOT's TCP-bipolar montage to coco-pipe; just supply modern
            # channel names (T3/T4/T5/T6 -> T7/T8/P7/P8) so its montage resolves.
            signal_cfg = {**signal_cfg, "ch_names": _to_modern_nomenclature(signal_cfg.get("ch_names", []))}
        if model_key == "labram" and X.shape[-1] < 3000:
            X, y, groups = _concat_and_slice_epochs(X, y, groups, window=3000)
        if model_key == "signaljepa" and X.shape[-1] > 400:
            X, y, groups = _slice_epochs(X, y, groups, window=400)
        if _balanced_sample:
            X, y, groups = _undersample_majority(X, y, groups)
            logger.info(f"After undersampling: {len(y)} samples, classes={np.unique(y, return_counts=True)}")
        run_fm_lora(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name=result_name)

    elif mode == "dim_reduction":
        X, y, groups, _feat_names = load_dim_reduction_data(analysis_cfg, label_df)
        run_dim_reduction(analysis_cfg, X, y, ids=groups, output_dir=output_dir,
                          feature_names=_feat_names)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    # Append to shared summary CSV for all modes that produce a results JSON.
    # fm_extract writes its own embeddings_performance_summary.csv inside the handler.
    if summary_csv and mode not in ("dim_reduction", "fm_extract"):
        json_path = output_dir / f"{args.result_name}.json"
        if json_path.exists():
            cohort = _parse_group_id(args.group_id)
            if not cohort["condition"] and args.result_name:
                # Fallback: extract condition from result_name ("results_{model}_{cond}")
                parts = args.result_name.split("_", 2)
                if len(parts) >= 3:
                    cohort["condition"] = parts[2]
            rows = _rows_from_result_json(json_path, cohort)
            extra_col_dict = {}
            for kv in (args.extra_col or []):
                k, v = kv.split("=", 1)
                extra_col_dict[k] = v
            if extra_col_dict:
                for row in rows:
                    row.update(extra_col_dict)
            _append_to_summary(summary_csv, rows, extra_fields=list(extra_col_dict.keys()) or None)
            logger.info(f"Appended {len(rows)} row(s) to {summary_csv}")

    logger.info(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
