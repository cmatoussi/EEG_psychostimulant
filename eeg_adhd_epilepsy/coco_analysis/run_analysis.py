import json
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from coco_pipe.decoding.configs import (
    CVConfig,
    ClassicalModelConfig,
    ExperimentConfig,
)
from coco_pipe.decoding.experiment import Experiment
from coco_pipe.decoding.posthoc import posthoc_metrics_from_result as augment_posthoc_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Earliest-per-condition labels: one recording/patient -> group-CV stays patient-level.
EARLIEST_LABELS = {
    "EO": "/home/mat/scratch/labels_earliest_EO.csv",
    "EC": "/home/mat/scratch/labels_earliest_EC.csv",
}


def resolve_label_csv(condition, override=None):
    """`override` if given, else the earliest-recording file for `condition`."""
    if override:
        return override
    cond = str(condition).replace("_baseline", "")
    if cond not in EARLIEST_LABELS:
        raise ValueError(f"no earliest-label file for condition {condition!r}; "
                         f"pass --label-csv explicitly.")
    return EARLIEST_LABELS[cond]







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


def restrict_for_asm_resistant(df: pd.DataFrame) -> pd.DataFrame:
    """Cohort for the asm_resistant target: treated epilepsy with a known resistance
    label (epilepsy==1 & asm==1 & asm_resistant present). Resistance is undefined for
    never-treated patients, and asm_resistant==1 with asm==0 is a data mismatch; both
    are dropped."""
    epi = pd.to_numeric(df["epilepsy"], errors="coerce").fillna(0) == 1
    asm = pd.to_numeric(df["asm"], errors="coerce").fillna(0) == 1
    known = df["asm_resistant"].notna()
    return df[epi & asm & known].copy()




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
    target_col = signal_cfg.get("target_col", "epilepsy")
    if target_col not in label_df.columns:
        raise ValueError(f"target_col {target_col!r} not in label_df columns; "
                         f"pass signal.target_col explicitly.")

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


# Post-hoc scoring (both epoch- and subject-level metrics from one CV run's saved
# predictions) now lives in coco_pipe; imported at the top as augment_posthoc_metrics.


# ---------------------------------------------------------------------------
# Mode Handlers
# ---------------------------------------------------------------------------



def run_embed_head(analysis_cfg, X, y, groups, output_dir, result_name="results"):
    """Attach a classical head to pre-extracted FM embeddings (no extraction).

    Same CV/head machinery as the handcrafted mode: X is the embedding matrix.
    """
    models_raw = analysis_cfg.get("models", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get(
        "metrics", ["accuracy", "roc_auc", "balanced_accuracy", "youden_threshold_balanced_accuracy", "f1"]
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

    # Post-hoc metrics (youden_threshold_balanced_accuracy + subject-level aggregation),
    # computed from the saved fold predictions so results match the CV loop.
    level = "subject_level" if cv_raw.get("subject_level_metrics") else "epoch_level"
    summary, sidecar = augment_posthoc_metrics(json_path, analysis_level=level)
    for m, lv in summary.items():
        sel = lv.get(level) or lv.get("epoch_level", {})
        logger.info(f"[{m}] post-hoc {level}: " +
                    ", ".join(f"{k}={v['mean']:.3f}" for k, v in sel.items()))
    logger.info(f"post-hoc metrics -> {sidecar}")
    return result




















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


def _prep_model_data(X, y, groups, model_key, signal_cfg):
    """Model-specific input preprocessing for the finetune drivers."""
    ch_names = signal_cfg.get("ch_names")
    if model_key == "biot":
        # Modern channel names (T3/T4/T5/T6 -> T7/T8/P7/P8) so BIOT's TCP-bipolar
        # montage resolves; the montage itself is applied by coco-pipe.
        ch_names = _to_modern_nomenclature(signal_cfg.get("ch_names", []))
    if model_key == "labram" and X.shape[-1] < 3000:
        X, y, groups = _concat_and_slice_epochs(X, y, groups, window=3000)
    if model_key == "signaljepa" and X.shape[-1] > 400:
        X, y, groups = _slice_epochs(X, y, groups, window=400)
    return X, y, groups, ch_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



