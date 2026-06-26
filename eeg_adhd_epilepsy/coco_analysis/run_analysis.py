import argparse
import csv
import fcntl
import json
import logging
import yaml
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
from coco_pipe.dim_reduction.core import DimReduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_FIELDNAMES = [
    "model", "condition", "sex", "age", "comorbidities", "medication",
    "status", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean",
    "balanced_accuracy_std", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
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


def _append_to_summary(summary_csv: Path, rows: list) -> None:
    """Append result rows to the shared summary CSV with an exclusive file lock."""
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            write_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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
# Data Loaders
# ---------------------------------------------------------------------------

def load_handcrafted_data(
    analysis_cfg: dict, label_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature CSV and align it to the cohort subjects in label_df."""
    data_path = analysis_cfg["data_path"]
    target_col = analysis_cfg.get("target_col", "Epilepsy")
    subject_col = "Study ID"

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
    drop_cols = {subject_col, target_col, "Sex", "Age", "TSA", "TDAH"}
    X_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])

    # Optional spatial / region slicing
    analysis_unit = analysis_cfg.get("analysis_unit", "all")
    spatial_units = analysis_cfg.get("spatial_units", "all")
    if analysis_unit == "region" and spatial_units != "all":
        X_df = map_sensor_to_region(X_df, spatial_units)
    elif analysis_unit == "sensor" and isinstance(spatial_units, list):
        X_df = X_df[[c for c in X_df.columns if any(c.startswith(f"{s}_") for s in spatial_units)]]

    return X_df.values.astype(np.float32), y, groups


def load_eeg_epochs(
    config: dict, label_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed EEG epoch .fif files for every subject in label_df."""
    data_root = Path(config["paths"]["data_root"])
    signal_cfg = config.get("signal", {})
    epoch_desc = signal_cfg.get("epoch_desc", "base")
    ch_names = signal_cfg.get("ch_names")
    conditions = signal_cfg.get("conditions")
    target_col = "Epilepsy"

    all_X, all_y, all_groups = [], [], []

    for _, row in label_df.iterrows():
        raw_id = str(row["Study ID"]).strip()
        try:
            subject_id = f"{int(raw_id):04d}"
        except ValueError:
            subject_id = raw_id

        label = int(row[target_col])
        sub_eeg = data_root / f"sub-{subject_id}" / "eeg"
        fif_files = sorted(sub_eeg.glob(f"sub-{subject_id}*desc-{epoch_desc}*_epo.fif"))
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
    return CVConfig(
        strategy=cv_raw.get("strategy", "group_kfold"),
        n_splits=cv_raw.get("n_splits", 5),
    )


def _build_classical_models(models_raw: dict) -> dict:
    models = {}
    for name, mcfg in models_raw.items():
        method = mcfg.get("method", "LogisticRegression")
        params = {k: v for k, v in mcfg.items() if k != "method"}
        models[name] = ClassicalModelConfig(estimator=method, params=params)
    return models


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
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "f1"])

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


def run_fm_frozen(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name="results"):
    """Mode: frozen backbone — only the linear head trains."""
    model_key = analysis_cfg["model_key"]
    sfreq = float(signal_cfg.get("sfreq", 200.0))
    ch_names = signal_cfg.get("ch_names")
    trainer_raw = analysis_cfg.get("trainer", {})
    cv_raw = analysis_cfg.get("cv", {})
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "f1"])

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
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "f1"])

    backend_kwargs = {}
    if model_key in {"labram", "bendr"}:
        backend_kwargs["interpolate_channels"] = True

    models = {
        model_key: NeuralFineTuneConfig(
            model_key=model_key,
            train_mode="lora",
            sfreq=sfreq,
            ch_names=ch_names,
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
    metrics = analysis_cfg.get("metrics", ["accuracy", "roc_auc", "balanced_accuracy", "f1"])
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


def run_dim_reduction(analysis_cfg, X, y, ids, output_dir):
    """Mode 4: Dimensionality reduction via coco_pipe.dim_reduction."""
    models_cfg = analysis_cfg.get("models", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in models_cfg.items():
        method = cfg["method"]
        n_components = cfg.get("n_components", 2)

        reducer = DimReduction(method=method, n_components=n_components)
        for k, v in cfg.items():
            if k not in ("method", "n_components") and hasattr(reducer.reducer, k):
                setattr(reducer.reducer, k, v)

        embedding = reducer.fit_transform(X)
        metrics = dict(reducer.get_metrics())

        artifact_dir = output_dir / f"reducer-{method}_comp-{n_components}"
        artifact_dir.mkdir(exist_ok=True)
        np.save(artifact_dir / "embedding.npy", embedding)
        np.save(artifact_dir / "ids.npy", np.array(ids))
        with open(artifact_dir / "metrics.json", "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        logger.info(f"Saved {method} artifacts to {artifact_dir}")


# ---------------------------------------------------------------------------
# BIOT bipolar derivation
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = config.get("paths", {})
    signal_cfg = config.get("signal", {})

    label_csv = args.label_csv or paths.get("label_csv")
    if not label_csv:
        raise ValueError("Provide --label-csv or set paths.label_csv in the config.")
    label_df = pd.read_csv(label_csv)
    logger.info(f"Loaded {len(label_df)} subjects from {label_csv}")

    mode = analysis_cfg.get("mode")
    summary_csv = Path(args.summary_csv) if args.summary_csv else None

    result_name = args.result_name

    if mode == "handcrafted":
        X, y, groups = load_handcrafted_data(analysis_cfg, label_df)
        run_handcrafted(analysis_cfg, X, y, groups, output_dir, result_name=result_name)

    elif mode == "fm_embed":
        X, y, groups = load_eeg_epochs(config, label_df)
        model_key = analysis_cfg.get("model_key", "")
        if model_key == "biot":
            X, biot_ch_names = _to_biot_bipolar(X, signal_cfg.get("ch_names", []))
            signal_cfg = {**signal_cfg, "ch_names": biot_ch_names}
        if model_key == "signaljepa" and X.shape[-1] > 400:
            X, y, groups = _slice_epochs(X, y, groups, window=400)
        run_fm_embed(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name=result_name)

    elif mode == "fm_frozen":
        X, y, groups = load_eeg_epochs(config, label_df)
        model_key = analysis_cfg.get("model_key", "")
        if model_key == "signaljepa" and X.shape[-1] > 400:
            X, y, groups = _slice_epochs(X, y, groups, window=400)
        run_fm_frozen(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name=result_name)

    elif mode == "fm_lora":
        X, y, groups = load_eeg_epochs(config, label_df)
        model_key = analysis_cfg.get("model_key", "")
        if model_key == "biot":
            X, biot_ch_names = _to_biot_bipolar(X, signal_cfg.get("ch_names", []))
            signal_cfg = {**signal_cfg, "ch_names": biot_ch_names}
        if model_key == "labram" and X.shape[-1] < 3000:
            X, y, groups = _concat_and_slice_epochs(X, y, groups, window=3000)
        if model_key == "signaljepa" and X.shape[-1] > 400:
            X, y, groups = _slice_epochs(X, y, groups, window=400)
        run_fm_lora(analysis_cfg, X, y, groups, output_dir, signal_cfg, result_name=result_name)

    elif mode == "dim_reduction":
        X, y, groups = load_handcrafted_data(analysis_cfg, label_df)
        run_dim_reduction(analysis_cfg, X, y, ids=groups, output_dir=output_dir)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    # Append to shared summary CSV for all modes that produce a results JSON
    if summary_csv and mode != "dim_reduction":
        json_path = output_dir / f"{args.result_name}.json"
        if json_path.exists():
            cohort = _parse_group_id(args.group_id)
            if not cohort["condition"] and args.result_name:
                # Fallback: extract condition from result_name ("results_{model}_{cond}")
                parts = args.result_name.split("_", 2)
                if len(parts) >= 3:
                    cohort["condition"] = parts[2]
            rows = _rows_from_result_json(json_path, cohort)
            _append_to_summary(summary_csv, rows)
            logger.info(f"Appended {len(rows)} row(s) to {summary_csv}")

    logger.info(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
