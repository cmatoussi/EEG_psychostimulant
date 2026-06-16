#!/usr/bin/env python3
"""
crbamod_predict_og.py

Evaluates the original pretrained CBraMod model (NO fine-tuning) on epilepsy
classification for each EEG condition.

Pipeline:
  1. Load frozen CBraMod backbone (original pretrained weights, no adaptation)
  2. For each condition: extract pooled embeddings from all epochs in-memory
  3. Run 5-fold StratifiedGroupKFold with a Logistic Regression head
  4. Report AUC, Accuracy, Balanced Accuracy, F1 per fold
  5. Save results → /home/mat/projects/EEG_psychostimulant/data/results/eval/cbramod_without_finetuning.csv

Usage:
  python crbamod_predict_og.py
  python crbamod_predict_og.py --conditions EO_baseline EC_baseline
"""

import os
import sys
import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# CBraMod path
# ──────────────────────────────────────────────────────────────────────────────
CBRAMOD_PATH = "/home/mat/CBraMod/"
if CBRAMOD_PATH not in sys.path:
    sys.path.append(CBRAMOD_PATH)

from models.cbramod import CBraMod

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT    = "/home/mat/scratch/preproc/"
LABEL_CSV    = "/home/mat/scratch/epilepsy_label_cleaned.csv"
WEIGHTS_PATH = "/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth"
OUT_CSV      = "/home/mat/projects/EEG_psychostimulant/data/results/eval/cbramod_without_finetuning.csv"

FS_TARGET    = 200
INPUT_SEC    = 10
PATCH_SIZE   = 200          # 1-second patches
NUM_SEGMENTS = 10           # 10 patches per epoch
TARGET_LEN   = PATCH_SIZE * NUM_SEGMENTS   # 2000 samples
BATCH_SIZE   = 64
N_SPLITS     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CONDITIONS = [
    "EO_baseline", "EC_baseline",
    "HV_EC", "HV_EO",
    "PostHV_EO", "PostHV_EC",
    "PHOTO_EC", "PHOTO_EO",
]

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (identical pattern to cbramod_fine_lp_ft.py)
# ──────────────────────────────────────────────────────────────────────────────
def load_label_map(label_csv: str) -> dict:
    df = pd.read_csv(label_csv)
    df["Study ID"] = df["Study ID"].astype(int)
    return dict(zip(df["Study ID"], df["Epilepsy"]))


def collect_epoch_metadata(data_root: str, label_map: dict,
                            target_conditions: list | None):
    """Return (epoch_metadata, labels, subjects, ch_names)."""
    epoch_meta, labels, subjects = [], [], []
    ch_names = None
    data_path = Path(data_root)

    print(f"Scanning {len(label_map)} subjects ...")
    for sub_id, label in tqdm(label_map.items()):
        if pd.isna(label):
            continue
        sub_id_str = f"{sub_id:04d}"
        fif_path = (data_path / f"sub-{sub_id_str}" / "eeg" /
                    f"sub-{sub_id_str}_desc-base_epo.fif")
        if not fif_path.exists():
            continue
        try:
            epochs = mne.read_epochs(str(fif_path), preload=False, verbose=False)
            if target_conditions:
                present = [c for c in target_conditions if c in epochs.event_id]
                if not present:
                    continue
                epochs = epochs[present]
            n = len(epochs)
            if n == 0:
                continue
            if ch_names is None:
                ch_names = epochs.ch_names
            for i in range(n):
                epoch_meta.append((str(fif_path), i))
                labels.append(int(label))
                subjects.append(sub_id)
        except Exception:
            continue

    print(f"  → {len(epoch_meta)} epochs from {len(set(subjects))} subjects.")
    return epoch_meta, np.array(labels), np.array(subjects), ch_names


@functools.lru_cache(maxsize=1)
def _read_epochs_cached(path):
    return mne.read_epochs(path, preload=True, verbose=False)


def load_and_preprocess_epoch(fif_path: str, local_idx: int) -> torch.Tensor:
    """Load one epoch, normalise, reshape → (C, S, P)."""
    epochs_obj = _read_epochs_cached(fif_path)
    data = epochs_obj.get_data(item=local_idx)[0]   # (C, T)

    # Fix length
    if data.shape[1] > TARGET_LEN:
        data = data[:, :TARGET_LEN]
    elif data.shape[1] < TARGET_LEN:
        pad = TARGET_LEN - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)))

    # Z-score per channel
    mean = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True)
    data = (data - mean) / (std + 1e-6)

    # Reshape → (C, NUM_SEGMENTS, PATCH_SIZE)
    data = data.reshape(data.shape[0], NUM_SEGMENTS, PATCH_SIZE)
    return torch.FloatTensor(data)   # (C, S, P)

# ──────────────────────────────────────────────────────────────────────────────
# Frozen CBraMod feature extractor
# ──────────────────────────────────────────────────────────────────────────────
class FrozenCBraMod(nn.Module):
    """
    Loads pretrained CBraMod backbone and extracts mean-pooled patch embeddings.
    No classification head — output is a fixed-size feature vector per epoch.
    """
    def __init__(self, num_channels: int, weights_path: str):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=PATCH_SIZE, out_dim=PATCH_SIZE, d_model=PATCH_SIZE
        )
        print(f"Loading pretrained CBraMod weights from {weights_path} ...")
        self.backbone.load_state_dict(
            torch.load(weights_path, map_location="cpu"), strict=False
        )
        # Remove the projection head so backbone outputs raw patch embeddings
        self.backbone.proj_out = nn.Identity()

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

        self.num_channels = num_channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, S, P)
        Returns: (B, C*S*P) — mean-pooled patch features flattened
        """
        # backbone expects (B, C, S, P) and returns (B, C, S, P)
        features = self.backbone(x)          # (B, C, S, P)
        # Global average pool over S and P → (B, C, P) → flatten → (B, C*P)
        # Then average over channels gives (B, P); or just flatten all
        B = features.shape[0]
        return features.reshape(B, -1)       # (B, C*S*P)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_embeddings(model: FrozenCBraMod, epoch_meta: list,
                       batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Return (N, feat_dim) array of CBraMod embeddings."""
    all_embs = []
    buf = []

    def flush():
        if not buf:
            return
        batch = torch.stack(buf).to(DEVICE)       # (B, C, S, P)
        embs  = model(batch).cpu().float().numpy() # (B, feat_dim)
        all_embs.extend(embs)
        buf.clear()

    print(f"  Extracting embeddings for {len(epoch_meta)} epochs ...")
    for fif_path, local_idx in tqdm(epoch_meta):
        tensor = load_and_preprocess_epoch(fif_path, local_idx)
        buf.append(tensor)
        if len(buf) >= batch_size:
            flush()
    flush()

    return np.array(all_embs, dtype=np.float32)   # (N, feat_dim)

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_condition(X: np.ndarray, y: np.ndarray,
                       subjects: np.ndarray, condition: str) -> list[dict]:
    """5-fold CV with LR head. Returns list of per-fold result dicts."""
    if len(np.unique(y)) < 2:
        print(f"  [SKIP] {condition}: only one class present.")
        return []

    skf  = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(X, y, groups=subjects)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = 0.5

        rows.append({
            "Condition": condition,
            "Fold":      fold + 1,
            "Val_AUC":   round(float(auc), 6),
            "Val_Acc":   round(float(accuracy_score(y_val, y_pred)), 6),
            "Val_BAcc":  round(float(balanced_accuracy_score(y_val, y_pred)), 6),
            "Val_F1":    round(float(f1_score(y_val, y_pred, zero_division=0)), 6),
        })
        print(f"    Fold {fold+1}: AUC={auc:.4f}  BAcc={rows[-1]['Val_BAcc']:.4f}")

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="CBraMod original-weights evaluation (no fine-tuning)")
    p.add_argument("--data-root",    default=DATA_ROOT)
    p.add_argument("--label-csv",    default=LABEL_CSV)
    p.add_argument("--weights-path", default=WEIGHTS_PATH)
    p.add_argument("--out-csv",      default=OUT_CSV)
    p.add_argument("--conditions",   nargs="+", default=None,
                   help="Conditions to evaluate (default: all 8)")
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    print(f"Conditions : {conditions}")
    print(f"Device     : {DEVICE}")

    label_map = load_label_map(args.label_csv)
    all_rows  = []
    model     = None   # lazy init once we know num_channels

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {cond}")
        print(f"{'='*60}")

        # 1. Collect epoch indices for this condition
        epoch_meta, y, subjects, ch_names = collect_epoch_metadata(
            args.data_root, label_map, target_conditions=[cond]
        )
        if len(epoch_meta) == 0:
            print(f"  [SKIP] No epochs found for {cond}.")
            continue

        num_channels = len(ch_names)

        # 2. Lazy-init model (num_channels known after first condition scan)
        if model is None:
            model = FrozenCBraMod(num_channels=num_channels,
                                  weights_path=args.weights_path)
            model.to(DEVICE)
            model.eval()
            # Quick shape check
            dummy = torch.zeros(1, num_channels, NUM_SEGMENTS, PATCH_SIZE).to(DEVICE)
            feat_dim = model(dummy).shape[1]
            print(f"  Feature dim per epoch: {feat_dim}")

        # 3. Extract frozen CBraMod embeddings
        X = extract_embeddings(model, epoch_meta, batch_size=args.batch_size)
        print(f"  Embeddings shape: {X.shape}")

        # 4. 5-fold LR evaluation
        rows = evaluate_condition(X, y, subjects, cond)
        if not rows:
            continue

        # 5. Summary rows
        metrics_cols = ["Val_AUC", "Val_Acc", "Val_BAcc", "Val_F1"]
        fold_df  = pd.DataFrame(rows)
        mean_row = {"Condition": cond, "Fold": "Average",
                    **{c: round(fold_df[c].mean(), 6) for c in metrics_cols}}
        std_row  = {"Condition": cond, "Fold": "Std",
                    **{c: round(fold_df[c].std(), 6) for c in metrics_cols}}

        all_rows.extend(rows)
        all_rows.append(mean_row)
        all_rows.append(std_row)

        print(f"  → Mean AUC : {mean_row['Val_AUC']:.4f} ± {std_row['Val_AUC']:.4f}")
        print(f"  → Mean BAcc: {mean_row['Val_BAcc']:.4f} ± {std_row['Val_BAcc']:.4f}")

    # Save
    if all_rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"\nResults saved → {out_path}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
