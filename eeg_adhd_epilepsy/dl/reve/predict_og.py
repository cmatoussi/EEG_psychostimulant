#!/usr/bin/env python3
"""
predict_og.py

Evaluates the original pretrained REVE model (NO fine-tuning) on epilepsy
classification for each EEG condition.

Pipeline:
  1. Load frozen REVE-base + position bank (original weights, no adaptation)
  2. For each condition: extract pooled embeddings from all epochs in-memory
  3. Run 5-fold StratifiedGroupKFold with a Logistic Regression head
  4. Report AUC, Accuracy, Balanced Accuracy, F1 per fold
  5. Save results to /home/mat/projects/EEG_psychostimulant/data/results/eval/reve_without_finetuning.csv

Usage:
  python predict_og.py
  python predict_og.py --conditions EO_baseline EC_baseline
  python predict_og.py --model-size large
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mne
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/home/mat/scratch/preproc/"
LABEL_CSV   = "/home/mat/scratch/epilepsy_label_cleaned.csv"
OUT_CSV     = "/home/mat/projects/EEG_psychostimulant/data/results/eval/reve_without_finetuning.csv"

REVE_MODEL_ID = "brain-bzh/reve-base"
POS_MODEL_ID  = "brain-bzh/reve-positions"

FS_TARGET    = 200          # Hz expected by REVE
INPUT_SEC    = 10
TARGET_LEN   = FS_TARGET * INPUT_SEC   # 2000 samples per epoch
BATCH_SIZE   = 16           # epochs per forward-pass batch (GPU memory safe)
N_SPLITS     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CONDITIONS = [
    "EO_baseline", "EC_baseline",
    "HV_EC", "HV_EO",
    "PostHV_EO", "PostHV_EC",
    "PHOTO_EC", "PHOTO_EO",
]

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (same as reve_preprocessing.py)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_signal(data: np.ndarray) -> torch.Tensor:
    """Z-score per channel then clip at ±15 σ.  Input: (C, T)"""
    mean = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True)
    data = (data - mean) / (std + 1e-6)
    data = np.clip(data, -15, 15)
    return torch.tensor(data, dtype=torch.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_label_map(label_csv: str) -> dict:
    df = pd.read_csv(label_csv)
    df["Study ID"] = df["Study ID"].astype(int)
    return dict(zip(df["Study ID"], df["Epilepsy"]))


def collect_epoch_metadata(data_root: str, label_map: dict,
                            target_conditions: list | None):
    """Return (epoch_metadata, labels, subjects, ch_names).
    epoch_metadata: list of (fif_path_str, local_epoch_idx)
    """
    epoch_meta, labels, subjects = [], [], []
    ch_names = None
    data_path = Path(data_root)

    print(f"Scanning {len(label_map)} subjects for epochs ...")
    for sub_id, label in tqdm(label_map.items()):
        if pd.isna(label):
            continue
        sub_id_str = f"{sub_id:04d}"
        fif_path = data_path / f"sub-{sub_id_str}" / "eeg" / \
                   f"sub-{sub_id_str}_desc-base_epo.fif"
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

# ──────────────────────────────────────────────────────────────────────────────
# REVE frozen model wrapper
# ──────────────────────────────────────────────────────────────────────────────
class FrozenREVE(torch.nn.Module):
    """Loads original REVE weights, extracts attention-pooled embeddings."""

    def __init__(self, model_size: str = "base"):
        super().__init__()
        model_id = f"brain-bzh/reve-{model_size}"
        print(f"Loading REVE encoder from {model_id} ...")
        self.encoder  = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                                   torch_dtype="auto", token=True)
        print("Loading REVE position bank ...")
        self.pos_bank = AutoModel.from_pretrained(POS_MODEL_ID, trust_remote_code=True,
                                                   torch_dtype="auto", token=True)
        # Determine embedding dim
        self.dim = getattr(self.encoder.config, "hidden_size",
                           getattr(self.encoder.config, "embed_dim",
                                   512 if model_size == "base" else 1216))

        # Simple attention pooler
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(self.dim, 1, bias=False),
        )
        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(self, x: torch.Tensor, ch_names: list) -> np.ndarray:
        """
        x: (B, C, T) float32 on DEVICE
        Returns: (B, dim) numpy float32
        """
        self.pos_bank.to(x.device)
        coords = self.pos_bank(ch_names).to(x.device)          # (C, 3)
        pos    = coords.unsqueeze(0).expand(x.shape[0], -1, -1) # (B, C, 3)

        outputs = self.encoder(x, pos=pos, return_output=True)

        if isinstance(outputs, (list, tuple)):
            # Stack all layer outputs → (B, L*tokens, dim), then mean-pool
            massive = torch.cat(list(outputs), dim=1)  # (B, L*T, dim)
            emb = massive.mean(dim=1)                  # (B, dim)
        elif hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state.mean(dim=1)
        else:
            emb = outputs.mean(dim=1)

        return emb.cpu().float().numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction (lazy, batched)
# ──────────────────────────────────────────────────────────────────────────────
import functools

@functools.lru_cache(maxsize=64)
def _read_epochs_cached(path):
    return mne.read_epochs(path, preload=True, verbose=False)


def extract_embeddings(model: FrozenREVE, epoch_meta: list,
                       ch_names: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Return (N, dim) array of REVE embeddings for all epochs."""
    all_embs = []
    buf_data, buf_idx = [], []

    def flush():
        if not buf_data:
            return
        batch = torch.stack(buf_data).to(DEVICE)       # (B, C, T)
        embs  = model.embed(batch, ch_names)            # (B, dim)
        all_embs.extend(embs)
        buf_data.clear()
        buf_idx.clear()

    print(f"  Extracting embeddings for {len(epoch_meta)} epochs ...")
    for i, (fif_path, local_idx) in enumerate(tqdm(epoch_meta)):
        epochs_obj = _read_epochs_cached(fif_path)
        raw_data   = epochs_obj.get_data(item=local_idx)[0]  # (C, T)

        # Ensure correct length
        if raw_data.shape[1] > TARGET_LEN:
            raw_data = raw_data[:, :TARGET_LEN]
        elif raw_data.shape[1] < TARGET_LEN:
            pad = TARGET_LEN - raw_data.shape[1]
            raw_data = np.pad(raw_data, ((0, 0), (0, pad)))

        tensor = preprocess_signal(raw_data)   # (C, T)
        buf_data.append(tensor)
        buf_idx.append(i)

        if len(buf_data) >= batch_size:
            flush()

    flush()
    return np.array(all_embs, dtype=np.float32)   # (N, dim)

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_condition(X: np.ndarray, y: np.ndarray,
                       subjects: np.ndarray, condition: str) -> list[dict]:
    """5-fold CV with LR head.  Returns list of per-fold result dicts."""
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
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        y_pred  = clf.predict(X_val)
        y_prob  = clf.predict_proba(X_val)[:, 1]

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
    p = argparse.ArgumentParser(description="REVE original-weights evaluation (no fine-tuning)")
    p.add_argument("--data-root",    default=DATA_ROOT)
    p.add_argument("--label-csv",    default=LABEL_CSV)
    p.add_argument("--out-csv",      default=OUT_CSV)
    p.add_argument("--model-size",   default="base", choices=["base", "large"])
    p.add_argument("--conditions",   nargs="+", default=None,
                   help="Conditions to evaluate (default: all 8)")
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()

    conditions = args.conditions if args.conditions else ALL_CONDITIONS
    print(f"Conditions to evaluate: {conditions}")
    print(f"Device: {DEVICE}")

    # Load labels
    label_map = load_label_map(args.label_csv)

    # Load REVE model (frozen)
    model = FrozenREVE(model_size=args.model_size)
    model.to(DEVICE)
    model.eval()

    all_rows = []

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

        # 2. Extract frozen REVE embeddings
        X = extract_embeddings(model, epoch_meta, ch_names,
                               batch_size=args.batch_size)
        print(f"  Embeddings shape: {X.shape}")

        # 3. 5-fold LR evaluation
        rows = evaluate_condition(X, y, subjects, cond)
        if not rows:
            continue

        # 4. Compute per-condition summary
        metrics_cols = ["Val_AUC", "Val_Acc", "Val_BAcc", "Val_F1"]
        fold_df = pd.DataFrame(rows)
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
