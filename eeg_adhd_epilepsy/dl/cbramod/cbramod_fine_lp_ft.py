import os
import sys
import random
import copy
import math
import argparse
import json
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from einops.layers.torch import Rearrange
import mne
from tqdm import tqdm

# Add CBraMod to sys path to import models
CBRAMOD_PATH = "/home/mat/CBraMod/"
if CBRAMOD_PATH not in sys.path:
    sys.path.append(CBRAMOD_PATH)

from models.cbramod import CBraMod

# ==========================================
# 0. Configuration & Hyperparameters
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fixed Hyperparameters (requested by USER)
BATCH_SIZE = 64
EPOCHS_STEP1 = 5
EPOCHS_STEP2 = 25 
LR_STEP1 = 1e-3
LR_STEP2 = 1e-4
WEIGHT_DECAY = 5e-2
FS_TARGET = 200
INPUT_SEC = 10  # 10 second recordings
PATCH_SIZE = 200 # 1 second patches
NUM_SEGMENTS = 10 # 10s / 1s
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
import functools

@functools.lru_cache(maxsize=32)
def _read_epochs_cached(file_path):
    return mne.read_epochs(file_path, preload=True, verbose=False)

class CBraModDataset(Dataset):
    def __init__(self, epoch_files, labels, subject_ids, num_segments=10, patch_size=200):
        self.epoch_files = epoch_files
        self.labels = torch.LongTensor(labels)
        self.subjects = subject_ids
        self.num_segments = num_segments
        self.patch_size = patch_size
        
    def __len__(self): 
        return len(self.epoch_files)
        
    def __getitem__(self, idx):
        file_path, local_idx = self.epoch_files[idx]
        label = self.labels[idx]
        
        epochs_obj = _read_epochs_cached(str(file_path))
        data = epochs_obj.get_data(item=local_idx)[0]
        
        num_channels = data.shape[0]
        total_points = data.shape[1]
        
        expected_points = self.num_segments * self.patch_size
        if total_points > expected_points:
            data = data[:, :expected_points]
        elif total_points < expected_points:
            pad_width = expected_points - total_points
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        data = (data - mean) / (std + 1e-6)
        
        data = data.reshape(num_channels, self.num_segments, self.patch_size)
        return torch.FloatTensor(data), label, self.subjects[idx]

def get_data_indices(data_root, label_csv, target_conditions=None):
    labels_df = pd.read_csv(label_csv)
    labels_df['Study ID'] = labels_df['Study ID'].astype(int)
    label_map = dict(zip(labels_df['Study ID'], labels_df['Epilepsy']))

    epoch_metadata = []
    all_labels = []
    all_subjects = []
    ch_names = None
    data_path = Path(data_root)
    
    if target_conditions and (len(target_conditions) == 0 or "all" in [c.lower() for c in target_conditions]):
        target_conditions = None
        
    print(f"Mapping recordings for {len(label_map)} subjects...")
    for sub_id, label in tqdm(label_map.items()):
        if pd.isna(label): continue
        sub_id_str = f"{sub_id:04d}"
        fif_path = data_path / f"sub-{sub_id_str}" / "eeg" / f"sub-{sub_id_str}_desc-base_epo.fif"
        
        if not fif_path.exists(): continue
        
        try:
            epochs = mne.read_epochs(str(fif_path), preload=False, verbose=False)
            if target_conditions:
                present_conditions = [c for c in target_conditions if c in epochs.event_id]
                if not present_conditions: continue
                epochs = epochs[present_conditions]
            
            n_epochs = len(epochs)
            if n_epochs == 0: continue
            if ch_names is None: ch_names = epochs.ch_names
                
            for i in range(n_epochs):
                epoch_metadata.append((str(fif_path), i))
                all_labels.append(int(label))
                all_subjects.append(sub_id)
        except Exception: continue
            
    print(f"Mapped {len(epoch_metadata)} total epochs from {len(set(all_subjects))} subjects.")
    return epoch_metadata, np.array(all_labels), np.array(all_subjects), ch_names

# ==========================================
# 2. Model Architecture
# ==========================================
class CBraModClassifier(nn.Module):
    def __init__(self, num_channels, num_segments, patch_size, num_classes=2, dropout=0.1):
        super().__init__()
        self.backbone = CBraMod(in_dim=patch_size, out_dim=patch_size, d_model=patch_size)
        flat_dim = num_channels * num_segments * patch_size
        self.classifier = nn.Sequential(
            Rearrange('b c s p -> b (c s p)'),
            nn.Linear(flat_dim, 4 * patch_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * patch_size, patch_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_size, num_classes),
        )

    def load_pretrained(self, weights_path):
        print(f"Loading pretrained weights from {weights_path}...")
        self.backbone.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
        self.backbone.proj_out = nn.Identity()
        print("Backbone prepared for feature extraction.")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# ==========================================
# 3. Training Utilities
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_auc, model):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose: print(f"      EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model_state = copy.deepcopy(model.state_dict())

def train_fold(fold_idx, train_loader, val_loader, num_channels, num_segments, patch_size, weights_path, 
               lr_step1=LR_STEP1, lr_step2=LR_STEP2, weight_decay=WEIGHT_DECAY, dropout=0.1, patience=10, 
               class_weights=None):
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    model = CBraModClassifier(num_channels, num_segments, patch_size, NUM_CLASSES, dropout=dropout).to(DEVICE)
    if weights_path:
        model.load_pretrained(weights_path)
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # PHASE 1: Linear Probing
    print(f"      Step 1: Linear Probing (Head Only)...")
    model.backbone.requires_grad_(False)
    optimizer1 = optim.AdamW(model.classifier.parameters(), lr=lr_step1, weight_decay=weight_decay)
    
    for epoch in range(EPOCHS_STEP1):
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer1.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.update()
            train_loss += loss.item()
        print(f"        Step 1 Epoch {epoch+1}/{EPOCHS_STEP1} | Loss: {train_loss/len(train_loader):.4f}")

    # PHASE 2: Global Fine-tuning
    print(f"      Step 2: Global Fine-tuning (Backbone Unfrozen)...")
    model.backbone.requires_grad_(True)
    optimizer2 = optim.AdamW(model.parameters(), lr=lr_step2, weight_decay=weight_decay)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.5, patience=math.ceil(patience/2))
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_val_auc = 0
    best_metrics = {}

    for epoch in range(EPOCHS_STEP2):
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer2.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer2)
            scaler.update()
            train_loss += loss.item()
            
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        all_preds = (all_probs >= 0.5).astype(int)
        
        try: val_auc = roc_auc_score(all_targets, all_probs)
        except ValueError: val_auc = 0.5 
            
        val_acc = accuracy_score(all_targets, all_preds)
        val_bacc = balanced_accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds)
        
        scheduler2.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {'auc': val_auc, 'acc': val_acc, 'bacc': val_bacc, 'f1': val_f1}
            
        print(f"      Step 2 Epoch {epoch+1}/{EPOCHS_STEP2} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val BAcc: {val_bacc:.4f}")
        
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("      Early stopping triggered. Restoring best weights.")
            break

    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    return model, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/mat/scratch/preproc/")
    parser.add_argument("--label-csv", default="/home/mat/scratch/epilepsy_label_cleaned.csv")
    parser.add_argument("--weights-path", default="/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--save-dir", default="/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod/lp_ft/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold (0-4) for job arrays")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_metadata, y, subjects, ch_names = get_data_indices(args.data_root, args.label_csv, args.conditions)
    num_channels = len(ch_names)
    cond_suffix = "_".join(args.conditions) if args.conditions and "all" not in [c.lower() for c in args.conditions] else "all"
    
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    epoch_metadata = np.array(epoch_metadata, dtype=object)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(epoch_metadata, y, groups=subjects)):
        if args.fold is not None and fold != args.fold: continue
            
        print(f"\n<<< FOLD {fold+1} (Fixed Params) >>>")
        save_path = save_dir / f"cbramod_fold{fold+1}_{cond_suffix}_lp_ft.pt"
        metrics_path = save_dir / f"cbramod_metrics_fold{fold+1}_{cond_suffix}_lp_ft.json"

        if save_path.exists() and metrics_path.exists():
            print(f"      Fold {fold+1} already completed.")
            with open(metrics_path, 'r') as f:
                fold_results.append(json.load(f)['metrics'])
            continue

        meta_train, y_train, sub_train = epoch_metadata[train_idx], y[train_idx], subjects[train_idx]
        meta_val, y_val, sub_val = epoch_metadata[val_idx], y[val_idx], subjects[val_idx]
        
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        class_weights = torch.FloatTensor(weights).to(DEVICE)

        train_ds = CBraModDataset(meta_train, y_train, sub_train, num_segments=NUM_SEGMENTS, patch_size=PATCH_SIZE)
        val_ds = CBraModDataset(meta_val, y_val, sub_val, num_segments=NUM_SEGMENTS, patch_size=PATCH_SIZE)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
        
        # Use fixed hyperparameters as requested
        model, metrics = train_fold(fold, train_loader, val_loader, 
                                     num_channels, NUM_SEGMENTS, PATCH_SIZE, 
                                     args.weights_path, 
                                     lr_step1=LR_STEP1, lr_step2=LR_STEP2, 
                                     weight_decay=WEIGHT_DECAY, 
                                     dropout=0.1, patience=args.patience,
                                     class_weights=class_weights)
        fold_results.append(metrics)
        
        save_data = {
            "metrics": metrics,
            "params": {"lr_step1": LR_STEP1, "lr_step2": LR_STEP2, "wd": WEIGHT_DECAY, "dropout": 0.1},
            "condition": cond_suffix,
            "fold": fold + 1
        }
        
        torch.save(model.state_dict(), save_path)
        with open(metrics_path, 'w') as f:
            json.dump(save_data, f, indent=4)
            
    # Final Results CSV logic with Mean/Std
    # Always write when running as a single-fold job array task (args.fold is not None)
    # or when all 5 folds ran in the same process
    if len(fold_results) > 0:
        results_csv = save_dir / "finetune_results_cbramod_lp_ft.csv"
        rows = []
        for i, m in enumerate(fold_results):
            rows.append({
                'Fold': f"Fold {i+1}",
                'Condition': cond_suffix,
                'Val_AUC': m['auc'], 'Val_Acc': m['acc'],
                'Val_BAcc': m['bacc'], 'Val_F1': m['f1']
            })
        
        results_df = pd.DataFrame(rows)
        
        # Add Mean and Std rows if all 5 folds are available
        if len(results_df) >= 5:
            metrics_cols = ['Val_AUC', 'Val_Acc', 'Val_BAcc', 'Val_F1']
            mean_vals = results_df[metrics_cols].mean()
            std_vals = results_df[metrics_cols].std()
            
            summary_rows = [
                {'Fold': 'Average', 'Condition': cond_suffix, **mean_vals.to_dict()},
                {'Fold': 'Std', 'Condition': cond_suffix, **std_vals.to_dict()}
            ]
            results_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)

        with open(results_csv, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            is_empty = len(f.read().strip()) == 0
            f.seek(0, 2) 
            results_df.to_csv(f, index=False, header=is_empty)
            fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == "__main__":
    main()
