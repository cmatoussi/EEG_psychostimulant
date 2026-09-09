import os
import sys
import random
import copy
import math
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from transformers import AutoModel
import mne
from tqdm import tqdm

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

# Default Hyperparameters (as per instruction/example)
# Optimized Hyperparameters
BATCH_SIZE = 64
EPOCHS_STEP1 = 5
LR_STEP1 = 5e-4
EPOCHS_STEP2 = 15
LR_STEP2 = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
MIXUP_ALPHA = 0.4
FS_TARGET = 200
INPUT_SEC = 10  # Detected from .fif metadata
TARGET_LEN = int(FS_TARGET * INPUT_SEC)
NUM_CLASSES = 2

# Model IDs
REVE_MODEL_ID = "brain-bzh/reve-base"
POS_MODEL_ID = "brain-bzh/reve-positions"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Custom LoRA Implementation
# ==========================================
class LoRALayer(nn.Module):
    """
    Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA).
    Wraps existing linear layers in the transformer blocks.
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze original weights
        self.weight = nn.Parameter(original_layer.weight.detach(), requires_grad=False)
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.detach(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + (lora_out * self.scaling)

def apply_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA):
    """Injects LoRA layers into the attention blocks (QKVO projection layers)."""
    target_modules = ['to_qkv', 'to_out'] 
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            
    # Ensure only LoRA parameters are trainable
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model

# ==========================================
# 2. Data Loading & Augmentation
# ==========================================
import functools

@functools.lru_cache(maxsize=32)
def _read_epochs_cached(file_path):
    return mne.read_epochs(file_path, preload=True, verbose=False)

class EEGEpochDataset(Dataset):
    """
    Lazy-loading Dataset for EEG epochs to prevent OOM errors.
    Uses an LRU cache to speed up batch processing.
    """
    def __init__(self, epoch_files, labels, subject_ids):
        # epoch_files: list of (file_path, local_epoch_idx)
        self.epoch_files = epoch_metadata = epoch_files
        self.labels = torch.LongTensor(labels)
        self.subjects = subject_ids
        
    def __len__(self): 
        return len(self.epoch_files)
        
    def __getitem__(self, idx):
        file_path, local_idx = self.epoch_files[idx]
        label = self.labels[idx]
        
        epochs_obj = _read_epochs_cached(str(file_path))
        data = epochs_obj.get_data(item=local_idx)[0] # (channels, times)
        
        # Z-score normalization
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        data = (data - mean) / (std + 1e-6)
        
        return torch.FloatTensor(data), label, self.subjects[idx]

def intra_class_mixup(x, y, alpha=0.4):
    """
    Augmentation that mixes two samples from the same class.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.arange(batch_size).to(x.device)
    
    for i in range(batch_size):
        same_class_indices = torch.where(y == y[i])[0]
        if len(same_class_indices) > 1:
            idx = same_class_indices[torch.randint(0, len(same_class_indices), (1,))].item()
            index[i] = idx
            
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y

def get_data_indices(data_root, label_csv, target_conditions=None):
    """
    Discovers all epochs across subjects and maps them to labels.
    Uses a cache file to avoid slow rescanning of 1000+ files.
    """
    cond_tag = "_".join(sorted(target_conditions)) if target_conditions else "all"
    cache_path = Path(data_root) / f"epoch_metadata_cache_{cond_tag}.json"
    
    if cache_path.exists():
        print(f"Loading cached metadata from {cache_path}...")
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            return cache_data['metadata'], np.array(cache_data['labels']), np.array(cache_data['subjects']), cache_data['ch_names']
        except Exception as e:
            print(f"Warning: Cached metadata failed to load: {e}. Rescanning...")

    print(f"Loading labels from {label_csv}...")
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
        
    print(f"Scanning recordings for {len(label_map)} subjects in {data_root}...")
    for sub_id, label in tqdm(label_map.items()):
        if pd.isna(label): continue
        sub_id_str = f"{sub_id:04d}"
        sub_dir = data_path / f"sub-{sub_id_str}"
        fif_path = sub_dir / "eeg" / f"sub-{sub_id_str}_desc-base_epo.fif"
        
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
        except Exception as e:
            print(f"Error mapping {sub_id_str}: {e}")
            
    print(f"Mapped {len(epoch_metadata)} total epochs from {len(set(all_subjects))} subjects.")
    
    # Save to cache for next run/fold
    try:
        cache_data = {
            'metadata': epoch_metadata,
            'labels': all_labels,
            'subjects': all_subjects,
            'ch_names': ch_names
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"Metadata cached to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save metadata cache: {e}")

    return epoch_metadata, np.array(all_labels), np.array(all_subjects), ch_names

# ==========================================
# 3. Model Architecture
# ==========================================
class ReveClassifier(nn.Module):
    def __init__(self, ch_names, num_classes=2):
        super().__init__()
        # Load from HF with trust_remote_code
        self.backbone = AutoModel.from_pretrained(REVE_MODEL_ID, trust_remote_code=True, token=True)
        self.pos_bank = AutoModel.from_pretrained(POS_MODEL_ID, trust_remote_code=True, token=True)

        # Freeze by default
        for param in self.backbone.parameters():
            param.requires_grad = False 
        for param in self.pos_bank.parameters():
            param.requires_grad = False
            
        # Initialize positions for the specific channel set
        pos_embeddings = self.pos_bank(ch_names).unsqueeze(0) # (1, C, 3)
        self.register_buffer('positions', pos_embeddings)
        
        # Classifier Head (pooling is often handled by REVE natively or via mean tokens)
        # REVE hidden_size is typically 512 for base, 1216 for large (called embed_dim)
        if hasattr(self.backbone.config, 'embed_dim'):
            self.dim = self.backbone.config.embed_dim
        elif hasattr(self.backbone.config, 'hidden_size'):
            self.dim = self.backbone.config.hidden_size
        else:
            self.dim = 512
        
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dim * len(ch_names), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        pos = self.positions.repeat(batch_size, 1, 1).to(x.device)
        num_channels = self.positions.size(1)
        
        # REVE forward usually returns a list of layers if return_output=True
        # Or a single tensor if return_output=False (default)
        outputs = self.backbone(x, pos=pos)
        
        # Get last hidden state
        if isinstance(outputs, (list, tuple)):
            feat = outputs[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            feat = outputs.last_hidden_state
        else:
            feat = outputs

        # Pooling: tokens are (Batch, C * P, Dim)
        # We need to mean pool over P (patches per channel)
        # Reshape to (Batch, num_channels, num_patches, Dim)
        feat = feat.view(batch_size, num_channels, -1, self.dim)
        feat = feat.mean(dim=2) # mean pool over patches -> (Batch, num_channels, Dim)

        logits = self.classifier_head(feat)
        return logits

def soup_models(best_state, final_state, alpha=0.5):
    """Average weights to smooth the landscape and improve generalization."""
    souped_state = {}
    for key in best_state.keys():
        souped_state[key] = alpha * best_state[key].float() + (1 - alpha) * final_state[key].float()
    return souped_state

# ==========================================
# 4. Training Pipeline
# ==========================================
def train_fold(fold_idx, train_loader, val_loader, ch_names, class_weights=None):
    model = ReveClassifier(ch_names=ch_names, num_classes=NUM_CLASSES).to(DEVICE)
    # Step 1 & 2: We removed the weighted loss to prevent double-balancing with the sampler
    criterion = nn.CrossEntropyLoss()

    # PHASE 1: Linear Probing
    print(f"      Step 1: Linear Probing (Head Only)...")
    optimizer1 = optim.AdamW(model.classifier_head.parameters(), lr=LR_STEP1, weight_decay=0.01)
    
    for epoch in range(EPOCHS_STEP1):
        model.train()
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer1.zero_grad()
            
            # Mixup
            mixed_x, _ = intra_class_mixup(inputs, targets, MIXUP_ALPHA)
            combined_x = torch.cat([inputs, mixed_x], dim=0)
            combined_y = torch.cat([targets, targets], dim=0)
            
            logits = model(combined_x)
            loss = criterion(logits, combined_y)
            loss.backward()
            optimizer1.step()

    # PHASE 2: LoRA Fine-tuning
    print(f"      Step 2: Injecting LoRA Adaptors & Fine-tuning...")
    model.backbone = apply_lora(model.backbone, rank=LORA_RANK, alpha=LORA_ALPHA)
    model.to(DEVICE) # Ensure new parameters are on device
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = optim.AdamW(trainable_params, lr=LR_STEP2, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.5, patience=3)

    best_val_auc = 0
    best_model_state = None

    for epoch in range(EPOCHS_STEP2):
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer2.zero_grad()
            
            mixed_x, _ = intra_class_mixup(inputs, targets, MIXUP_ALPHA)
            combined_x = torch.cat([inputs, mixed_x], dim=0)
            combined_y = torch.cat([targets, targets], dim=0)
            
            logits = model(combined_x)
            loss = criterion(logits, combined_y)
            loss.backward()
            optimizer2.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        all_preds = (np.array(all_probs) >= 0.5).astype(int)
        val_auc = roc_auc_score(all_targets, all_probs)
        val_acc = accuracy_score(all_targets, all_preds)
        val_bacc = balanced_accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {
                'auc': val_auc,
                'acc': val_acc,
                'bacc': val_bacc,
                'f1': val_f1
            }
            best_model_state = copy.deepcopy(model.state_dict())
            
        print(f"      FT Epoch {epoch+1}/{EPOCHS_STEP2} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val BAcc: {val_bacc:.4f}")

    # Model Souping
    print("      Applying Model Souping...")
    final_state = model.state_dict()
    if best_model_state:
        souped_state = soup_models(best_model_state, final_state, alpha=0.5)
        model.load_state_dict(souped_state)
        
    return model, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/mat/scratch/preproc/")
    parser.add_argument("--label-csv", default="/home/mat/scratch/epilepsy_label_cleaned.csv")
    parser.add_argument("--conditions", nargs="+", default=None, 
                        help="Filtering by conditions_tested (e.g. EO_baseline EC_baseline). Pass 'all' or omit to use all conditions.")
    parser.add_argument("--save-dir", default="/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/base/balanced", help="Directory to save fine-tuned parameters.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold (0-4) for job arrays")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Map Data Indices
    epoch_metadata, y, subjects, ch_names = get_data_indices(args.data_root, args.label_csv, args.conditions)
    
    # 2. Determine Condition Suffix for naming
    if args.conditions and "all" not in [c.lower() for c in args.conditions]:
        cond_suffix = "_".join(args.conditions)
    else:
        cond_suffix = "all"
    
    # 3. Subject-wise Split (8x Train, 2x Val)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    print("\nStarting Cross-Validation (Subject-Wise)...")
    fold_results = []
    
    # Convert epoch_metadata to a numpy array for easier indexing
    epoch_metadata = np.array(epoch_metadata, dtype=object)

    for fold, (train_idx, val_idx) in enumerate(skf.split(epoch_metadata, y, groups=subjects)):
        if args.fold is not None and fold != args.fold:
            continue
            
        print(f"\n<<< FOLD {fold+1} >>>")
        
        # Define paths for checkpointing
        save_path = save_dir / f"reve_finetuned_fold{fold+1}_{cond_suffix}_balanced.pt"
        metrics_path = save_dir / f"reve_metrics_fold{fold+1}_{cond_suffix}_balanced.json"

        # Resume logic: Check if both model and metrics exist
        if save_path.exists() and metrics_path.exists():
            print(f"      Fold {fold+1} already completed. Resuming from saved results.")
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                fold_results.append(metrics)
                continue
            except Exception as e:
                print(f"      Warning: Error loading metrics for fold {fold+1}: {e}. Re-training...")

        meta_train, y_train, sub_train = epoch_metadata[train_idx], y[train_idx], subjects[train_idx]
        meta_val, y_val, sub_val = epoch_metadata[val_idx], y[val_idx], subjects[val_idx]
        
        # Verify no subject leakage
        assert set(sub_train).isdisjoint(set(sub_val)), "CRITICAL: Subject leakage detected!"
        
        train_ds = EEGEpochDataset(meta_train, y_train, sub_train)
        val_ds = EEGEpochDataset(meta_val, y_val, sub_val)
        
        # --- BALANCING LOGIC ---
        # 1. Calculate Class Weights for Loss Function
        class_counts = np.bincount(y_train)
        weights = 1.0 / class_counts
        # Normalize weights so they average to 1 (optional, but good practice)
        weights = weights / weights.sum() * len(class_counts)
        class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)
        
        # 2. Setup WeightedRandomSampler for Batch Balancing
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            sampler=sampler, 
            num_workers=8, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )
        
        model, metrics = train_fold(fold, train_loader, val_loader, ch_names, class_weights=class_weights_tensor)
        fold_results.append(metrics)
        auc = metrics['auc']
        
        # Save fold model
        torch.save(model.state_dict(), save_path)
        
        # Save fold metrics for resuming
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        print(f"      Best Val AUC: {auc:.4f} | Saved to {save_path}")

    # 3. Final Results & CSV Export (with file locking for Slurm arrays)
    if args.fold is None:
        avg_metrics = {
            'auc': np.mean([m['auc'] for m in fold_results]),
            'acc': np.mean([m['acc'] for m in fold_results]),
            'bacc': np.mean([m['bacc'] for m in fold_results]),
            'f1': np.mean([m['f1'] for m in fold_results]),
        }
        
        std_metrics = {
            'auc': np.std([m['auc'] for m in fold_results]),
            'acc': np.std([m['acc'] for m in fold_results]),
            'bacc': np.std([m['bacc'] for m in fold_results]),
            'f1': np.std([m['f1'] for m in fold_results]),
        }
    
    print("\n" + "="*50)
    print(f"Final Results (5 Folds) for {cond_suffix}:")
    print(f"  Avg AUC:  {avg_metrics['auc']:.4f} +/- {std_metrics['auc']:.4f}")
    print("="*50)
    
    rows = []
    if args.fold is not None:
        # Just log the single fold that ran
        m = fold_results[0]
        rows.append({
            'Fold': args.fold + 1,
            'Condition': cond_suffix,
            'Val_AUC': m['auc'], 'Val_Acc': m['acc'],
            'Val_BAcc': m['bacc'], 'Val_F1': m['f1']
        })
    else:
        # Log all folds plus summary
        for i, m in enumerate(fold_results):
            rows.append({
                'Fold': i + 1,
                'Condition': cond_suffix,
                'Val_AUC': m['auc'], 'Val_Acc': m['acc'],
                'Val_BAcc': m['bacc'], 'Val_F1': m['f1']
            })
        
        # Summary rows
        rows.append({
            'Fold': 'Average', 'Condition': cond_suffix,
            'Val_AUC': avg_metrics['auc'], 'Val_Acc': avg_metrics['acc'],
            'Val_BAcc': avg_metrics['bacc'], 'Val_F1': avg_metrics['f1']
        })
        rows.append({
            'Fold': 'StdDev', 'Condition': cond_suffix,
            'Val_AUC': std_metrics['auc'], 'Val_Acc': std_metrics['acc'],
            'Val_BAcc': std_metrics['bacc'], 'Val_F1': std_metrics['f1']
        })
    
    results_df = pd.DataFrame(rows)
    results_csv = save_dir / "finetune_results_reve_balanced.csv"

    # Simple append with file locking (ensures no overwriting, just adding)
    import fcntl
    print(f"Adding results for {cond_suffix} to {results_csv}...")
    
    try:
        # Open in 'a+' to create if missing and append safely
        with open(results_csv, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX) # Exclusive lock to prevent interleaving
            
            # Check if file is empty to decidce on header
            f.seek(0)
            is_empty = len(f.read().strip()) == 0
            
            # Go to end for append
            f.seek(0, 2) 
            results_df.to_csv(f, index=False, header=is_empty)
            
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN) # Unlock
        print(f"Results for {cond_suffix} added to {results_csv}")
    except Exception as e:
        print(f"Warning: Could not add results to {results_csv}: {e}")
        # Fallback
        results_df.to_csv(save_dir / f"finetune_results_{cond_suffix}.csv", index=False)

if __name__ == "__main__":
    main()
