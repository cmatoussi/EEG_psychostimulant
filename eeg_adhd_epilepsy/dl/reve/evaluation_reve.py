#!/usr/bin/env python3
"""
Evaluation script for REVE and CBraMod embeddings.
Supports filtering by model, size, stage, and pooling/layers.
Computes classification metrics using Logistic Regression or Random Forest.
Optimized for memory efficiency during condition discovery.
"""

import argparse
import csv
import fcntl
import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
COCO_PIPE_ROOT = Path("/home/mat/projects/coco-pipe")

for path in [PROJECT_ROOT, COCO_PIPE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eeg_adhd_epilepsy.dl.temp_loader import load_temp_dl_data
from eeg_adhd_epilepsy.analysis.utils import apply_representation
from eeg_adhd_epilepsy.io.patients import load_raw_patients_df, clean_patients_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embeddings with classification")
    parser.add_argument("--embeddings_dir", type=str, default="/home/mat/scratch/extracted_embeddings", help="Path to extracted .npy files root")
    parser.add_argument("--metadata", type=str, default="/home/mat/scratch/epilepsy_label_cleaned.csv", help="Path to clinical CSV")
    parser.add_argument("--target_col", type=str, default="has_epilepsy", help="Target column for classification")
    parser.add_argument("--model", type=str, choices=["reve", "cbramod"], default="reve", help="Model used (reve or cbramod)")
    parser.add_argument("--model_size", type=str, default="base", help="Model size filter (e.g. base, large)")
    parser.add_argument("--stage", type=str, default="baseline", help="Preprocessing stage filter (e.g. baseline, denoiseAr)")
    parser.add_argument("--pooling", type=str, default="pool", help="Pooling filter (REVE: pool/no_pool, CBraMod: NA)")
    parser.add_argument("--all_layers", action="store_true", help="Use all_layers embeddings for CBraMod")
    parser.add_argument("--classifier", type=str, choices=["lr", "rf"], default="lr", help="Classifier type")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--conditions", nargs="+", default=None, help="Specific conditions to evaluate.")
    parser.add_argument("--representation", type=str, choices=["epoch_flat", "subject_flat"], default="epoch_flat", help="Aggregation strategy")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of subjects to load")
    parser.add_argument(
        "--csv_out",
        type=str,
        default="/home/mat/projects/EEG_psychostimulant/data/results/eval/embeddings_evaluation_epilepsy.csv",
        help="Path to shared CSV file to append results to"
    )
    return parser.parse_args()

def discover_conditions(model_dir: Path, metadata_df: pd.DataFrame, model: str, desc: str, all_layers: bool) -> List[str]:
    """
    Lightning-fast discovery of conditions by targeting specific subject metadata files.
    Avoids filesystem scanning (rglob) which is slow on large datasets.
    """
    conditions: Set[str] = set()
    logger.info(f"Rapid scanning for {model} conditions...")
    
    # Try the first 20 subjects to ensure we find a valid metadata file
    sample_subjects = metadata_df["Study ID"].head(20).tolist()
    
    for sub_id in sample_subjects:
        sub_id_str = f"{int(sub_id):04d}"
        sub_dir = model_dir / f"sub-{sub_id_str}"
        
        # Determine the expected JSON filename based on model and all_layers
        if model == "cbramod":
            suffix = "_all_layers" if all_layers else ""
            json_name = f"sub-{sub_id_str}_desc-{desc}_metadata_cbramod{suffix}.json"
        else:
            # REVE naming convention (fall back to common patterns)
            json_name = f"sub-{sub_id_str}_condition_labels.json"
            
        json_path = sub_dir / json_name
        
        # If the preferred name doesn't exist, try a few common fallbacks
        if not json_path.exists():
            fallbacks = [
                sub_dir / f"sub-{sub_id_str}_desc-{desc}_metadata_{model}.json",
                sub_dir / f"sub-{sub_id_str}_condition_labels.json"
            ]
            for fb in fallbacks:
                if fb.exists():
                    json_path = fb
                    break
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    labels = json.load(f)
                    if isinstance(labels, dict):
                        conditions.update(labels.keys())
                    elif isinstance(labels, list):
                        conditions.update(labels)
                if conditions: # Found them!
                    break
            except Exception:
                continue
            
    return sorted(list(conditions))

def run_evaluation(args):
    # 1. Load Metadata
    logger.info(f"Loading metadata from {args.metadata}...")
    try:
        raw_meta_df = load_raw_patients_df(Path(args.metadata))
        clean_meta_df, _ = clean_patients_df(raw_meta_df)
        if args.limit:
            clean_meta_df = clean_meta_df.head(args.limit)
        logger.info(f"Loaded metadata for {len(clean_meta_df)} subjects.")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return

    # 2. Model-Specific Argument Handling
    model_dir = Path(args.embeddings_dir) / args.model
    
    if args.model == "cbramod":
        # CBraMod logic: internal parameters are None to avoid REVE-specific filtering
        internal_size = None 
        internal_pooling = None
    else:
        # REVE logic
        internal_size = args.model_size
        internal_pooling = "with" if args.pooling == "pool" else "without"

    # 3. Discover available conditions if none specified
    if not args.conditions:
        target_conditions = discover_conditions(
            model_dir, 
            clean_meta_df, 
            args.model, 
            args.stage, 
            args.all_layers
        )
        if not target_conditions:
            logger.error("No conditions found. Check if metadata JSON files exist in subject folders.")
            return
        logger.info(f"Discovered conditions: {target_conditions}")
    else:
        target_conditions = args.conditions

    # 4. Iterate through conditions
    for condition in target_conditions:
        print(f"\n" + "="*60)
        print(f" Evaluating Condition: {condition} ".center(60, "="))
        print(f" Model: {args.model} | Size: {args.model_size} | Pooling: {args.pooling} ".center(60, "-"))
        print("="*60)

        # 4.1 Load data for specific condition
        try:
            container = load_temp_dl_data(
                embeddings_root=model_dir,
                segments_root=None,
                model=args.model,
                desc=args.stage,
                metadata_df=clean_meta_df,
                subject_col="Study ID",
                target_col=args.target_col,
                subjects=clean_meta_df["Study ID"].tolist(),
                conditions=[condition],
                model_size=internal_size,
                pooling=internal_pooling,
                all_layers=args.all_layers,
                drop_unassigned=True
            )
        except Exception as e:
            logger.warning(f"Could not load data for {condition}: {e}")
            continue

        if container.X.size == 0:
            logger.warning(f"No data found for condition {condition}. Skipping.")
            continue

        # 4.2 Apply Representation
        try:
            container = apply_representation(
                container, 
                representation=args.representation, 
                study_id_col="Study ID"
            )
        except Exception as e:
            logger.error(f"Failed to apply representation {args.representation} for {condition}: {e}")
            continue

        # 4.3 Classification Setup
        X = container.X
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        y = np.asarray(container.y).astype(str)
        groups = np.asarray(container.coords["Study ID"]).astype(str)

        if len(np.unique(y)) < 2:
            print(f"Skipping {condition}: only one class present in data ({np.unique(y)})")
            continue

        print(f"Data shape: {X.shape}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        # 4.4 Cross-Validation
        skf = StratifiedGroupKFold(n_splits=args.n_splits)
        metrics = {"accuracy": [], "balanced_accuracy": [], "f1": []}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if args.classifier == "lr":
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
            else:
                clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
            metrics["f1"].append(f1_score(y_test, y_pred, average="macro"))

        # 4.5 Calculate Averages
        avg_acc = np.mean(metrics["accuracy"])
        avg_bal_acc = np.mean(metrics["balanced_accuracy"])
        avg_f1 = np.mean(metrics["f1"])

        print(f"Results for {condition}:")
        print(f"  Accuracy:          {avg_acc:.4f}")
        print(f"  Balanced Accuracy: {avg_bal_acc:.4f}")
        print(f"  F1 (Macro):        {avg_f1:.4f}")

        # 4.6 Save to CSV
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        row = {
            "model": args.model,
            "size": args.model_size if args.model == "reve" else "NA",
            "pooling": args.pooling if args.model == "reve" else "NA",
            "last_layer": ("no" if args.all_layers else "yes") if args.model == "cbramod" else "NA",
            "representation": args.representation,
            "condition": condition,
            "balanced accuracy": round(float(avg_bal_acc), 4),
            "accuracy": round(float(avg_acc), 4),
            "f1": round(float(avg_f1), 4),
        }

        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            fcntl.flock(f, fcntl.LOCK_UN)

    print(f"\nAll evaluations complete. Unified results saved to: {args.csv_out}")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
