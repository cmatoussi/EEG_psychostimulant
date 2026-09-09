#!/usr/bin/env python3
"""
Standalone Scoring for Deep Embeddings (ReVe/CBraMod).
Evaluates separation between EO and EC in high-dimensional space using Group K-Fold.
Does NOT depend on shared dimensionality reduction utilities.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig
from eeg_adhd_epilepsy.analysis.utils import apply_representation
from eeg_adhd_epilepsy.dl.temp_loader import load_temp_dl_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_scoring(X, y, groups):
    """Run Group K-Fold Logistic Regression and return Accuracy and Balanced Accuracy."""
    # Ensure labels are strings for coco-pipe
    y = np.asarray(y).ravel().astype(str)
    groups = np.asarray(groups).ravel().astype(str)
    
    # Calculate number of splits (min 5 or number of subjects per class)
    if pd.Index(y).nunique() < 2:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan}
        
    subject_labels = pd.DataFrame({"group": groups, "label": y})
    # For EO_EC, we have multiple entries per group. We want to know how many groups have both labels.
    # But more simply, StratifiedGroupKFold needs at least n_splits groups.
    unique_groups = len(np.unique(groups))
    n_splits = min(5, unique_groups // 2) if unique_groups > 4 else 2
    
    if n_splits < 2:
         return {"accuracy": np.nan, "balanced_accuracy": np.nan}

    config = ExperimentConfig(
        task="classification",
        tag="standalone-scoring",
        models={
            "logreg": LogisticRegressionConfig(
                method="LogisticRegression",
                max_iter=1000,
                class_weight="balanced",
            )
        },
        cv=CVConfig(
            strategy="stratified_group_kfold",
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ),
        metrics=["accuracy", "balanced_accuracy"],
        use_scaler=True,
        verbose=False,
        n_jobs=1,
    )
    
    experiment = Experiment(config)
    result = experiment.run(X, y, groups=groups)
    summary = result.summary()
    
    return {
        "accuracy": float(summary.loc["logreg", "accuracy_mean"]),
        "balanced_accuracy": float(summary.loc["logreg", "balanced_accuracy_mean"])
    }

def main():
    parser = argparse.ArgumentParser(description="Standalone Scoring for Deep Embeddings")
    parser.add_argument("--embeddings_dir", type=str, default="/home/mat/scratch/motor_extracted_embeddings/", help="Path to extracted .npy files")
    parser.add_argument("--model", type=str, choices=["reve", "cbramod"], required=True, help="Which model was used")
    parser.add_argument("--reve_size", type=str, choices=["base", "large"], default=None, help="REVE model size")
    parser.add_argument("--pooling", type=str, choices=["with", "without"], default=None, help="Whether embeddings are pooled")
    parser.add_argument("--desc", type=str, default="baseline", help="EEG stage descriptor")
    parser.add_argument("--representation", type=str, default="epoch_flat", choices=["epoch_flat", "subject_flat"], help="Aggregation strategy")
    parser.add_argument("--all_layers", type=int, choices=[0, 1], default=0, help="Analyze all-layer embeddings (1) or last-layer (0)")
    args = parser.parse_args()

    logger.info(f"Scoring {args.model} embeddings ({args.representation})...")

    # 1. Load Embeddings
    try:
        # EO_EC needs 'condition' as the loading target
        container = load_temp_dl_data(
            embeddings_root=Path(args.embeddings_dir),
            segments_root=None,
            model=args.model,
            desc=args.desc,
            metadata_df=None,
            subject_col="Study ID",
            target_col="condition",
            conditions=["eyes_open", "eyes_closed"],
            min_overlap_fraction=0.8,
            drop_unassigned=True,
            model_size=args.reve_size,
            pooling=args.pooling,
            all_layers=(args.all_layers == 1)
        )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return

    # 2. Filter for subjects with BOTH conditions
    all_study_ids = np.asarray(container.coords.get("Study ID")).astype(str)
    all_conditions = np.asarray(container.coords["condition"]).astype(str)
    
    subject_cond_map = {}
    for s, c in zip(all_study_ids, all_conditions):
        if s not in subject_cond_map:
            subject_cond_map[s] = set()
        subject_cond_map[s].add(c)
    
    subjects_with_both = {s for s, cs in subject_cond_map.items() if "eyes_open" in cs and "eyes_closed" in cs}
    
    final_mask = np.array([
        (c in ["eyes_open", "eyes_closed"]) and (s in subjects_with_both)
        for s, c in zip(all_study_ids, all_conditions)
    ])
    
    container = container.isel(obs=final_mask)
    logger.info(f"Final subject count (with both EO and EC): {len(subjects_with_both)}")

    # 3. Apply Representation
    # Re-extract study IDs and conditions from filtered container
    study_ids = np.asarray(container.coords.get("Study ID")).astype(str)
    conds = np.asarray(container.coords["condition"]).astype(str)
    container.coords["Study ID_condition"] = np.array([f"{s}_{c}" for s, c in zip(study_ids, conds)])
    
    try:
        rep_container = apply_representation(
            container, 
            representation=args.representation, 
            study_id_col="Study ID_condition"
        )
    except Exception as e:
        logger.error(f"Failed to apply representation {args.representation}: {e}")
        return

    # 4. Extract high-dimensional X, y, and groups
    X = rep_container.X
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
        
    composite_keys = np.asarray(rep_container.coords.get("Study ID_condition")).astype(str)
    reconstructed_conditions = np.array([k.split("_", 1)[1] for k in composite_keys])
    labels = (reconstructed_conditions == "eyes_open").astype(int)
    groups = np.array([k.split("_", 1)[0] for k in composite_keys])

    logger.info(f"Computing scores on {X.shape[0]} samples, {X.shape[1]} features (Group K-Fold)...")

    # 5. Score
    scores = run_scoring(X, labels, groups)

    print("-" * 40)
    print(f"RESULTS FOR {args.model.upper()} ({args.representation})")
    print("-" * 40)
    print(f"Accuracy:          {scores['accuracy']:.4f}")
    print(f"Balanced Accuracy: {scores['balanced_accuracy']:.4f}")
    print("-" * 40)
    print(f"Samples used: {X.shape[0]}")
    print(f"Features:     {X.shape[1]}")
    print("-" * 40)

if __name__ == "__main__":
    main()
