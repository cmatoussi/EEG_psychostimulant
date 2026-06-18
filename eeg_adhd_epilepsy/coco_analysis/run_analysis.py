import argparse
import logging
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut

# coco_pipe imports
from coco_pipe.decoding.configs import (
    ExperimentConfig,
    CVConfig,
    FrozenBackboneDecoderConfig,
    FoundationEmbeddingModelConfig,
    NeuralFineTuneConfig,
    LoRAConfig,
    TrainerConfig,
    ModelConfig
)
from coco_pipe.decoding.experiments import Experiment

# dim_reduction imports
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.artifacts import save_fit_artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region Mapping Logic
# ---------------------------------------------------------------------------
REGION_MAP = {
    "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Temporal": ["T3", "T4", "T5", "T6"],
    "Parietal": ["P3", "Pz", "P4"],
    "Occipital": ["O1", "O2"]
}

def map_sensor_to_quadrants(X_df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Given a dataframe of sensor_*.csv features where columns look like
    'Fp1_alpha', 'Fp2_alpha', slices the dataframe to only include
    the columns belonging to the specified region.
    """
    if region not in REGION_MAP:
        raise ValueError(f"Unknown region {region}")
    valid_sensors = REGION_MAP[region]
    
    # Keep columns that start with any of the valid sensors + "_"
    keep_cols = [c for c in X_df.columns if any(c.startswith(f"{s}_") for s in valid_sensors)]
    return X_df[keep_cols]

# ---------------------------------------------------------------------------
# Mode Handlers
# ---------------------------------------------------------------------------

def run_fm_embed(analysis_cfg, X, y, groups, output_dir, unique_subjects):
    """Mode 1: Frozen backbone -> embed -> classical CV"""
    pass

def run_fm_lora(analysis_cfg, X, y, groups, output_dir):
    """Mode 2: Neural Finetuning with LoRA"""
    pass

def run_handcrafted(analysis_cfg, X, y, groups, output_dir):
    """Mode 3: Classical ML on Handcrafted Features"""
    pass

def run_dim_reduction(analysis_cfg, X, y, groups, ids, output_dir):
    """Mode 4: Dimensionality Reduction using coco_pipe.dim_reduction"""
    models_cfg = analysis_cfg.get("models", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, cfg in models_cfg.items():
        reducer_method = cfg["method"]
        n_components = cfg["n_components"]
        
        logger.info(f"Running dim_reduction: {reducer_method} (n_components={n_components})")
        
        reducer = DimReduction(method=reducer_method, n_components=n_components)
        
        # Additional params like n_neighbors
        for k, v in cfg.items():
            if k not in ["method", "n_components"]:
                if hasattr(reducer.reducer, k):
                    setattr(reducer.reducer, k, v)
        
        embedding = reducer.fit_transform(X)
        score_payload = reducer.score(embedding, X=X)
        score_metrics = dict(reducer.get_metrics())
        
        # Save fit artifact
        stem = f"dimred_{reducer_method}"
        fit_payload = {"reducer": reducer_method, "n_components": n_components, "artifact_stem": stem}
        
        artifact_path = output_dir / f"reducer-{reducer_method}_comp-{n_components}"
        artifact_path.mkdir(exist_ok=True)
        
        save_fit_artifact(
            path=artifact_path,
            embedding=embedding,
            ids=ids,
            fit_payload=fit_payload,
            metrics_payload=score_metrics,
            diagnostics={"score_payload": score_payload}
        )

# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified EEG Epilepsy pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiments_config.yaml")
    parser.add_argument("--analysis-id", type=str, required=True, help="ID of the analysis block to run")
    parser.add_argument("--group-id", type=str, required=True, help="ID of the demographic group")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Parse configs
    # ... mock for now

if __name__ == "__main__":
    main()
