import yaml
import os
import subprocess
from copy import deepcopy

# List of all 8 physiological conditions
CONDITIONS = [
    "EC_baseline", "EO_baseline",
    "HV_EC", "HV_EO",
    "PostHV_EO", "PostHV_EC",
    "PHOTO_EC", "PHOTO_EO",
]

BASE_CONFIG = "config_feature_importance.yml"
OUT_DIR = "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/multi_condition"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load the base configuration
    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    for cond in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"=== Running Pipeline for Condition: {cond} ===")
        print(f"{'='*60}")
        
        cond_cfg = deepcopy(cfg)
        
        # Prevent files from overwriting each other
        cond_cfg["global_experiment_id"] = f"epilepsy_lasso_{cond}"
        cond_cfg["results_dir"] = OUT_DIR
        cond_cfg["results_file"] = f"feature_importance_{cond}"
        
        # Dynamically change the row_filter for condition
        for analysis in cond_cfg.get("analyses", []):
            for row_filter in analysis.get("row_filter", []):
                if row_filter.get("column") == "condition":
                    row_filter["values"] = cond

        # Save to a temporary yaml file to feed into the pipeline
        tmp_cfg_path = f"tmp_config_{cond}.yml"
        with open(tmp_cfg_path, "w") as f:
            yaml.dump(cond_cfg, f, default_flow_style=False)
            
        # Execute the main pipeline using the temporary config
        cmd = ["python3", "run_ml_pipe.py", "--config", tmp_cfg_path]
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing condition {cond}: {e}")
        finally:
            # Clean up the temporary config file
            if os.path.exists(tmp_cfg_path):
                os.remove(tmp_cfg_path)
                
        print(f"Finished condition: {cond}\n")

if __name__ == "__main__":
    main()
