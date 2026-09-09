import os
import subprocess
import yaml
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualize.viz_utils import make_label_map_keep_sensor

phys_conditions = ['EC_baseline', 'EO_baseline', 'HV_EC', 'PHOTO_EC', 'PostHV_EC', 'PostHV_EO', 'HV_EO', 'PHOTO_EO']

config_types = {
    "sensor_epoch": "eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_epoch.yml",
    "sensor_subject": "eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml",
    "pooled_epoch": "eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_epoch.yml",
    "pooled_subject": "eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_subject.yml"
}

def generate_and_run_configs():
    out_dir = "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/multi_condition"
    os.makedirs(out_dir, exist_ok=True)
    
    generated_pkls = {cond: {} for cond in phys_conditions}

    for phys_cond in phys_conditions:
        for ctype, base_cfg_path in config_types.items():
            if not os.path.exists(base_cfg_path):
                print(f"Skipping missing base config: {base_cfg_path}")
                continue
                
            with open(base_cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
                
            cfg['results_dir'] = out_dir
            cfg['results_file'] = f"feature_importance_lasso_{ctype}_{phys_cond}"
            cfg['global_experiment_id'] = f"epilepsy_feature_importance_lasso_{ctype}_{phys_cond}"
            
            for analysis in cfg.get('analyses', []):
                if 'row_filter' not in analysis:
                    analysis['row_filter'] = []
                    
                # Remove any existing condition filters
                analysis['row_filter'] = [rf for rf in analysis['row_filter'] if rf.get('column') != 'condition']
                
                # Add the new condition filter
                analysis['row_filter'].insert(0, {
                    'column': 'condition',
                    'values': phys_cond,
                    'operator': '=='
                })
                
            tmp_cfg_path = os.path.join(out_dir, f"tmp_config_{ctype}_{phys_cond}.yml")
            with open(tmp_cfg_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
                
            print(f"Running ML pipeline for {phys_cond} - {ctype}...")
            subprocess.run(["python3", "eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py", "--config", tmp_cfg_path], check=True)
            
            generated_pkls[phys_cond][ctype] = os.path.join(out_dir, f"epilepsy_feature_importance_lasso_{ctype}_{phys_cond}.pkl")
            
    return generated_pkls

def summarize_all(generated_pkls):
    summary_out_dir = "/home/mat/projects/EEG_psychostimulant/data/results/hand_crafted/condition_summaries"
    os.makedirs(summary_out_dir, exist_ok=True)
    
    model = "Logistic Regression"
    metric = "balanced_accuracy"

    for phys_cond, ctype_pkls in generated_pkls.items():
        summary_data = []
        md_lines = []
        md_lines.append(f"# Summary for Condition: {phys_cond}\n")
        
        for ctype, pkl_path in ctype_pkls.items():
            if not os.path.exists(pkl_path):
                continue
            all_results = pd.read_pickle(pkl_path)
            
            # Global Lasso
            res_all = all_results.get('classification_lasso_all', {}).get(model, {})
            fi = res_all.get('feature_importances', {})
            
            top_5_features = []
            if fi:
                s = pd.Series({k: v.get("weighted_mean", v.get("mean", 0.0)) for k, v in fi.items()})
                s_top = s.abs().sort_values(ascending=False).head(5)
                try:
                    label_map = make_label_map_keep_sensor(s_top.index.tolist())
                    top_5_features = [(label_map.get(k, k), s[k]) for k in s_top.index]
                except Exception:
                    top_5_features = [(k, s[k]) for k in s_top.index]
                
            # Per-Sensor/Region
            res_sens = all_results.get('classification_lasso_per_sensor', {})
            sens_metrics = {}
            for sens_key, models_res in res_sens.items():
                if model in models_res:
                    scores = models_res[model].get('metric_scores', {})
                    if metric in scores:
                        clean_sens = sens_key.replace("_ch-", "")
                        sens_metrics[clean_sens] = float(scores[metric]['mean'])
                        
            top_5_sensors = []
            if sens_metrics:
                s_sens = pd.Series(sens_metrics).sort_values(ascending=False)
                top_5_sensors = [(k, v) for k, v in s_sens.head(5).items()]
                
                plt.figure(figsize=(8, 6))
                s_plot = s_sens.head(10)
                s_plot.sort_values(ascending=True).plot(kind='barh', color='skyblue')
                sp_type = "Region" if "pooled" in ctype else "Sensor"
                plt.title(f"{phys_cond} - {ctype}: Top {sp_type}s ({metric})")
                plt.xlabel(metric.capitalize())
                plt.tight_layout()
                plt.savefig(os.path.join(summary_out_dir, f"{phys_cond}_{ctype}_{sp_type}_bar.png"), dpi=300)
                plt.close()
                
            summary_data.append({
                "Type": ctype,
                "Top_5_Features": top_5_features,
                "Top_5_Sensors": top_5_sensors,
            })
            
            md_lines.append(f"## Configuration: {ctype}")
            md_lines.append("### Top 5 Features (Global Lasso)")
            md_lines.append("| Rank | Feature | Importance (Coefficient) |")
            md_lines.append("|---|---|---|")
            for i, (feat, imp) in enumerate(top_5_features):
                md_lines.append(f"| {i+1} | {feat} | {imp:.4f} |")
            md_lines.append("\n")
            
            sp_type = "Region" if "pooled" in ctype else "Sensor"
            md_lines.append(f"### Top 5 {sp_type}s")
            md_lines.append(f"| Rank | {sp_type} | Decoding Accuracy |")
            md_lines.append("|---|---|---|")
            for i, (sens, acc) in enumerate(top_5_sensors):
                md_lines.append(f"| {i+1} | {sens} | {acc:.4f} |")
            md_lines.append("\n")
            
        with open(os.path.join(summary_out_dir, f"{phys_cond}_summary.md"), "w") as f:
            f.write("\n".join(md_lines))

if __name__ == "__main__":
    pkls = generate_and_run_configs()
    summarize_all(pkls)
    print("Done generating condition-by-condition results and summaries!")
