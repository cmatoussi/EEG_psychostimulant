import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

phys_conditions = ['EC_baseline', 'EO_baseline', 'HV_EC', 'PHOTO_EC', 'PostHV_EC', 'PostHV_EO', 'HV_EO', 'PHOTO_EO']
in_dir = "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/multi_condition"
out_dir = "/home/mat/projects/EEG_psychostimulant/data/results/hand_crafted/overall_summary"
os.makedirs(out_dir, exist_ok=True)

model = "Logistic Regression"
metric = "balanced_accuracy"

def generate_plot(config_type, is_region, color, title, out_filename):
    data = []
    for cond in phys_conditions:
        pkl_path = os.path.join(in_dir, f"epilepsy_feature_importance_lasso_{config_type}_{cond}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Missing {pkl_path}")
            continue
            
        all_results = pd.read_pickle(pkl_path)
        res_sens = all_results.get('classification_lasso_per_sensor', {})
        
        sens_metrics = {}
        for sens_key, models_res in res_sens.items():
            if model in models_res:
                scores = models_res[model].get('metric_scores', {})
                if metric in scores:
                    clean_sens = sens_key.replace("_ch-", "")
                    sens_metrics[clean_sens] = float(scores[metric]['mean'])
                    
        if sens_metrics:
            best_spatial = max(sens_metrics, key=sens_metrics.get)
            best_acc = sens_metrics[best_spatial]
            data.append((cond, best_spatial, best_acc))

    if data:
        conditions = [d[0] for d in data]
        spatials = [d[1] for d in data]
        accuracies = [d[2] for d in data]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(conditions, accuracies, color=color, edgecolor='black')
        
        for bar, spatial_name, acc in zip(bars, spatials, accuracies):
            yval = bar.get_height()
            if is_region:
                # Replace underscores with newlines so long region names don't crash into each other
                spatial_name = spatial_name.replace("_", "\n")
                
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, 
                     f"{spatial_name}:\n{acc:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
                     
        plt.ylim(0, max(accuracies) + 0.15) # Give vertical space for multiline text
        plt.ylabel(f"Decoding Accuracy ({metric})")
        plt.xlabel("Physiological Condition")
        plt.title(title)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        
        out_path = os.path.join(out_dir, out_filename)
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")
    else:
        print(f"No data found to plot for {config_type}.")

# Generate Sensor Subject Plot (Blue)
generate_plot(
    config_type="sensor_subject", 
    is_region=False, 
    color='#4c72b0', 
    title="Subject-Level Analysis: Best Sensor Decoding Accuracy per Condition",
    out_filename="Subject_Level_Best_Sensor_Per_Condition.png"
)

# Generate Pooled Subject Plot (Red)
generate_plot(
    config_type="pooled_subject", 
    is_region=True, 
    color='#d62728', 
    title="Subject-Level Analysis: Best Region Decoding Accuracy per Condition",
    out_filename="Subject_Level_Best_Region_Per_Condition.png"
)

# Generate Sensor Epoch Plot (Blue)
generate_plot(
    config_type="sensor_epoch", 
    is_region=False, 
    color='#4c72b0', 
    title="Epoch-Level Analysis: Best Sensor Decoding Accuracy per Condition",
    out_filename="Epoch_Level_Best_Sensor_Per_Condition.png"
)

# Generate Pooled Epoch Plot (Red)
generate_plot(
    config_type="pooled_epoch", 
    is_region=True, 
    color='#d62728', 
    title="Epoch-Level Analysis: Best Region Decoding Accuracy per Condition",
    out_filename="Epoch_Level_Best_Region_Per_Condition.png"
)
