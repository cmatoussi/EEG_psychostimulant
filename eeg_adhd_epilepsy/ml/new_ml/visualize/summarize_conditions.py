import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from viz_utils import make_label_map_keep_sensor

conditions = {
    "Sensor Epoch": "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_epoch.pkl",
    "Sensor Subject": "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_subject.pkl",
    "Pooled Epoch": "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_epoch.pkl",
    "Pooled Subject": "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_subject.pkl"
}

out_dir = "/home/mat/projects/EEG_psychostimulant/data/results/hand_crafted/summary"
os.makedirs(out_dir, exist_ok=True)

model = "Logistic Regression"
metric = "balanced_accuracy"

summary_data = []

# 1. Process each condition
for cond_name, pkl_path in conditions.items():
    if not os.path.exists(pkl_path):
        print(f"Missing {pkl_path}")
        continue
    all_results = pd.read_pickle(pkl_path)
    
    # Global Lasso
    res_all = all_results.get('classification_lasso_all', {}).get(model, {})
    fi = res_all.get('feature_importances', {})
    
    top_5_features = []
    if fi:
        s = pd.Series({k: v.get("weighted_mean", v.get("mean", 0.0)) for k, v in fi.items()})
        s_top = s.abs().sort_values(ascending=False).head(5)
        # map names
        label_map = make_label_map_keep_sensor(s_top.index.tolist())
        top_5_features = [(label_map[k], s[k]) for k in s_top.index]
        
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
        
        # Plot bar chart for this condition (Top 10 sensors/regions)
        plt.figure(figsize=(8, 6))
        s_plot = s_sens.head(10)
        s_plot.sort_values(ascending=True).plot(kind='barh', color='skyblue')
        sp_type = "Region" if "Pooled" in cond_name else "Sensor"
        agg_type = "Epoch" if "Epoch" in cond_name else "Subject"
        plt.title(f"Top {sp_type}s ({metric})\n{sp_type}-Based, Aggregated by {agg_type}")
        plt.xlabel(metric.capitalize())
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cond_name.replace(' ', '_')}_{sp_type}_bar.png"), dpi=300)
        plt.close()
        
    summary_data.append({
        "Condition": cond_name,
        "Top_5_Features": top_5_features,
        "Top_5_Sensors": top_5_sensors,
        "Best_Spatial_Name": top_5_sensors[0][0] if top_5_sensors else None,
        "Best_Spatial_Acc": top_5_sensors[0][1] if top_5_sensors else None
    })

# 2. Overall Bar Chart
names = [d["Condition"] for d in summary_data]
formatted_names = []
for n in names:
    sp = "Region" if "Pooled" in n else "Sensor"
    ag = "Epoch" if "Epoch" in n else "Subject"
    formatted_names.append(f"{sp}-Based\n(Aggregated by {ag})")

accs = [d["Best_Spatial_Acc"] for d in summary_data]
labels = [d["Best_Spatial_Name"] for d in summary_data]

plt.figure(figsize=(10, 6))
bars = plt.bar(formatted_names, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
for bar, label, acc in zip(bars, labels, accs):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{label}\n{acc:.3f}", ha='center', va='bottom', fontweight='bold')
plt.ylim(0, max(accs) + 0.1)
plt.ylabel(metric.capitalize())
plt.title("Best Spatial Decoding Accuracy per Condition\n(Sensor vs Region, Epoch vs Subject Aggregation)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Overall_Best_Spatial_Acc.png"), dpi=300)
plt.close()

# 3. Generate Markdown Table
md_lines = []
md_lines.append("# Condition Summaries\n")

for d in summary_data:
    cond = d["Condition"]
    md_lines.append(f"## {cond}")
    
    # Top 5 Features Table
    md_lines.append("### Top 5 Features")
    md_lines.append("| Rank | Feature | Importance (Coefficient) |")
    md_lines.append("|---|---|---|")
    for i, (feat, imp) in enumerate(d["Top_5_Features"]):
        md_lines.append(f"| {i+1} | {feat} | {imp:.4f} |")
    md_lines.append("\n")
    
    # Top 5 Sensors Table
    sp_type = "Region" if "Pooled" in cond else "Sensor"
    md_lines.append(f"### Top 5 {sp_type}s")
    md_lines.append(f"| Rank | {sp_type} | Decoding Accuracy |")
    md_lines.append("|---|---|---|")
    for i, (sens, acc) in enumerate(d["Top_5_Sensors"]):
        md_lines.append(f"| {i+1} | {sens} | {acc:.4f} |")
    md_lines.append("\n")
    
with open(os.path.join(out_dir, "summary_tables.md"), "w") as f:
    f.write("\n".join(md_lines))

print(f"Summary generated at: {out_dir}")
