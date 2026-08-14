import os
import pandas as pd
from viz_utils import make_label_map_keep_sensor

phys_conditions = ['EC_baseline', 'EO_baseline', 'HV_EC', 'PHOTO_EC', 'PostHV_EC', 'PostHV_EO', 'HV_EO', 'PHOTO_EO']
config_types = ['sensor_epoch', 'sensor_subject', 'pooled_epoch', 'pooled_subject']
in_dir = "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/multi_condition"
model = "Logistic Regression"

for ctype in config_types:
    print(f"### {ctype}")
    print("| Condition | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |")
    print("|---|---|---|---|---|---|")
    for cond in phys_conditions:
        pkl_path = os.path.join(in_dir, f"epilepsy_feature_importance_lasso_{ctype}_{cond}.pkl")
        if not os.path.exists(pkl_path):
            continue
            
        all_results = pd.read_pickle(pkl_path)
        res_all = all_results.get('classification_lasso_all', {}).get(model, {})
        fi = res_all.get('feature_importances', {})
        
        if fi:
            s = pd.Series({k: v.get("weighted_mean", v.get("mean", 0.0)) for k, v in fi.items()})
            s_top = s.abs().sort_values(ascending=False).head(5)
            try:
                label_map = make_label_map_keep_sensor(s_top.index.tolist())
                top_features = [f"{label_map.get(k, k)}: {s[k]:.3f}" for k in s_top.index]
            except Exception as e:
                top_features = [f"{k}: {s[k]:.3f}" for k in s_top.index]
            
            # Ensure there are 5 elements
            while len(top_features) < 5:
                top_features.append("")
                
            print(f"| {cond} | {top_features[0]} | {top_features[1]} | {top_features[2]} | {top_features[3]} | {top_features[4]} |")
    print()
