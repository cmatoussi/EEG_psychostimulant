import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys

sys.path.append("/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/ml/new_ml/visualize")
from viz_utils import make_label_map

phys_conditions = ['EC_baseline', 'EO_baseline', 'HV_EC', 'PHOTO_EC', 'PostHV_EC', 'PostHV_EO', 'HV_EO', 'PHOTO_EO']
config_types = ['sensor_epoch', 'sensor_subject', 'pooled_epoch', 'pooled_subject']
in_dir = "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/multi_condition"
out_dir = "/home/mat/projects/EEG_psychostimulant/data/results/hand_crafted/overall_summary"
model = "Logistic Regression"

# Dictionary to store accumulated scores for each base feature
# base_feature_name -> score
feature_scores = {}

print("Aggregating feature importance and accuracy...")

for ctype in config_types:
    for cond in phys_conditions:
        pkl_path = os.path.join(in_dir, f"epilepsy_feature_importance_lasso_{ctype}_{cond}.pkl")
        if not os.path.exists(pkl_path):
            continue
            
        all_results = pd.read_pickle(pkl_path)
        
        # Get global lasso feature importances
        res_lasso = all_results.get('classification_lasso_all', {}).get(model, {})
        fi = res_lasso.get('feature_importances', {})
        
        # Get per-feature accuracies
        res_per_feature = all_results.get('classification_lasso_per_feature', {}).get(model, {})
        acc_dict = res_per_feature.get('scores', {})
        
        if not fi:
            continue
            
        # Parse feature importances into a Series
        s_fi = pd.Series({k: v.get("weighted_mean", v.get("mean", 0.0)) for k, v in fi.items()})
        
        # Parse per-feature accuracies into a Series (using balanced_accuracy if available, else first metric)
        if acc_dict:
            first_metric = list(acc_dict.keys())[0] # Usually balanced_accuracy
            s_acc = pd.Series({k: v.get("mean", 0.5) for k, v in acc_dict[first_metric].items()})
        else:
            s_acc = pd.Series(0.5, index=s_fi.index)

        # Get base names using make_label_map (which removes sensor/region info)
        label_map = make_label_map(s_fi.index.tolist())
        
        for feature_key in s_fi.index:
            base_name = label_map.get(feature_key, feature_key)
            importance = abs(s_fi[feature_key])
            accuracy = s_acc.get(feature_key, 0.5)
            
            # Simple scoring: Importance * (Accuracy^2) so that high accuracy boosts importance
            score_contribution = importance * (accuracy ** 2)
            
            if base_name not in feature_scores:
                feature_scores[base_name] = 0.0
            
            feature_scores[base_name] += score_contribution

# Filter out features with 0 score
feature_scores = {k: v for k, v in feature_scores.items() if v > 0}

print(f"Generated scores for {len(feature_scores)} unique base features.")

# Generate Word Cloud
print("Generating WordCloud...")
wordcloud = WordCloud(
    width=1600, 
    height=800, 
    background_color='white',
    colormap='viridis',
    max_words=100,
    contour_width=0,
    collocations=False # Avoids grouping 'band' and 'abs' into 'band abs'
).generate_from_frequencies(feature_scores)

# Plot and save
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("EEG Biomarkers (Bag of Words)\nWeighted by Feature Importance across All Conditions & Configurations", fontsize=24, pad=20)

os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "feature_bag_of_words.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"WordCloud saved to {out_path}")
