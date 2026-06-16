import os
import pandas as pd
import re

RESULTS_DIR = "/home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/multi_condition"
CONDITIONS = [
    "EC_baseline", "EO_baseline",
    "HV_EC", "HV_EO",
    "PostHV_EO", "PostHV_EC",
    "PHOTO_EC", "PHOTO_EO",
]

def clean_feature_name(raw: str) -> str:
    """Cleans up the raw feature names to make them readable for the table."""
    s = raw.replace("feature-", "")
    s = re.sub(r"(\.spaces-|_ch-)([-A-Za-z0-9]+)$", "", s)
    s = s.replace(".bands-", " ").replace("Epochs", "")
    
    # Convert greek letters
    rep = {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "theta": "Theta", "delta": "Delta"}
    for k, v in rep.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)
        
    abbrev = {
        "BandRatiosFromAverageFooof": "(Corrected)",
        "BandRatiosFromAverageSpectrum": "",
        "RelativeBandPowerFromAverageFooof": "(Corrected)",
        "RelativeBandPowerFromAverageSpectrum": "",
        "higuchiFd": "Higuchi FD",
        "katzFd": "Katz FD",
        "petrosianFd": "Petrosian FD",
        "hjorthComplexity": "Hjorth Complexity",
        "hjorthMobility": "Hjorth Mobility",
        "numZerocross": "ZeroCross",
        "svdEntropy": "SVD Entropy",
        "spectralEntropy": "Spectral Entropy",
        "sampleEntropy": "Sample Entropy",
        "permEntropy": "Perm Entropy",
        "entropyMultiscale": "MSE",
        "fooofExponent": "1/f slope",
        "foofOffset": "1/f Offset",
        "lzivComplexity": "LZ Complexity",
        "Mean": "",
    }
    
    m_pair = re.search(r"bands_pairs-\((.+)\)", s)
    if m_pair:
        pair = m_pair.group(1).replace("'", "").replace(" ", "").replace(",", "/")
        head = s[: m_pair.start()].rstrip(".")
        corrected = False
        for k, v in abbrev.items():
            if k in head:
                head = head.replace(k, v)
        if "(Corrected)" in head or "(corrected)" in head:
            corrected = True
            head = head.replace("(Corrected)", "").replace("(corrected)", "").strip()
        feat_label = f"{head} {pair}".strip()
        if corrected: feat_label += " (Corrected)"
    else:
        feat_label = s
        corrected = False
        for k, v in abbrev.items():
            if k in feat_label:
                feat_label = feat_label.replace(k, v)
        if "(Corrected)" in feat_label or "(corrected)" in feat_label:
            corrected = True
            feat_label = feat_label.replace("(Corrected)", "").replace("(corrected)", "").strip()
        feat_label = re.sub(r"[_.-]+$", "", feat_label).strip()
        if corrected: feat_label += " (Corrected)"
        
    return re.sub(r"\s+", " ", feat_label).strip()


def main():
    top_n_per_cond = 15
    model_name = "Logistic Regression"
    metric = "balanced_accuracy"
    
    all_features_data = []
    
    for cond in CONDITIONS:
        pkl_path = os.path.join(RESULTS_DIR, f"epilepsy_lasso_{cond}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Warning: Results for {cond} not found at {pkl_path}. Make sure the pipeline finishes first.")
            continue
            
        data = pd.read_pickle(pkl_path)
        
        # Extract the Per-Feature Decoding accuracy
        res_feat = data.get('classification_lasso_per_feature', {})
        
        feat_metrics = {}
        for feat_key, models_res in res_feat.items():
            if model_name in models_res:
                scores = models_res[model_name].get('metric_scores', {})
                if metric in scores:
                    feat_metrics[feat_key] = float(scores[metric]['mean'])
                    
        if not feat_metrics:
            print(f"No per-feature results for {cond}.")
            continue
            
        # Sort features by balanced accuracy in this condition
        sf = pd.Series(feat_metrics).sort_values(ascending=False)
        top_feats = sf.head(top_n_per_cond)
        
        for raw_feat, acc in top_feats.items():
            clean_name = clean_feature_name(raw_feat)
            all_features_data.append({
                "Condition": cond,
                "RawFeature": raw_feat,
                "CleanFeature": clean_name,
                "Accuracy": acc
            })
            
    if not all_features_data:
        print("No data collected across conditions.")
        return
        
    df = pd.DataFrame(all_features_data)
    
    # Aggregate frequency and mean accuracy across conditions
    summary = df.groupby("CleanFeature").agg(
        Frequency=("Condition", "count"),
        AvgAccuracy=("Accuracy", "mean"),
        Conditions=("Condition", lambda x: ", ".join(x))
    ).reset_index()
    
    # Sort by Frequency descending, then by AvgAccuracy descending
    summary = summary.sort_values(by=["Frequency", "AvgAccuracy"], ascending=[False, False])
    
    # Extract the top 10
    top_10 = summary.head(10)
    
    print("\n" + "="*100)
    print("TOP 10 MOST COMMON HANDCRAFTED FEATURES ACROSS CONDITIONS")
    print("="*100)
    print(top_10.to_string(index=False))
    print("="*100)
    
    # Save table to CSV
    out_csv = os.path.join(RESULTS_DIR, "top_10_common_features.csv")
    top_10.to_csv(out_csv, index=False)
    print(f"\nSaved summary table to: {out_csv}")

if __name__ == "__main__":
    main()
