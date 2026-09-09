#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from coco_pipe.viz import plot_bar
except ImportError:
    print("Warning: unable to import coco_pipe.viz. Using fallbacks if possible.")

from viz_utils import make_label_map, generate_coords_from_mne

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to pkl file")
    parser.add_argument("--out-dir", default="/home/mat/projects/EEG_psychostimulant/data/results/hand_crafted")
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--model", default="Logistic Regression")
    parser.add_argument("--config", help="Path to config file to infer epoch/subject and sensor/region", default=None)
    args = parser.parse_args()

    suffix = ""
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
            pooling = cfg.get("pooling")
            representation = cfg.get("representation")
            
            # Detect spatial
            if pooling:
                spatial = "_region" if pooling == "region" else "_sensor"
            else:
                dp = cfg.get("data_path", "")
                if "pooled" in dp or "region" in dp:
                    spatial = "_region"
                elif "sensor" in dp:
                    spatial = "_sensor"
                else:
                    spatial = ""
                    
            # Detect temporal
            if representation:
                temporal = f"_{representation}"
            else:
                dp = cfg.get("data_path", "")
                if "epoch" in dp:
                    temporal = "_epoch"
                elif "subject" in dp:
                    temporal = "_subject"
                else:
                    temporal = ""
                
            suffix = f"{temporal}{spatial}"

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = pd.read_pickle(args.results)

    coords_df = generate_coords_from_mne()
    sensors_order = coords_df.index.tolist()
    
    print("--- 3. Generating Per-Sensor Visualizations")
    try:
        res_sens_dict = all_results.get('classification_lasso_per_sensor', {})
        
        sens_metrics = {}
        for sens_key, models_res in res_sens_dict.items():
            if args.model in models_res:
                scores = models_res[args.model].get('metric_scores', {})
                if args.metric in scores:
                    clean_sens = sens_key.replace("_ch-", "")
                    sens_metrics[clean_sens] = float(scores[args.metric]['mean'])
                    
        if sens_metrics:
            REGIONS = {
                "front_left": ["F7", "F3"],
                "front_midline": ["Fz"],
                "front_right": ["F8", "F4"],
                "central_left": ["T3", "C3"],
                "central_midline": ["Cz"],
                "central_right": ["T4", "C4"],
                "posterior_left": ["T5", "P3", "O1"],
                "posterior_midline": ["Pz"],
                "posterior_right": ["T6", "P4", "O2"]
            }
            
            # Expand regions to sensors if pooling='region'
            expanded_sens_metrics = {}
            for k, v in sens_metrics.items():
                if k in REGIONS:
                    for s in REGIONS[k]:
                        expanded_sens_metrics[s] = v
                else:
                    expanded_sens_metrics[k] = v
            acc_vec = np.array([expanded_sens_metrics.get(ch, np.nan) for ch in sensors_order])
            
            fig3, ax3 = plt.subplots(figsize=(8,8))
            import mne
            valid = np.isfinite(acc_vec)
            if valid.sum() > 3:
                valid_sensors = [sensors_order[i] for i in range(len(sensors_order)) if valid[i]]
                valid_acc = acc_vec[valid]
                max_idx = np.nanargmax(valid_acc)
                max_val = valid_acc[max_idx]
                
                mask = np.zeros(len(valid_acc), dtype=bool)
                mask[max_idx] = True
                
                mask_params = dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=2, markersize=15)
                
                info = mne.create_info(ch_names=valid_sensors, sfreq=100, ch_types="eeg")
                info.set_montage("standard_1020", match_case=False)
                try:
                    im, _ = mne.viz.plot_topomap(
                        valid_acc, info, axes=ax3, show=False, contours=0,
                        cmap="inferno", extrapolate="head", mask=mask, mask_params=mask_params,
                        vlim=(0.5, None)
                    )
                except TypeError:
                    im, _ = mne.viz.plot_topomap(
                        valid_acc, info, axes=ax3, show=False, contours=0,
                        cmap="inferno", extrapolate="head", mask=mask, mask_params=mask_params,
                        vmin=0.5, vmax=None
                    )
                try:
                    from mne.channels.layout import _find_topomap_coords
                    pos = _find_topomap_coords(info, picks="all")
                except ImportError:
                    pos = mne.viz.topomap._find_topomap_coords(info, picks="all")
                x, y = pos[max_idx]
                ax3.annotate(f"{max_val:.3f}", xy=(x, y), xytext=(0, 15), textcoords='offset points', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
                cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                cbar.set_label(args.metric.capitalize())
            ax3.set_title(f"Per-Sensor Decoding {args.metric.capitalize()}")
            fig3.savefig(os.path.join(args.out_dir, f"3_per_sensor_accuracy_topomap{suffix}.png"), dpi=300)
            plt.close(fig3)
            
            top_sensor = pd.Series(sens_metrics).idxmax()
            print(f"Top Sensor identified as: {top_sensor}")
            real_key = top_sensor if top_sensor in res_sens_dict else None
            if not real_key:
                for k in res_sens_dict.keys():
                    if top_sensor in k:
                        real_key = k
                        break
                        
            if real_key:
                sens_fi = res_sens_dict[real_key][args.model].get('feature_importances', {})
                if sens_fi:
                    sf_sens = pd.Series({k: v.get("weighted_mean", v.get("mean", 0.0)) for k, v in sens_fi.items()})
                    sf_sens = sf_sens.loc[sf_sens.abs().sort_values(ascending=False).head(args.top_n).index]
                    
                    fig4, ax4 = plot_bar(
                        sf_sens,
                        labels=sf_sens.index.tolist(),
                        label_map=make_label_map(sf_sens.index.tolist()),
                        top_n=args.top_n,
                        ascending=False,
                        orientation="horizontal",
                        title=f"Top Features for Sensor {top_sensor}\n({args.model})",
                        xlabel="Coefficient magnitude",
                        cmap="magma",
                        figsize=(8, 6),
                        abs_values=True,
                        remove_spines="right top"
                    )
                    fig4.savefig(os.path.join(args.out_dir, f"3_top12_features_best_sensor_{top_sensor}{suffix}.png"), dpi=300, bbox_inches="tight")
                    plt.close(fig4)
                
    except Exception as e:
        print(f"Failed step 3: {e}")

    print(f"Per-Sensor visualizations saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
