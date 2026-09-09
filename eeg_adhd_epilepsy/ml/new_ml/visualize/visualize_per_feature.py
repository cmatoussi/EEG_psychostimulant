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
    
    print("--- 2. Generating Per-Feature Visualizations")
    try:
        res_feat_dict = all_results.get('classification_lasso_per_feature', {})
        
        feat_metrics = {}
        for feat_key, models_res in res_feat_dict.items():
            if args.model in models_res:
                scores = models_res[args.model].get('metric_scores', {})
                if args.metric in scores:
                    feat_metrics[feat_key] = float(scores[args.metric]['mean'])
        
        if feat_metrics:
            sf = pd.Series(feat_metrics).sort_values(ascending=False)
            top_feats_keys = sf.head(args.top_n).index.tolist()
            
            fig2, ax2 = plot_bar(
                sf,
                orientation="vertical",
                title=f"Per-Feature Decoding {args.metric.capitalize()}",
                ylabel=f"{args.metric.capitalize()}",
                cmap="viridis",
                label_map=make_label_map(sf.index.tolist()),
                top_n=args.top_n,
                nice_axis_limits=True,
                remove_spines="right top",
                figsize=(10, 6)
            )
            fig2.savefig(os.path.join(args.out_dir, f"2_per_feature_decoding_accuracies{suffix}.png"), dpi=300)
            plt.close(fig2)
            
            data_list = []
            
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
            
            grid_labels = []
            for fname in top_feats_keys:
                fi_feat = res_feat_dict[fname][args.model].get('feature_importances', {})
                
                # Expand regions to sensors if pooling='region'
                expanded_sensor_imp = {}
                for k, v in fi_feat.items():
                    if "_chgrp-" in k:
                        sname = k.split("_chgrp-")[-1]
                    elif "_ch-" in k:
                        sname = k.split("_ch-")[-1]
                    else:
                        sname = k
                    val = abs(v.get("weighted_mean", v.get("mean", 0.0)))
                    if sname in REGIONS:
                        for s in REGIONS[sname]:
                            expanded_sensor_imp[s] = val
                    else:
                        expanded_sensor_imp[sname] = val
                
                vec = np.array([expanded_sensor_imp.get(ch, np.nan) for ch in sensors_order])
                data_list.append(vec)
                grid_labels.append(make_label_map([fname])[fname])
            if data_list:
                all_vals = np.concatenate([np.nan_to_num(v, nan=0.0) for v in data_list])
                vmax = float(np.nanmax(all_vals)) if all_vals.size else 1.0
                vmin = 0.0
                
                n_rows, n_cols = 3, 4
                import mne
                fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
                axes = np.atleast_2d(axes)
                im_last = None
                for ax, data, title in zip(axes.ravel(), data_list, grid_labels):
                    valid = np.isfinite(data)
                    if valid.sum() > 3:
                        valid_sensors = [sensors_order[i] for i in range(len(sensors_order)) if valid[i]]
                        info = mne.create_info(ch_names=valid_sensors, sfreq=100, ch_types="eeg")
                        info.set_montage("standard_1020", match_case=False)
                        im, cn = mne.viz.plot_topomap(
                            data[valid], info, axes=ax, show=False, contours=0,
                            cmap="magma", vlim=(vmin, vmax), extrapolate="head"
                        )
                        im_last = im
                    ax.set_title(title, fontsize=16)
                
                fig_grid.tight_layout(rect=[0, 0.03, 0.9, 0.95])
                if im_last is not None:
                    cax = fig_grid.add_axes([0.92, 0.15, 0.02, 0.7])
                    cbar = fig_grid.colorbar(im_last, cax=cax)
                    cbar.set_label("Importance Coefficient", rotation=90, fontsize=18)
                fig_grid.suptitle(f"Top {len(top_feats_keys)} Features - Sensor Importance Maps", fontsize=24)
                fig_grid.savefig(os.path.join(args.out_dir, f"2_per_feature_topomaps_grid{suffix}.png"), dpi=300)
                plt.close(fig_grid)
    except Exception as e:
         print(f"Failed step 2: {e}")

    print(f"Per-Feature visualizations saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
