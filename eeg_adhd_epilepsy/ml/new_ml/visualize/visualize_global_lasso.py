#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

try:
    from coco_pipe.viz import plot_bar
except ImportError:
    print("Warning: unable to import coco_pipe.viz. Using fallbacks if possible.")

from viz_utils import make_label_map_keep_sensor, make_label_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to pkl file")
    parser.add_argument("--out-dir", default="/home/mat/projects/EEG_psychostimulant/data/results/hand_crafted")
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

    print("--- 1. Generating Global Lasso Bar Chart")
    try:
        res_all = all_results['classification_lasso_all'][args.model]
        fi = res_all.get('feature_importances', {})
        if fi:
            s = pd.Series({k: v.get("weighted_mean", v.get("mean", 0.0)) for k, v in fi.items()})
            s_top = s.loc[s.abs().sort_values(ascending=False).head(15).index]
            label_map = make_label_map_keep_sensor(s_top.index.tolist())
            
            sp_type = "Region" if "region" in spatial else "Sensor"
            tm_type = "Subject" if "subject" in temporal else "Epoch"
            fig, ax = plot_bar(
                s_top,
                labels=s_top.index.tolist(),
                label_map=label_map,
                top_n=15,
                ascending=False,
                orientation="horizontal",
                title=f"Top 15 Feature + {sp_type} Combinations (Aggregated by {tm_type})\n({args.model})",
                xlabel="Coefficient magnitude",
                cmap="magma",
                figsize=(7, 8),
                abs_values=True,
                remove_spines="right top",
                remove_ticks="both"
            )
            fig.savefig(os.path.join(args.out_dir, f"1_global_lasso_top15{suffix}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            
        else:
            print("No feature importances for Global Lasso.")
    except Exception as e:
        print(f"Failed step 1: {e}")

    print(f"Global Lasso visualization saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
