#!/usr/bin/env python3
"""
Visualize ML results based on three configurations (All, per-feature, per-sensor).
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from coco_pipe.viz import plot_topomap, plot_bar
except ImportError:
    print("Warning: unable to import coco_pipe.viz. Using fallbacks if possible.")

def _greekify(text: str) -> str:
    rep = {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "theta": "Theta", "delta": "Delta"}
    for k, v in rep.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text

def make_label_map_keep_sensor(cols):
    """Build compact labels per column, preserving the sensor name."""
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
    }
    out = {}
    for raw in cols:
        s = raw
        if s.startswith("feature-"):
            s = s[len("feature-") :]
        
        sensor = None
        # Handle trailing '.spaces-<SENSOR>' OR '_ch-<SENSOR>'
        m_sensor = re.search(r"(\.spaces-|_ch-)([-A-Za-z0-9_]+)$", s)
        if m_sensor:
            sensor = m_sensor.group(2)
            s = s[: m_sensor.start()]
            
        s = s.replace(".bands-", " ").replace("Epochs", " ")
        m_pair = re.search(r"bands_pairs-\((.+)\)", s)
        if m_pair:
            pair = m_pair.group(1).replace("'", "").replace(" ", "").replace(",", "/")
            pair = _greekify(pair)
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
            feat_label = _greekify(feat_label)
            feat_label = feat_label.replace("MeanEpochs", "")
            feat_label = re.sub(r"[_.-]+$", "", feat_label).strip()
            if corrected: feat_label += " (Corrected)"
            
        feat_label = re.sub(r"\s+", " ", feat_label).strip()
        out[raw] = f"{sensor or ''} — {feat_label}".strip(" —")
    return out

def make_label_map(items):
    out = {}
    full_map = make_label_map_keep_sensor(items)
    for raw, formatted in full_map.items():
        # Remove the leading sensor name 'C3 - ' if it's there
        if " — " in formatted:
            formatted = formatted.split(" — ")[1]
        out[raw] = formatted
    return out

def generate_coords_from_mne(montage="standard_1020", restrict_to=None):
    import mne
    std_montage = mne.channels.make_standard_montage(montage)
    pos = std_montage.get_positions()
    ch_pos = pos.get('ch_pos', {})
    rows = []
    names = list(restrict_to) if restrict_to else list(ch_pos.keys())
    for name in names:
        key = name
        if key not in ch_pos:
            if name.upper() in ch_pos: key = name.upper()
            elif name.capitalize() in ch_pos: key = name.capitalize()
            else: continue
        xyz = ch_pos[key]
        rows.append((name, float(xyz[0]), float(xyz[1])))
    if not rows:
        rows = [(n, float(v[0]), float(v[1])) for n, v in ch_pos.items()]
    return pd.DataFrame(rows, columns=["name", "x", "y"]).set_index("name")

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

    # Coordinates
    coords_df = generate_coords_from_mne()
    sensors_order = coords_df.index.tolist()
    
    # ---------------------------------------------------------
    # 1. Global Lasso (All Features & Sensors)
    # ---------------------------------------------------------
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
            
            # Extract top base features (ignoring sensor) 
            # We look at the actual names and extract base feature strings
            base_features = list(dict.fromkeys([make_label_map([x])[x] for x in s_top.index]))
        else:
            print("No feature importances for Global Lasso.")
            base_features = []
    except Exception as e:
        print(f"Failed step 1: {e}")
        base_features = []


    # ---------------------------------------------------------
    # 2. Per-Feature Decoding
    # ---------------------------------------------------------
    print("--- 2. Generating Per-Feature Visualizations")
    try:
        res_feat_dict = all_results.get('classification_lasso_per_feature', {})
        
        # Gather metrics across all features
        feat_metrics = {}
        for feat_key, models_res in res_feat_dict.items():
            if args.model in models_res:
                scores = models_res[args.model].get('metric_scores', {})
                if args.metric in scores:
                    feat_metrics[feat_key] = float(scores[args.metric]['mean'])
        
        if feat_metrics:
            sf = pd.Series(feat_metrics).sort_values(ascending=False)
            top_feats_keys = sf.head(args.top_n).index.tolist()
            
            # Plot Bar chart of Per-Feature Accuracies
            fig2, ax2 = plot_bar(
                sf,
                orientation="vertical",
                title=f"Per-Feature Decoding {args.metric.capitalize()} (Aggregated by {tm_type})\n({args.model})",
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
            
            # Plot Grid of Topomaps for these Top 12 features
            data_list = []
            grid_labels = []
            for fname in top_feats_keys:
                fi_feat = res_feat_dict[fname][args.model].get('feature_importances', {})
                # fi_feat keys usually are just sensor names like '_ch-C3' or 'C3' depending on pipeline
                sensor_imp = {}
                for k, v in fi_feat.items():
                    # extract sensor
                    sname = k.split("_ch-")[-1] if "_ch-" in k else k
                    sensor_imp[sname] = abs(v.get("weighted_mean", v.get("mean", 0.0)))
                
                vec = np.array([sensor_imp.get(ch, np.nan) for ch in sensors_order])
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
                pos = coords_df[["x", "y"]].values
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

    # ---------------------------------------------------------
    # 3. Per-Sensor Decoding
    # ---------------------------------------------------------
    print("--- 3. Generating Per-Sensor Visualizations")
    try:
        res_sens_dict = all_results.get('classification_lasso_per_sensor', {})
        
        sens_metrics = {}
        for sens_key, models_res in res_sens_dict.items():
            if args.model in models_res:
                scores = models_res[args.model].get('metric_scores', {})
                if args.metric in scores:
                    # Clean sensor name, normally it strips the _ch- if we passed it nicely,
                    # but if sens_key is the slice label, it might just be the sensor name!
                    clean_sens = sens_key.replace("_ch-", "")
                    sens_metrics[clean_sens] = float(scores[args.metric]['mean'])
                    
        if sens_metrics:
            # Expand region keys → individual sensor names (for pooled/region configs)
            REGIONS = {
                "front_left":        ["F7", "F3"],
                "front_midline":     ["Fz"],
                "front_right":       ["F8", "F4"],
                "central_left":      ["T3", "C3"],
                "central_midline":   ["Cz"],
                "central_right":     ["T4", "C4"],
                "posterior_left":    ["T5", "P3", "O1"],
                "posterior_midline": ["Pz"],
                "posterior_right":   ["T6", "P4", "O2"],
            }
            expanded_sens_metrics = {}
            for k, v in sens_metrics.items():
                if k in REGIONS:
                    for s in REGIONS[k]:
                        expanded_sens_metrics[s] = v
                else:
                    expanded_sens_metrics[k] = v

            # A. Plot Single Topomap of Decoding Accuracy
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
            ax3.set_title(f"Per-{sp_type} Decoding {args.metric.capitalize()} (Aggregated by {tm_type})\n({args.model})")
            fig3.savefig(os.path.join(args.out_dir, f"3_per_sensor_accuracy_topomap{suffix}.png"), dpi=300)
            plt.close(fig3)
            
            # B. Top Sensor Feature Importances
            top_sensor = pd.Series(sens_metrics).idxmax()
            print(f"Top Sensor identified as: {top_sensor}")
            # The key in res_sens_dict might be the raw top_sensor name
            real_key = top_sensor if top_sensor in res_sens_dict else None
            # fallback
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

    print(f"All visualizations saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
