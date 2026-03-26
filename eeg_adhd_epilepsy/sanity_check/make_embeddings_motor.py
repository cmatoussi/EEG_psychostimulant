#!/usr/bin/env python3
import argparse
import gc
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne

# We expect this to run with models in PYTHONPATH or from root
from models.cbramod import CBraMod

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CBraMod embeddings for motor dataset (R01/R02)."
    )
    parser.add_argument("--deriv-proc-root", required=True, help="Root directory containing subject folders (scratch/motor)")
    parser.add_argument("--out-file", required=True, help="Output file path (.h5)")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=10.0,
        help="Patch duration in seconds.",
    )
    parser.add_argument(
        "--points-per-patch",
        type=int,
        default=None,
        help="Number of samples per patch (overrides --segment-duration).",
    )
    parser.add_argument(
        "--stage", 
        default="base", 
        choices=["base", "correct_ica", "correct_dss", "denoise_ar", "denoise_dss_ar"], 
        help="Preprocessing stage to extract"
    )
    parser.add_argument("--max-subjects", type=int, default=None, help="Debug: limit number of subjects")
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Extract embeddings from all transformer encoder layers.",
    )

    
    # args unused but kept/modified for compatibility if needed
    parser.add_argument("--out-csv", help="Legacy argument, use --out-file") 
    parser.add_argument("--ses", help="Legacy argument, ignored for this flat structure")

    return parser.parse_args()


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def find_subject_dirs(root):
    """Finds all subject directories (starting with sub-) in root."""
    subs = []
    if not os.path.exists(root):
        return subs
    
    # Sort to ensure consistent processing order
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith("S") and name[1:].isdigit():
            subs.append((name, full))
    return subs

def find_eeg_file(sub_dir):
    """
    Looks for motor eeg files (R01.edf, R02.edf) in sub_dir
    """
    if not os.path.exists(sub_dir):
        return []
        
    runs = []
    for f in os.listdir(sub_dir):
        if not f.endswith(".edf"):
            continue
        # Check for R01 or R02 in filename
        match = re.search(r"R(\d{2})\.edf$", f)
        if match:
            run_num = match.group(1)
            if run_num in ["01", "02"]:
                runs.append((run_num, os.path.join(sub_dir, f)))
    runs.sort(key=lambda x: x[0])
    return runs

def load_cbramod(weights_path, device="cuda"):
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    LOG.info("Loading CBraMod from %s (device=%s)", weights_path, dev)
    
    # Initialize model
    model = CBraMod().to(dev)
    
    # Load weights
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    
    # Remove projection head if present to get embeddings
    if hasattr(model, "proj_out"):
        model.proj_out = nn.Identity()
        
    model.eval()
    return model, dev


def compute_patches(eeg_data, sfreq, points_per_patch=None):
    """
    Split EEG into 1-second patches across the full recording. Output shape: (C, S, P).
    """
    C, T = eeg_data.shape
    
    # CBraMod strictly expects 1-second patches (e.g., 200 points at 200Hz).
    P = int(points_per_patch) if points_per_patch is not None else int(1.0 * sfreq) 
    
    if P != 200:
        LOG.warning(f"CBraMod is optimized for P=200. You are using P={P}.")

    S = T // P  # Use the full recording; any partial trailing patch is discarded

    if S < 1:
        raise ValueError(f"Recording too short: T={T} < P={P}")

    usable = S * P
    data_block = eeg_data[:, :usable]
    
    # Reshape to (Channels, Segments, Points) -> (C, S, 200)
    patches = data_block.reshape(C, S, P)
    return patches, S, P


# ---------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------
def compute_embedding(
    file_path,
    model,
    device,
    seg_dur=10.0,
    points_per_patch=None,
    all_layers=False,
):
    LOG.info("Processing %s", file_path)
    
    # Load .edf and resample to 200 Hz
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raw.resample(200.0, npad="auto")
    
    # Motor data may have variable channels. We should ensure consistent channel subset if needed, 
    # but CBraMod handles up to 128 usually with padding or projection. 
    # For now, we take all channels from the data as-is
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    
    # Create Patches: (C, S, P)
    # Create Patches: (C, S, P) — full recording, 1-second patches
    patches, S, P = compute_patches(data, sfreq, points_per_patch)

    # Prepare input: (1, C, S, P)
    x = torch.from_numpy(patches).float().unsqueeze(0).to(device)

    layer_outputs = []
    hooks = []

    if all_layers:
        def get_hook():
            def hook(module, input, output):
                layer_outputs.append(output)
            return hook
        
        found_layers = False
        for name, module in model.named_modules():
            if "CrissCrossTransformerBlock" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__ or "TransformerEncoderLayer" in module.__class__.__name__:
                hooks.append(module.register_forward_hook(get_hook()))
                found_layers = True
                
        if not found_layers:
            LOG.warning("Could not automatically find transformer blocks to hook into!")

    try:
        with torch.no_grad():
            out = model(x) 
            
            if all_layers and len(layer_outputs) > 0:
                all_embs = []
                for l_out in layer_outputs:
                    if isinstance(l_out, tuple):
                        l_out = l_out[0]
                    if l_out.dim() == 4:
                        l_out = l_out.permute(0, 2, 1, 3) 
                        l_out = l_out.reshape(1, l_out.size(1), -1)
                        emb = l_out.squeeze(0).cpu().numpy()
                        all_embs.append(emb)
                    else:
                        raise RuntimeError(f"Unexpected CBraMod layer output shape: {tuple(l_out.shape)}")
                
                # Stack all layers into (S, num_layers, C*200)
                emb_flat = np.stack(all_embs, axis=1)
            else:
                # Check output shape
                # Expected: (Batch, Channel, Segment, Time/Feat) -> (1, C, S, 200)
                if out.dim() == 4:
                    # Stack everything into a flat vector per segment
                    # Target shape: (S, C*200)
                    
                    # 1. Permute to (Batch, Segment, Channel, Time) -> (1, S, C, 200)
                    out = out.permute(0, 2, 1, 3) 
                    
                    # 2. Flatten Channel and Time dimensions
                    # (1, S, C*200)
                    out = out.reshape(1, out.size(1), -1)
                    
                    emb_flat = out.squeeze(0).cpu().numpy()
                else:
                    raise RuntimeError(f"Unexpected CBraMod output shape: {tuple(out.shape)}")
    finally:
        for h in hooks:
            h.remove()

    # Identify whether this was R01 (Eyes Open) or R02 (Eyes Closed)
    # We pass label explicitly through to here later, or infer from filename
    label_map = {"R01.edf": "eyes_open", "R02.edf": "eyes_closed"}
    base_name = os.path.basename(file_path)
    
    # Simple search for R01 or R02 in name
    if "R01" in base_name:
        condition_label = "eyes_open"
    elif "R02" in base_name:
        condition_label = "eyes_closed"
    else:
        condition_label = "unknown"
        
    segment_labels = [condition_label] * S

    return emb_flat, patches.shape[0], S, P, segment_labels


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    
    # output path
    out_dir = args.out_file if args.out_file else args.out_csv
    if not out_dir:
        raise ValueError("Output directory path is required (--out-file)")

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Find subjects
    subs = find_subject_dirs(args.deriv_proc_root)
    if not subs:
        LOG.error("No subject directories found in %s", args.deriv_proc_root)
        return

    # Check for existing results to resume (look for .npy files)
    processed_subs = set()
    for f in os.listdir(out_dir):
        if f.endswith("_embeddings.npy"):
            sub_id = f.replace("_embeddings.npy", "")
            processed_subs.add(sub_id)
    
    if processed_subs:
        LOG.info("Resuming: Found %d subjects already processed in %s.", len(processed_subs), out_dir)

    # Filter subjects
    original_count = len(subs)
    subs = [s for s in subs if s[0] not in processed_subs]
    
    if args.max_subjects:
        subs = subs[: args.max_subjects]

    if not subs and len(processed_subs) > 0:
        LOG.info("All subjects already processed! (%d total)", len(processed_subs))
        return

    LOG.info("Found %d pending subjects (out of %d total). Starting processing...", len(subs), original_count)
    
    # Load Model
    model, device = load_cbramod(args.weights, args.device)

    # Processing Loop
    total_segments = 0
    expected_n_feats = None
    
    # Monitoring
    skipped_subs = []
    failed_subs = []
    success_subs = []
    
    for sub_id, sub_dir in subs:
        target_runs = find_eeg_file(sub_dir)
        
        if not target_runs:
            LOG.warning("SKIP [%s]: No R01/R02 runs found in %s", sub_id, sub_dir)
            skipped_subs.append(sub_id)
            continue
            
        # Accumulate runs for this subject so they end up in one file (subject_flat later)
        all_emb = []
        all_labels = []
        total_S_sub = 0
        final_C = -1
        final_P = -1
        
        for run_id, target_file in target_runs:
            try:
                emb_flat, C, S, P, segment_labels = compute_embedding(
                    target_file,
                    model,
                    device,
                    args.segment_duration,
                    args.points_per_patch,
                    getattr(args, "all_layers", False)
                )
                if S > 0:
                    all_emb.append(emb_flat)
                    all_labels.extend(segment_labels)
                    total_S_sub += S
                    final_C = C
                    final_P = P
            except Exception as e:
                import traceback
                LOG.error("ERROR [%s run %s]: Processing failed: %s", sub_id, run_id, e)
                continue
                
        if not all_emb:
            failed_subs.append(sub_id)
            continue
            
        emb_flat_cat = np.concatenate(all_emb, axis=0) # (Total_S, C*200) or similar
        
        C = final_C
        P = final_P
        S = total_S_sub
        segment_labels = all_labels
        emb_flat = emb_flat_cat

        # Validate channel consistency
        current_n_feats = C * 200
        if expected_n_feats is None:
            expected_n_feats = current_n_feats
            LOG.info("First subject [%s]: %d channels, %d features", sub_id, C, current_n_feats)
        elif expected_n_feats != current_n_feats:
            LOG.error("ERROR [%s]: Channel mismatch (Expected %d, Got %d). Skipping.", 
                      sub_id, expected_n_feats, current_n_feats)
            failed_subs.append(sub_id)
            continue
        
        # Save as .npy 
        try:
            import json
            
            # Create subject-specific output directory
            sub_out_dir = os.path.join(out_dir, sub_id)
            os.makedirs(sub_out_dir, exist_ok=True)
            
            desc = "base"
            
            # Transform S001 -> sub0001
            sub_tag = f"sub{int(sub_id[1:]):03d}" if sub_id.startswith("S") else sub_id
            
            suffix = "alllayers" if getattr(args, "all_layers", False) else "lastlayer"
            emb_file = os.path.join(sub_out_dir, f"{sub_tag}_cbramod_{suffix}.npy")
            meta_file = os.path.join(sub_out_dir, f"{sub_tag}_cbramod_{suffix}.json")
            
            # Save embeddings array
            np.save(emb_file, emb_flat)
            
            # Save metadata separately
            metadata = {
                "subject": sub_id,
                "n_channels": int(C),
                "n_segments": int(S),
                "n_features": int(current_n_feats),
                "points_per_patch": int(P),
                "embedding_shape": list(emb_flat.shape),
                "all_layers": getattr(args, "all_layers", False),
                "event_labels": segment_labels
            }
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            total_segments += S
            success_subs.append(sub_id)
            LOG.info("SAVED [%s]: %d segments → %s", sub_id, S, emb_file)
            
        except Exception as e:
            LOG.error("ERROR [%s]: Failed to write .npy: %s", sub_id, e)
            failed_subs.append(sub_id)
        
        # Explicit memory cleanup
        del emb_flat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    LOG.info("All finished. Output saved to %s", out_dir)
    LOG.info("SUMMARY: Successfully processed %d subjects (Total segments: %d)", len(success_subs), total_segments)
    if skipped_subs:
        LOG.info("SUMMARY: Skipped %d subjects: %s", len(skipped_subs), skipped_subs)
    if failed_subs:
        LOG.info("SUMMARY: Failed %d subjects: %s", len(failed_subs), failed_subs)

if __name__ == "__main__":
    main()
