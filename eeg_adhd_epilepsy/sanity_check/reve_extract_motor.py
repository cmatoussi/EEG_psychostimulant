 
import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
import mne
from tqdm import tqdm
from reve_model import REVEFeatureExtractor
from reve_preprocessing import preprocess_signal
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Extract REVE embeddings for epilepsy classification")
    parser.add_argument("--data-root", required=True, help="Root directory containing BIDS-like subject folders")
    parser.add_argument("--output-dir", default="/home/mat/scratch/motor_extracted_embeddings/", help="Directory to save per-subject embeddings")
    parser.add_argument("--output-csv", default=None, help="Optional: Save all embeddings as CSV file")
    parser.add_argument("--model-size", default="base", choices=["base", "large"], help="REVE model size")
    parser.add_argument("--stage", default="baseline", choices=["baseline", "correct_ica", "correct_dss", "denoise_ar", "denoise_dss_ar"], help="Preprocessing stage to extract")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subject", default=None, help="Specific subject ID to process (e.g., '1' or 'sub-0001')")
    parser.add_argument("--no-pool", action="store_true", help="Disable pooling and return all tokens")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N subjects")
    return parser.parse_args()

def find_eeg_file(sub_dir):
    """Find motor EDF files in the subject directory."""
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

def extract_features(args):
    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Find Subjects
    sub_dirs = []
    if os.path.exists(args.data_root):
        for name in sorted(os.listdir(args.data_root)):
            full_path = os.path.join(args.data_root, name)
            # Handle specific subject filter
            if args.subject:
                target_sub = str(args.subject).replace('sub-', '')
                if target_sub not in name:
                    continue
                    
            if os.path.isdir(full_path) and name.startswith("S") and name[1:].isdigit():
                sub_dirs.append((name, full_path))
    
    if not sub_dirs:
        print(f"No valid subject directories found in {args.data_root}")
        return
    
    if args.limit:
        sub_dirs = sub_dirs[:args.limit]
    
    # 3. Initialize Model
    print(f"Loading REVE model ({args.model_size}) on {args.device}...")
    model = REVEFeatureExtractor(model_size=args.model_size)
    model.to(args.device)
    model.eval()
    
    # Tracking for summary
    success_count = 0
    skipped_count = 0
    error_count = 0

    # 4. Process Subjects
    print(f"Processing {len(sub_dirs)} subjects...")
    for sub_id, sub_dir in tqdm(sub_dirs):
        
        target_runs = find_eeg_file(sub_dir)
        if not target_runs:
            print(f"SKIP [{sub_id}]: No R01/R02 runs found")
            skipped_count += 1
            continue
            
        # We will accumulate the separate runs into one embedding matrix per subject (subject_flat)
        all_emb = []
        all_labels = []
        n_segments_total = 0
            
        for run_id, eeg_path in target_runs:
            try:
                # Load EEG (EDF format)
                raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
                
                # REVE preprocessing handles resampling implicitly, but we need target 200 Hz
                if raw.info["sfreq"] != 200.0:
                    raw.resample(200.0, npad="auto")
                    
                # We must filter out channels that REVE cannot map position to, otherwise 
                # the number of feature tokens won't match the number of position embeddings.
                supported_mapping = {name.upper(): name for name in model.pos_bank.position_names}
                
                # Motor data has channels like 'Fc5.', ending with a dot sometimes.
                # Let's map them to exactly the name REVE expects
                rename_dict = {}
                for ch in raw.ch_names:
                    cleaned = ch.strip('.')
                    if cleaned.upper() in supported_mapping:
                        rename_dict[ch] = supported_mapping[cleaned.upper()]
                
                mne.rename_channels(raw.info, rename_dict)
                
                valid_chs = [ch for ch in raw.ch_names if ch in model.pos_bank.position_names]

                if not valid_chs:
                    print(f"SKIP [{sub_id} run {run_id}]: No supported positions found.")
                    continue
                
                raw.pick_channels(valid_chs)
                    
                data = raw.get_data() # (C, T)
                ch_names = raw.ch_names
                
                # Z-score Normalization and Clipping
                data_tensor = preprocess_signal(data)
                
                # 10-second segmentation
                window_size = 2000
                n_samples = data_tensor.shape[1]
                n_segments = n_samples // window_size
                
                if n_segments == 0:
                    continue
                
                data_tensor = data_tensor[:, :n_segments * window_size]
                data_batch = data_tensor.view(data_tensor.shape[0], n_segments, window_size).permute(1, 0, 2)
                
                emb_list = []
                batch_size = 4
                with torch.no_grad():
                    for i in range(0, n_segments, batch_size):
                        batch = data_batch[i : i + batch_size]
                        if args.device != "cpu":
                             batch = batch.to(args.device)
                        
                        do_pool = not getattr(args, "no_pool", False)
                        # Convert to float manually to avoid type issues later?
                        emb_batch = model(batch, channel_names=ch_names, pool=do_pool) 
                        emb_batch_np = emb_batch.cpu().detach().numpy()
                        emb_list.append(emb_batch_np)
                        
                        del batch, emb_batch
                        if args.device != "cpu":
                            torch.cuda.empty_cache()
                
                emb_array_run = np.concatenate(emb_list, axis=0)
                all_emb.append(emb_array_run)
                
                # Match run to condition
                condition_label = "eyes_open" if run_id == "01" else "eyes_closed"
                all_labels.extend([condition_label] * n_segments)
                n_segments_total += n_segments
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"ERROR [{sub_id} run {run_id}]: {e}")
                continue
                
        if not all_emb:
            error_count += 1
            continue
            
        # Combine explicitly
        emb_array = np.concatenate(all_emb, axis=0)
        segment_labels = all_labels
        n_segments = n_segments_total

        # Save embeddings with BIDS naming
        sub_out_dir = os.path.join(args.output_dir, sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)
        
        # Subject tag: S001 format
        sub_tag = sub_id
        
        pooling_str = "without_pooling" if getattr(args, "no_pool", False) else "with_pooling"
        emb_file = os.path.join(sub_out_dir, f"{sub_tag}_reve_{args.model_size}_{pooling_str}.npy")
        meta_file = os.path.join(sub_out_dir, f"{sub_tag}_reve_{args.model_size}_{pooling_str}.json")
        
        np.save(emb_file, emb_array)
        
        # Save metadata
        metadata = {
            "subject": sub_id,
            "n_segments": int(n_segments),
            "embedding_shape": list(emb_array.shape),
            "all_layers": True,
            "event_labels": segment_labels
        }
        if len(emb_array.shape) > 1:
            metadata["n_features"] = int(emb_array.shape[-1])
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"SAVED [{sub_id}]: {n_segments} segments, shape {emb_array.shape} → {emb_file}")
        success_count += 1

        # CLEANUP
        del emb_array, all_emb
        import gc
        gc.collect()
        if args.device != "cpu":
            torch.cuda.empty_cache()

    # 5. Print Summary
    print("\n" + "="*60)
    print("REVE Embedding Extraction Complete")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Successfully processed: {success_count} subjects")
    print(f"Skipped: {skipped_count} subjects")
    print(f"Errors: {error_count} subjects")
    print("="*60)

if __name__ == "__main__":
    args = parse_args()
    extract_features(args)
