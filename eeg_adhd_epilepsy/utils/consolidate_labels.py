#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
# from tqdm import tqdm
import subprocess

def consolidate_labels(embeddings_root, model_name, dry_run=False):
    embeddings_root = Path(embeddings_root)
    print(f"Consolidating labels for {model_name} in {embeddings_root}")
    
    # Discovery: find all subject directories
    subject_dirs = sorted([d for d in embeddings_root.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    
    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        label_file = sub_dir / f"{subject_id}_condition_labels.json"
        
        # Look for any metadata file that has event_labels
        metadata_files = list(sub_dir.glob("*_metadata_*.json"))
        
        found_labels = None
        source_file = None
        
        for meta_path in metadata_files:
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if "event_labels" in meta and meta["event_labels"]:
                    found_labels = meta["event_labels"]
                    source_file = meta_path
                    break
            except Exception as e:
                print(f"Error reading {meta_path}: {e}")
                
        if found_labels:
            if not dry_run:
                with open(label_file, 'w') as f:
                    json.dump({"event_labels": found_labels, "source": str(source_file.name)}, f, indent=2)
            else:
                print(f"[DRY-RUN] Would create {label_file} from {source_file.name}")
        else:
            # print(f"No labels found for {subject_id}")
            pass

def main():
    parser = argparse.ArgumentParser(description="Consolidate event labels per subject")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()
    
    base_root = "/home/mat/scratch/extracted_embeddings"
    for model in ["cbramod", "reve"]:
        model_root = os.path.join(base_root, model)
        if os.path.exists(model_root):
            consolidate_labels(model_root, model, dry_run=args.dry_run)
        else:
            print(f"Skip {model}, root not found: {model_root}")

if __name__ == "__main__":
    main()
