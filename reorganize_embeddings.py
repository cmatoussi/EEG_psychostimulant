#!/usr/bin/env python3
"""
Reorganize CBRAMOD embeddings to BIDS-compliant structure.
Moves files from /home/mat/scratch/cbrmod_embeddings/ to
/home/mat/scratch/embeddings/cbramod/baseline/sub-XXXX/
with proper BIDS naming convention.
"""
import os
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = "/home/mat/scratch/cbrmod_embeddings"
TARGET_BASE = "/home/mat/scratch/embeddings/cbramod/baseline"
DRY_RUN = False  # Set to False to actually move files

def reorganize_embeddings():
    """Reorganize embedding files into BIDS structure."""
    
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return
    
    # Get all .npy files
    npy_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith("_embeddings.npy")])
    
    if not npy_files:
        print(f"No embedding files found in {SOURCE_DIR}")
        return
    
    print(f"Found {len(npy_files)} subjects to process")
    print(f"Mode: {'DRY RUN (no files will be moved)' if DRY_RUN else 'LIVE (files will be moved)'}")
    print("-" * 80)
    
    moved_files = []
    
    for npy_file in npy_files:
        # Extract subject ID (e.g., "sub-0001" from "sub-0001_embeddings.npy")
        subject_id = npy_file.replace("_embeddings.npy", "")
        
        # Define source files
        source_embedding = os.path.join(SOURCE_DIR, f"{subject_id}_embeddings.npy")
        source_metadata = os.path.join(SOURCE_DIR, f"{subject_id}_metadata.json")
        
        # Check if metadata exists
        if not os.path.exists(source_metadata):
            print(f"WARNING: Missing metadata for {subject_id}, skipping...")
            continue
        
        # Define target directory and files
        target_dir = os.path.join(TARGET_BASE, subject_id)
        target_embedding = os.path.join(target_dir, f"{subject_id}_desc-base_embed.npy")
        target_metadata = os.path.join(target_dir, f"{subject_id}_desc-base_metadata.json")
        
        # Create target directory
        if not DRY_RUN:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        else:
            print(f"[DRY RUN] Would create: {target_dir}")
        
        # Move files
        if not DRY_RUN:
            shutil.move(source_embedding, target_embedding)
            shutil.move(source_metadata, target_metadata)
            moved_files.append(subject_id)
            print(f"✓ Moved {subject_id}")
        else:
            print(f"[DRY RUN] Would move:")
            print(f"  {source_embedding} -> {target_embedding}")
            print(f"  {source_metadata} -> {target_metadata}")
    
    print("-" * 80)
    if DRY_RUN:
        print(f"DRY RUN complete. {len(npy_files)} subjects would be processed.")
        print(f"\nTo execute the move, set DRY_RUN = False in the script.")
    else:
        print(f"✓ Successfully reorganized {len(moved_files)} subjects")
        print(f"Files moved to: {TARGET_BASE}")
        
        # Check if source directory is now empty
        remaining = os.listdir(SOURCE_DIR)
        if remaining:
            print(f"\nWARNING: {len(remaining)} files remain in source directory:")
            for f in remaining[:10]:  # Show first 10
                print(f"  - {f}")
        else:
            print(f"\n✓ Source directory is now empty: {SOURCE_DIR}")

if __name__ == "__main__":
    reorganize_embeddings()
