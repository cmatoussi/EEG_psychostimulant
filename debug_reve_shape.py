
import numpy as np
import torch
import os
import sys

# Add path to finding reve_model
sys.path.append("/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/dl/reve")

try:
    from reve_model import REVEFeatureExtractor
except ImportError:
    # Try local import if running from same dir, or adjust path
    pass

def check_npy_shape():
    npy_path = "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy_psychostimulant/data/results/dl/test/sub-0001_desc-base_embed.npy"
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        print(f"NPY File Shape: {data.shape}")
        print(f"Total elements: {data.size}")
        if data.size > 512:
             print(f"Ratio to 512: {data.size / 512}")
    else:
        print("NPY file not found.")

def check_model_forward():
    print("Initializing model...")
    try:
        model = REVEFeatureExtractor(model_size='base')
        model.eval()
        
        # Create dummy input: (1, 19, 2000) - 19 channels, 10 seconds at 200Hz
        x = torch.randn(1, 19, 2000)
        ch_names = [f"Ch{i}" for i in range(19)]
        
        print("Running forward pass...")
        with torch.no_grad():
            out = model(x, channel_names=ch_names)
            
        print(f"Model Output Shape: {out.shape}")
    except Exception as e:
        print(f"Model run failed: {e}")

if __name__ == "__main__":
    check_npy_shape()
    print("-" * 20)
    check_model_forward()
