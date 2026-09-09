#!/bin/bash
#SBATCH --job-name=cbra_og
#SBATCH --output=logs/cbra_og_%j.out
#SBATCH --error=logs/cbra_og_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:1

# 1. Environment Setup
module load cuda
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# 2. Create logs dir if needed
mkdir -p logs

echo "Starting CBraMod original-weights evaluation (no fine-tuning)"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

# 3. Run evaluation across all 8 conditions
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/crbamod_predict_og.py \
    --data-root    /home/mat/scratch/preproc/ \
    --label-csv    /home/mat/scratch/epilepsy_label_cleaned.csv \
    --weights-path /home/mat/CBraMod/pretrained_weights/pretrained_weights.pth \
    --out-csv      /home/mat/projects/EEG_psychostimulant/data/results/eval/cbramod_without_finetuning.csv \
    --batch-size   64

echo "Job finished."
