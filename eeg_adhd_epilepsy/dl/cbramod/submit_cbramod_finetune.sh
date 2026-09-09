#!/usr/bin/env bash
#SBATCH --job-name=cbramod_ft
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/logs/cbramod_ft_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/logs/cbramod_ft_%j.err
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-8%8

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Default paths
DATA_ROOT="/home/mat/scratch/preproc/"
LABEL_CSV="/home/mat/scratch/epilepsy_label_cleaned.csv"
SAVE_DIR="/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod"
WEIGHTS_PATH="/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth"

# Full list of conditions
COND_LIST=("EO_baseline" "EC_baseline" "HV_EC" "HV_EO" "PostHV_EO" "PostHV_EC" "PHOTO_EC" "PHOTO_EO" "all")

# Get condition for this array task
COND=${COND_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Starting CBraMod Fine-tuning Job Array Task $SLURM_ARRAY_TASK_ID"
echo "  - Current Condition: $COND"
echo "  - Save Directory: $SAVE_DIR"

echo "===================================================="
echo "RUNNING CBraMod FINE-TUNING FOR CONDITION: $COND"
echo "===================================================="

python3 /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/cbramod_finetune.py \
    --data-root "$DATA_ROOT" \
    --label-csv "$LABEL_CSV" \
    --weights-path "$WEIGHTS_PATH" \
    --save-dir "$SAVE_DIR" \
    --conditions "$COND" \
    --seed 42

echo "Finished processing $COND at $(date)"
