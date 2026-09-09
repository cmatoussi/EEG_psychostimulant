#!/usr/bin/env bash
#SBATCH --job-name=reve_ft_L
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/logs/reve_ft_L_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/logs/reve_ft_L_%j.err
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-44%10
#SBATCH --exclude=fc[10608,10715,11019]

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Default paths
DATA_ROOT="/home/mat/scratch/preproc/"
LABEL_CSV="/home/mat/scratch/epilepsy_label_cleaned.csv"
SAVE_DIR="/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/large/balanced"

# Full list of conditions (9 items)
COND_LIST=("EO_baseline" "EC_baseline" "HV_EC" "HV_EO" "PostHV_EO" "PostHV_EC" "PHOTO_EC" "PHOTO_EO" "ALL") 

# Mapping: Array task ID to Condition and Fold
COND_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % 5))

COND=${COND_LIST[$COND_IDX]}

echo "Starting REVE Large Fine-tuning Job Array Task $SLURM_ARRAY_TASK_ID"
echo "  - Condition: $COND"
echo "  - Fold: $FOLD_IDX"
echo "  - Save Directory: $SAVE_DIR"

echo "===================================================="
echo "RUNNING FINE-TUNING FOR $COND | FOLD $FOLD_IDX"
echo "===================================================="

python3 /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/fine_tune/reve_finetune_large.py \
    --data-root "$DATA_ROOT" \
    --label-csv "$LABEL_CSV" \
    --save-dir "$SAVE_DIR" \
    --conditions "$COND" \
    --fold "$FOLD_IDX" \
    --seed 42

echo "Finished processing $COND at $(date)"
