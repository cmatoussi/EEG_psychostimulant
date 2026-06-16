#!/usr/bin/env bash
#SBATCH --job-name=reve_ft_res_L
#SBATCH --output=/home/mat/scratch/EEG_results/logs/reve_ft_resub_L_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/reve_ft_resub_L_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-8

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Default paths
DATA_ROOT="/home/mat/scratch/preproc/"
LABEL_CSV="/home/mat/scratch/epilepsy_label_cleaned.csv"
SAVE_DIR="/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/large/balanced"

# Full list of conditions
COND_LIST=("EO_baseline" "EC_baseline" "HV_EC" "HV_EO" "PostHV_EO" "PostHV_EC" "PHOTO_EC" "PHOTO_EO" "ALL") 

# Get condition for this array task
COND=${COND_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Starting REVE Large Fine-tuning Resubmission Array Task $SLURM_ARRAY_TASK_ID"
echo "  - Current Condition: $COND"
echo "  - Save Directory: $SAVE_DIR"

echo "===================================================="
echo "RUNNING FINE-TUNING FOR CONDITION: $COND"
echo "===================================================="

# The script has resume logic, so it will skip folds that are already completed.
python3 /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/fine_tune/reve_finetune_large.py \
    --data-root "$DATA_ROOT" \
    --label-csv "$LABEL_CSV" \
    --save-dir "$SAVE_DIR" \
    --conditions "$COND" \
    --seed 42

echo "Finished processing $COND at $(date)"
