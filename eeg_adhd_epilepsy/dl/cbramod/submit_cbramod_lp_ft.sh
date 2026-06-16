#!/bin/bash
#SBATCH --job-name=cbra_lpft
#SBATCH --output=logs/lpft_%A_%a.out
#SBATCH --error=logs/lpft_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:1
#SBATCH --array=0-44%10

# 1. Environment Setup
module load cuda
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# 2. Configuration for Job Array
COND_LIST=("EO_baseline" "EC_baseline" "HV_EC" "HV_EO" "PostHV_EO" "PostHV_EC" "PHOTO_EC" "PHOTO_EO" "ALL") 

COND_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % 5))
CONDITION=${COND_LIST[$COND_IDX]}

echo "Starting CBraMod LP+FT Fine-tuning Array Task $SLURM_ARRAY_TASK_ID"
echo "Condition: $CONDITION"
echo "Fold:      $FOLD_IDX"

# 3. Create logs dir inside the job if it doesn't exist
mkdir -p logs

# 4. Construct command
CMD="python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/cbramod_fine_lp_ft.py \
    --conditions $CONDITION \
    --fold $FOLD_IDX"

echo "Running command: $CMD"
$CMD

echo "Job finished."
