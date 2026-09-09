#!/bin/bash
#SBATCH --job-name=cbramod_eval
#SBATCH --output=logs/cbramod_eval_%j.log
#SBATCH --error=logs/cbramod_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --account=rrg-kjerbi

# Usage: 
# sbatch submit_cbramod_eval.sh [REPRESENTATION] [ALL_LAYERS] [CONDITION]
#
# Examples:
# sbatch submit_cbramod_eval.sh subject_flat 0   (Last Layer)
# sbatch submit_cbramod_eval.sh subject_flat 1   (All Layers)
# sbatch submit_cbramod_eval.sh epoch_flat 0

# 1. Environment Setup
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Default values
REPRESENTATION=${1:-subject_flat}
ALL_LAYERS=${2:-0} # 0 for false, 1 for true
CONDITION=${3:-""}

mkdir -p logs

echo "Starting CBraMod Evaluation"
echo "Representation: $REPRESENTATION"
echo "All Layers:     $ALL_LAYERS"
echo "Condition:      ${CONDITION:-all}"

# Construct command
CMD="python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/cbramod/cbramod_evaluate.py \
    --representation $REPRESENTATION \
    --stage base"

if [ "$ALL_LAYERS" -eq 1 ]; then
    CMD="$CMD --all_layers"
fi

if [ -n "$CONDITION" ]; then
    CMD="$CMD --conditions $CONDITION"
fi

echo "Running command: $CMD"
$CMD

echo "Job Finished."
