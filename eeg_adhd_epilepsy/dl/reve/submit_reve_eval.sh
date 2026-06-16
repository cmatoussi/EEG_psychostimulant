#!/bin/bash
#SBATCH --job-name=reve_eval
#SBATCH --output=logs/reve_eval_%j.log
#SBATCH --error=logs/reve_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --account=rrg-kjerbi

# Usage: 
# sbatch submit_reve_eval.sh [SIZE] [POOLING] [REPRESENTATION] [CONDITION]
#
# Examples:
# sbatch submit_reve_eval.sh base pool subject_flat
# sbatch submit_reve_eval.sh large no_pool epoch_flat

# 1. Environment Setup
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate

# Default values
SIZE=${1:-base}
POOLING=${2:-pool}
REPRESENTATION=${3:-subject_flat}
CONDITION=${4:-""}

mkdir -p logs

echo "Starting REVE Evaluation"
echo "Size:           $SIZE"
echo "Pooling:        $POOLING"
echo "Representation: $REPRESENTATION"
echo "Condition:      ${CONDITION:-all}"

# Construct command
CMD="python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/reve_evaluate.py \
    --model_size $SIZE \
    --pooling $POOLING \
    --representation $REPRESENTATION"

if [ -n "$CONDITION" ]; then
    CMD="$CMD --conditions $CONDITION"
fi

echo "Running command: $CMD"
$CMD

echo "Job Finished."
