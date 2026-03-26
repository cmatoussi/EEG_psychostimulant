#!/usr/bin/env bash
# Submit all embedding evaluation jobs at once.
# Results will be appended to a shared CSV file by each job.

SCRIPT=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/analysis/submit_evaluation.sh

echo "Submitting all evaluation jobs..."

# 1. CBraMod
MODEL=cbramod sbatch "$SCRIPT"

# 2. REVE Large - With Pooling
MODEL=reve MODEL_SIZE=large POOLING=pool sbatch "$SCRIPT"

# 3. REVE Large - No Pooling
MODEL=reve MODEL_SIZE=large POOLING=no_pool sbatch "$SCRIPT"

# 4. REVE Base - With Pooling
MODEL=reve MODEL_SIZE=base POOLING=pool sbatch "$SCRIPT"

# 5. REVE Base - No Pooling
MODEL=reve MODEL_SIZE=base POOLING=no_pool sbatch "$SCRIPT"

echo "All jobs submitted!"
