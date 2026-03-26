#!/bin/bash
#SBATCH --job-name=analyze_cbramod
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/analyze_cbramod_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/analyze_cbramod_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

# 1. Environment Setup
# Adjust the source path to your virtual environment if different
source /home/mat/projects/EEG_psychostimulant/dim_red/bin/activate

# 2. Configuration
MODEL="cbramod"
REPRESENTATION=${REPRESENTATION:-"epoch_flat"}
ALL_LAYERS=${ALL_LAYERS:-0}
TARGET_COL=${TARGET_COL:-"EO_EC"}
REDUCERS=${REDUCERS:-"pca"}
CONDITIONS=${CONDITIONS:-"EC_baseline EO_baseline"}
DESC="base"

# Paths
EMBEDDINGS_DIR="/home/mat/scratch/extracted_embeddings/cbramod"
METADATA_CSV="/home/mat/scratch/EEG_Psychostimulants_PatientList_08-2025.csv"
PROJECT_ROOT="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy"

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Running CBraMod analysis for:"
echo "  - Representation: $REPRESENTATION"
echo "  - All Layers: $ALL_LAYERS"
echo "  - Target: $TARGET_COL"
echo "  - Reducers: $REDUCERS"
echo "  - Conditions: $CONDITIONS"

# 3. Build command
cmd=(python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/analysis/analyze_deep_embeddings.py \
    --representation "$REPRESENTATION" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --model "$MODEL" \
    --desc "$DESC" \
    --metadata "$METADATA_CSV" \
    --target_col "$TARGET_COL" \
    --reducers $REDUCERS \
    --conditions $CONDITIONS \
    --all_layers "$ALL_LAYERS" \
    --max_samples 100000)

# 4. Execute
"${cmd[@]}"

echo "Job $SLURM_JOB_ID finished at $(date)"
