#!/usr/bin/env bash
#SBATCH --job-name=eval_embed
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/analysis/logs/eval_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/analysis/logs/eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

# Activate the environment
source /home/mat/projects/EEG_psychostimulant/dim_red/bin/activate

# Configuration
MODEL=${MODEL:-"reve"}
MODEL_SIZE=${MODEL_SIZE:-"large"}
POOLING=${POOLING:-"pool"}
TARGET_COL=${TARGET_COL:-"has_epilepsy"}
REPRESENTATION=${REPRESENTATION:-"subject_flat"}
CLASSIFIER=${CLASSIFIER:-"lr"}
ALL_LAYERS=${ALL_LAYERS:-"no"}

echo "Running evaluation for:"
echo "  - Model: $MODEL"
if [ "$MODEL" = "reve" ]; then
    echo "  - Size: $MODEL_SIZE"
    echo "  - Pooling: $POOLING"
fi
echo "  - All Layers: $ALL_LAYERS"
echo "  - Representation: $REPRESENTATION"
echo "  - Target: $TARGET_COL"
echo "  - Classifier: $CLASSIFIER"

# Build command
if [ "$MODEL" = "reve" ]; then
    EMBEDDINGS_DIR="/home/mat/scratch/extracted_embeddings/reve/"
    STAGE="baseline"
else
    EMBEDDINGS_DIR="/home/mat/scratch/extracted_embeddings/cbramod"
    STAGE="base"
fi

cmd=(python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve/evaluation.py \
    --model "$MODEL" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --metadata /home/mat/scratch/EEG_Psychostimulants_PatientList_08-2025.csv \
    --classifier "$CLASSIFIER" \
    --stage "$STAGE" \
    --representation "$REPRESENTATION")

if [ "$TARGET_COL" = "EO_EC" ]; then
    cmd+=(--eoec)
else
    cmd+=(--target_col "$TARGET_COL")
fi

if [ "$MODEL" = "reve" ]; then
    cmd+=(--model_size "$MODEL_SIZE")
    cmd+=(--pooling "$POOLING")
fi

if [ "$ALL_LAYERS" = "yes" ]; then
    cmd+=(--all_layers)
fi

echo "Executing: ${cmd[@]}"
"${cmd[@]}"

echo "Job $SLURM_JOB_ID finished at $(date)"
