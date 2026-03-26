#!/bin/bash
#SBATCH --job-name=analyze_emb
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/analyze_emb_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/analyze_emb_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

# Load modules if necessary (adjust if your cluster uses 'module load')
# module load python/3.11

# Activate the environment

source /home/mat/projects/EEG_psychostimulant/dim_red/bin/activate

# 2. Configuration
MODEL=${MODEL:-"reve"}
REVE_SIZE=${REVE_SIZE:-"base"}
POOLING=${POOLING:-"with"} # "with" or "without"
TARGET_COL=${TARGET_COL:-"EO_EC"}
REPRESENTATION=${REPRESENTATION:-"epoch_flat"}
REDUCERS=${REDUCERS:-"pca"}
CONDITIONS=${CONDITIONS:-"EC_baseline EO_baseline"}

# desc defaults differ by model: reve uses 'baseline', cbramod uses 'base'
if [ -z "${DESC:-}" ]; then
    if [ "$MODEL" = "cbramod" ]; then
        DESC="base"
    else
        DESC="baseline"
    fi
fi

echo "Running analysis for:"
echo "  - Model: $MODEL"
if [ "$MODEL" = "reve" ]; then
    echo "  - Size: $REVE_SIZE"
    echo "  - Pooling: $POOLING"
fi
echo "  - Target: $TARGET_COL"
echo "  - Reducers: $REDUCERS"
echo "  - Conditions: $CONDITIONS"

# 3. Build command
if [ "$MODEL" = "reve" ]; then
    EMBEDDINGS_DIR="/home/mat/scratch/extracted_embeddings/reve/"
else
    EMBEDDINGS_DIR="/home/mat/scratch/extracted_embeddings/cbramod"
fi

cmd=(python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/analysis/analyze_deep_embeddings.py \
    --representation "$REPRESENTATION" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --model "$MODEL" \
    --desc "$DESC" \
    --metadata /home/mat/scratch/EEG_Psychostimulants_PatientList_08-2025.csv \
    --target_col "$TARGET_COL" \
    --reducers $REDUCERS \
    --conditions $CONDITIONS \
    --max_samples 100000)

if [ "$MODEL" = "reve" ]; then
    cmd+=(--reve_size "$REVE_SIZE")
    cmd+=(--pooling "$POOLING")
fi

# 4. Execute
"${cmd[@]}"

echo "Job $SLURM_JOB_ID finished at $(date)"
