#!/bin/bash
#SBATCH --job-name=analyze_motor_sanity
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/analyze_motor_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/analyze_motor_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Environment Setup
source /home/mat/projects/EEG_psychostimulant/dim_red/bin/activate
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy:${PYTHONPATH:-}"

EMB_DIR="/home/mat/scratch/motor_extracted_embeddings/"
REDUCERS="pca umap phate isomap"

echo "Running Sanity Check Analysis (EO vs EC) for Motor Embeddings..."

# 1. REVE Base
echo "--- Running REVE Base (subject_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/analyze_deep_embeddings_motor.py \
    --embeddings_dir "$EMB_DIR" \
    --model reve \
    --reve_size base \
    --pooling with \
    --representation subject_flat \
    --reducers $REDUCERS

echo "--- Running REVE Base (epoch_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/analyze_deep_embeddings_motor.py \
    --embeddings_dir "$EMB_DIR" \
    --model reve \
    --reve_size base \
    --pooling with \
    --representation epoch_flat \
    --reducers $REDUCERS

# 2. CBraMod Last Layer
echo "--- Running CBraMod (subject_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/analyze_deep_embeddings_motor.py \
    --embeddings_dir "$EMB_DIR" \
    --model cbramod \
    --all_layers 0 \
    --representation subject_flat \
    --reducers $REDUCERS

echo "--- Running CBraMod (epoch_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/analyze_deep_embeddings_motor.py \
    --embeddings_dir "$EMB_DIR" \
    --model cbramod \
    --all_layers 0 \
    --representation epoch_flat \
    --reducers $REDUCERS

echo "Sanity check analysis complete. Reports saved in /home/mat/projects/EEG_psychostimulant/data/results/sanity_reports/"
