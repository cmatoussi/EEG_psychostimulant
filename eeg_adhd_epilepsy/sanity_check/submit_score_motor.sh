#!/bin/bash
#SBATCH --job-name=score_motor
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/score_motor_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/score_motor_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Environment Setup
source /home/mat/projects/EEG_psychostimulant/dim_red/bin/activate
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy:${PYTHONPATH:-}"

echo "Running Streamlined Embedding Scoring (EO vs EC) for Motor Embeddings..."

# 1. REVE Base
echo "--- Scoring REVE Base (epoch_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/score_embeddings_motor.py \
    --model reve --reve_size base --pooling with --representation epoch_flat

echo "--- Scoring REVE Base (subject_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/score_embeddings_motor.py \
    --model reve --reve_size base --pooling with --representation subject_flat

# 2. REVE Large
echo "--- Scoring REVE Large (epoch_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/score_embeddings_motor.py \
    --model reve --reve_size large --pooling with --representation epoch_flat

echo "--- Scoring REVE Large (subject_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/score_embeddings_motor.py \
    --model reve --reve_size large --pooling with --representation subject_flat

# 3. CBraMod Last Layer
echo "--- Scoring CBraMod (epoch_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/score_embeddings_motor.py \
    --model cbramod --all_layers 0 --representation epoch_flat

echo "--- Scoring CBraMod (subject_flat) ---"
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/score_embeddings_motor.py \
    --model cbramod --all_layers 0 --representation subject_flat

echo "Scoring complete."
