#!/bin/bash
#SBATCH --job-name=multi_cond_handcrafted
#SBATCH --output=/home/mat/scratch/EEG_results/logs/multi_cond_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/multi_cond_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-kjerbi

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant/

echo "=================================================="
echo "Starting Multi-Condition Feature Importance Pipeline"
echo "=================================================="

# 1. Run the ML Pipeline for all 8 conditions
python3 eeg_adhd_epilepsy/ml/new_ml/run_multi_condition_handcrafted.py

echo "=================================================="
echo "ML Pipeline Complete. Analyzing common features..."
echo "=================================================="

# 2. Extract Top 10 Most Common Features
python3 eeg_adhd_epilepsy/ml/new_ml/analyze_common_features.py

echo "=================================================="
echo "Job Complete."
echo "=================================================="
