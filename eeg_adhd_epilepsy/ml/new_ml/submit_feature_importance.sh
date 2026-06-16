#!/bin/bash
#SBATCH --job-name=feature_imp
#SBATCH --output=/home/mat/scratch/EEG_results/logs/feature_imp_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/feature_imp_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-kjerbi

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant/

echo "=================================================="
echo "Starting Feature Importance ML Pipeline"
echo "=================================================="

# 1. Run the ML Pipeline (Generates .pkl file)
python3 eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py \
    --config eeg_adhd_epilepsy/ml/new_ml/config_feature_importance.yml

echo "ML Pipeline Complete. Generating Custom Visualizations..."

# 2. Run the Visualizer (separated)
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py \
    --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso.pkl \
    --out-dir /home/mat/projects/EEG_psychostimulant/data/results/hand_crafted \
    --model "Logistic Regression" \
    --config eeg_adhd_epilepsy/ml/new_ml/config_feature_importance.yml

python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_feature.py \
    --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso.pkl \
    --out-dir /home/mat/projects/EEG_psychostimulant/data/results/hand_crafted \
    --model "Logistic Regression" \
    --config eeg_adhd_epilepsy/ml/new_ml/config_feature_importance.yml

python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_per_sensor.py \
    --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso.pkl \
    --out-dir /home/mat/projects/EEG_psychostimulant/data/results/hand_crafted \
    --model "Logistic Regression" \
    --config eeg_adhd_epilepsy/ml/new_ml/config_feature_importance.yml

echo "=================================================="
echo "Job Complete."
echo "=================================================="
