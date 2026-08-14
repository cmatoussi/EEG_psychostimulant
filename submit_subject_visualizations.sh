#!/bin/bash
#SBATCH --job-name=vis_subject
#SBATCH --output=/home/mat/scratch/EEG_results/logs/vis_subject_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/vis_subject_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-kjerbi

source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/ml/new_ml/visualize:/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant/

echo "Running sensor subject..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml

echo "Running pooled subject..."
python3 eeg_adhd_epilepsy/ml/new_ml/visualize/visualize_global_lasso.py --results /home/mat/scratch/EEG_results/results/ml/new_ml/feature_importance/epilepsy_feature_importance_lasso_pooled_subject.pkl --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_subject.yml

echo "Done!"
