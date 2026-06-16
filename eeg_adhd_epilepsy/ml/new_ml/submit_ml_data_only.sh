#!/bin/bash
#SBATCH --job-name=lasso_ml_data
#SBATCH --output=/home/mat/scratch/EEG_results/logs/lasso_ml_data_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/lasso_ml_data_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=def-kjerbi

# Activate environment
source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant/

echo "=================================================="
echo "Running ML Pipeline - Sensor Epoch"
echo "=================================================="
python3 eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_epoch.yml

echo "=================================================="
echo "Running ML Pipeline - Sensor Subject"
echo "=================================================="
python3 eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_sensor_subject.yml

echo "=================================================="
echo "Running ML Pipeline - Pooled Epoch"
echo "=================================================="
python3 eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_epoch.yml

echo "=================================================="
echo "Running ML Pipeline - Pooled Subject"
echo "=================================================="
python3 eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py --config eeg_adhd_epilepsy/ml/new_ml/config_lasso_pooled_subject.yml

echo "=================================================="
echo "Job Complete."
echo "=================================================="
