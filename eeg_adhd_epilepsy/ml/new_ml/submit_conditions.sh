#!/bin/bash
#SBATCH --job-name=feature_imp_conds
#SBATCH --output=/home/mat/scratch/EEG_results/logs/feature_imp_conds_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/feature_imp_conds_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=def-kjerbi

source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant/

python3 eeg_adhd_epilepsy/ml/new_ml/run_all_conditions_pipeline.py

