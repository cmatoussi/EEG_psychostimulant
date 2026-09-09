#!/bin/bash
#SBATCH --job-name=ml_pooled_feat
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/logs/ml_pooled_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/logs/ml_pooled_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=def-kjerbi

cd /home/mat/projects/EEG_psychostimulant
source .venv/bin/activate

echo "Starting ML Pipeline for Pooled Features"
export PYTHONPATH=$PYTHONPATH:/home/mat/projects/coco-pipe
python3 eeg_adhd_epilepsy/ml/new_ml/run_ml_pipe.py --config eeg_adhd_epilepsy/ml/new_ml/config_epilepsy_pooled.yml
echo "Done!"
