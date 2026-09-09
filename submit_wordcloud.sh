#!/bin/bash
#SBATCH --job-name=wordcloud
#SBATCH --output=/home/mat/scratch/EEG_results/logs/wordcloud_%j.out
#SBATCH --error=/home/mat/scratch/EEG_results/logs/wordcloud_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-kjerbi

source /home/mat/projects/EEG_psychostimulant/.venv/bin/activate
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/ml/new_ml/visualize:/home/mat/projects/coco-pipe:$PYTHONPATH"
cd /home/mat/projects/EEG_psychostimulant/

python3 generate_wordcloud.py
