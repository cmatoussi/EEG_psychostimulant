#!/bin/bash
#SBATCH --job-name=cbramod_layer_sf
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/logs/layer_eval_sf_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/logs/layer_eval_sf_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=def-kjerbi

export CONDITION=${1:-EO_baseline}
export LIMIT=${2:-1013}

cd /home/mat/projects/EEG_psychostimulant

echo "Starting CBraMod layer evaluation on Slurm..."
echo "Condition: $CONDITION | Limit: $LIMIT"

bash run_all_layers_subject_flat.sh --limit $LIMIT --condition $CONDITION
