#!/bin/bash
#SBATCH --job-name=reve_extract_motor
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/reve_extract_motor_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/reve_extract_motor_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

# 1. Environment Setup
source /home/mat/ep/bin/activate
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve:${PYTHONPATH:-}"

# 2. Configuration
DERIV_ROOT=${DERIV_ROOT:-/home/mat/scratch/motor}
OUT_DIR=${OUT_DIR:-/home/mat/scratch/motor_extracted_embeddings/}
MODEL_SIZE=${MODEL_SIZE:-base}
DEVICE=${DEVICE:-cuda}

echo "Submitting REVE extraction for motor data..."
echo "  - Model Size: $MODEL_SIZE"
echo "  - Output Dir: $OUT_DIR"

# 3. Execute (Removed --limit for full run)
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/reve_extract_motor.py \
  --data-root "$DERIV_ROOT" \
  --output-dir "$OUT_DIR" \
  --model-size "$MODEL_SIZE" \
  --device "$DEVICE"

echo "Job $SLURM_JOB_ID finished at $(date)"
