#!/bin/bash
#SBATCH --job-name=cbramod_extract_motor
#SBATCH --output=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/cbramod_extract_motor_%j.out
#SBATCH --error=/home/mat/projects/EEG_psychostimulant/data/results/analysis/logs/cbramod_extract_motor_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1

# 1. Environment Setup
source /home/mat/ep/bin/activate
PROJECT_ROOT="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 2. Configuration
DERIV_ROOT=${DERIV_ROOT:-/home/mat/scratch/motor}
OUT_DIR=${OUT_DIR:-/home/mat/scratch/motor_extracted_embeddings/}
WEIGHTS=${WEIGHTS:-/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth}
DEVICE=${DEVICE:-cuda}

echo "Submitting CBraMod extraction for motor data..."
echo "  - Output Dir: $OUT_DIR"

# 3. Execute (Removed --max-subjects for full run)
python /home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/make_embeddings_motor.py \
  --deriv-proc-root "$DERIV_ROOT" \
  --out-file "$OUT_DIR" \
  --weights "$WEIGHTS" \
  --device "$DEVICE"

echo "Job $SLURM_JOB_ID finished at $(date)"
