#!/usr/bin/env bash
set -euo pipefail

# Path configuration
DERIV_ROOT=${DERIV_ROOT:-/home/mat/scratch/motor}
OUT_DIR=${OUT_DIR:-/home/mat/scratch/motor_extracted_embeddings/}
DEVICE=${DEVICE:-cpu}
MODEL_SIZE=${MODEL_SIZE:-large}

PYTHON_BIN=${PYTHON_BIN:-/home/mat/ep/bin/python}
export PYTHONPATH="/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/dl/reve:${PYTHONPATH:-}"

cmd=("$PYTHON_BIN" "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/reve_extract_motor.py" \
  --data-root "$DERIV_ROOT" \
  --output-dir "$OUT_DIR" \
  --model-size "$MODEL_SIZE" \
  --device "$DEVICE")

echo "Starting REVE extraction on motor data..."
"${cmd[@]}"
