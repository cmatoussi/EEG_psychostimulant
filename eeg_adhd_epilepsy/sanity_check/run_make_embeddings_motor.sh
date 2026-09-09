#!/usr/bin/env bash
set -euo pipefail

# Path configuration
DERIV_ROOT=${DERIV_ROOT:-/home/mat/scratch/motor}
OUT_DIR=${OUT_DIR:-/home/mat/scratch/motor_extracted_embeddings/}
WEIGHTS=${WEIGHTS:-/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth}
DEVICE=${DEVICE:-cuda}

PYTHON_BIN=${PYTHON_BIN:-/home/mat/ep/bin/python}

cmd=("$PYTHON_BIN" "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/sanity_check/make_embeddings_motor.py" \
  --deriv-proc-root "$DERIV_ROOT" \
  --out-file "$OUT_DIR" \
  --weights "$WEIGHTS" \
  --device "$DEVICE")

echo "Starting extraction with 1s (200-sample) patches..."
"${cmd[@]}"
