# Motor Sanity Check Workflow

This directory contains scripts to extract and analyze deep embeddings from the Motor EEG dataset (EO vs EC).

## 1. Embedding Extraction
Run the appropriate shell script for the model you want to evaluate. These will save embeddings to `/home/mat/scratch/motor_extracted_embeddings/`.

```bash
# CBraMod (1s patches)
bash run_make_embeddings_motor.sh

# REVE (10s windows)
bash run_reve_extract_motor.sh
```

## 2. Dimensionality Reduction Reports
Generate interactive HTML reports to visualize the embeddings (PCA, UMAP). Reports are saved to `data/results/sanity_reports/`.

```bash
# Analyze CBraMod
bash run_analyze_deep_embeddings_motor.sh --model cbramod

# Analyze REVE
bash run_analyze_deep_embeddings_motor.sh --model reve
```

## 3. Direct Scoring
Compute high-dimensional classification accuracy (EO vs EC) using Group K-Fold Logistic Regression. Results are printed to stdout.

```bash
# Example for high-dim scoring
python score_embeddings_motor.py --model cbramod --representation subject_flat
```

## Directory Structure
- `make_embeddings_motor.py`: CBraMod extraction logic.
- `reve_extract_motor.py`: REVE extraction logic.
- `analyze_deep_embeddings_motor.py`: Visual analysis and report generation.
- `score_embeddings_motor.py`: Direct high-dimensional separation scoring.
