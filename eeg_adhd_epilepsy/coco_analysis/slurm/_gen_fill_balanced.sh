#!/bin/bash
# Fill the missing models in the BALANCED embedding results (male, age_9_12), which
# never produced output (heavy models timed out on SVM). Rerun just those models,
# no SVM, merged into the existing cohort CSVs (good models preserved).
# One array, 64 CPU / 3h / rrg-kjerbi.
set -u
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/run_embedding_cohorts.py
OUT=/home/mat/scratch/results/clean_results/embeddings_classification/epilepsy/balanced
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8;"

P=$SLURM/params/fillbal.params; : > "$P"
echo "fillbal_male|$ENV python $DRV --cohort-group sex --cohort M --balanced --models bendr biot eegpt reve --heads logreg histgb rf --aggregation both --target-col epilepsy --out-dir $OUT" >> "$P"
echo "fillbal_age9_12|$ENV python $DRV --cohort-group age --cohort 9-12 --balanced --models bendr eegpt reve --heads logreg histgb rf --aggregation both --target-col epilepsy --out-dir $OUT" >> "$P"
id=$($SLURM/_submit_array.sh fillbal "$P" --account=rrg-kjerbi --cpus-per-task=64 --mem=64G --time=3:00:00)
echo "fillbal: array $id (2 tasks)"
