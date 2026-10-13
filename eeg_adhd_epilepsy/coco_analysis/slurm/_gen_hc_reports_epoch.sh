#!/bin/bash
# EPOCH-level full handcrafted reports (feature_importance + summary/nested-SFS +
# per-sensor + HTML), corrected earliest-per-condition labels, epilepsy target.
# Heavier than subject-level (30-50k epochs): 32 CPU / 48G / 24h. SFS runs on a
# subject-grouped subsample (SFS_MAX_ROWS); RF/full-model use all epochs.
# One array task per cohort-group x condition.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
BASE=/home/mat/scratch/results/clean_results/handcrafted/epilepsy/epoch_level
declare -A OUT=( [all]="$BASE" [age]="$BASE/age" [sex]="$BASE/sex" [comorbidity]="$BASE/comorbidities" )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg;"

P=$SLURM/params/hcrepE.params; : > "$P"
for group in all age sex comorbidity; do for cond in EO EC; do
  echo "hcrepE_${group}_${cond}|$ENV mkdir -p ${OUT[$group]}; python $CA/run_handcrafted_cohorts.py --level epoch --cohort-group ${group} --condition ${cond}_baseline --target-col epilepsy --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir ${OUT[$group]}" >> "$P"
done; done
id=$($SLURM/_submit_array.sh hcrepE "$P" --cpus-per-task=32 --mem=48G --time=23:59:00)
echo "hcrepE: array $id ($(grep -c . "$P") epoch-level tasks) -> $BASE"
