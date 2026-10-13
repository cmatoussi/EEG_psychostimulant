#!/bin/bash
# Drug-SIGNATURE handcrafted reports (subject level), earliest-per-condition labels.
# Cohorts decode the medication WITHIN epilepsy (lev_vs_vpa / lev_vs_naive /
# vpa_vs_naive), so features reflect the drug's EEG effect, not epilepsy. The driver
# sets the per-cohort target column; --target-col here is just the restrict default.
# One array, task = condition.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
OUT=/home/mat/scratch/results/clean_results/handcrafted/epilepsy/subject_level/drug_signature
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg;"

P=$SLURM/params/hcrep_drug.params; : > "$P"
for cond in EO EC; do
  echo "hcrep_drug_${cond}|$ENV mkdir -p $OUT; python $CA/run_handcrafted_cohorts.py --level subject --cohort-group drug --condition ${cond}_baseline --target-col epilepsy --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir $OUT" >> "$P"
done
id=$($SLURM/_submit_array.sh hcrep_drug "$P" --cpus-per-task=32 --mem=32G --time=08:00:00)
echo "hcrep_drug: array $id (2 tasks) -> $OUT"
