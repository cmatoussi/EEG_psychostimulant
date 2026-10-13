#!/bin/bash
# FULL handcrafted reports (feature_importance + summary/SFS + per-sensor + HTML)
# predicting epilepsy, with corrected earliest-per-condition labels. One array task
# per cohort-group x condition (EO/EC now run in parallel instead of sequentially).
# Layout matches the original: subject_level/{group}/{cohort}/...
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
BASE=/home/mat/scratch/results/clean_results/handcrafted/epilepsy/subject_level
# group -> out-dir (nest cohorts under the group name, like the reference)
declare -A OUT=( [all]="$BASE" [age]="$BASE/age" [sex]="$BASE/sex" [comorbidity]="$BASE/comorbidities" )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg;"

P=$SLURM/params/hcrep.params; : > "$P"
for group in all age sex comorbidity; do for cond in EO EC; do
  echo "hcrep_${group}_${cond}|$ENV mkdir -p ${OUT[$group]}; python $CA/run_handcrafted_cohorts.py --level subject --cohort-group ${group} --condition ${cond}_baseline --target-col epilepsy --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir ${OUT[$group]}" >> "$P"
done; done
id=$($SLURM/_submit_array.sh hcrep "$P" --cpus-per-task=32 --mem=32G --time=08:00:00)
echo "hcrep: array $id ($(grep -c . "$P") tasks) -> $BASE"
