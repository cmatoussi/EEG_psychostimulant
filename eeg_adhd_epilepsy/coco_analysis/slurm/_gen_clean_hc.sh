#!/bin/bash
# FAST fixed-head HANDCRAFTED sensor-feature classification predicting epilepsy,
# corrected earliest-per-condition labels (2nd recording removed). Same heads/metrics
# as the clean embedding run, cohort-group=everything (all/sex/age/comorbidity).
# subject + epoch, EO + EC. One job array per level (walltimes differ).
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/probe_nested_optuna.py
BASE=/home/mat/scratch/results/clean_results/handcrafted/baseline
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
declare -A WT=( [subject]="04:00:00" [epoch]="23:59:00" )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

for level in subject epoch; do
  P=$SLURM/params/cleanhc_${level}.params; : > "$P"
  for cond in EO EC; do
    echo "cleanhc_${level}_${cond}|$ENV python $DRV --model handcrafted --condition ${cond} --level ${level} --fixed --target-col epilepsy --cohort-group everything --label-csv ${LBL[$cond]} --out-dir $BASE/${level}" >> "$P"
  done
  id=$($SLURM/_submit_array.sh cleanhc_${level} "$P" --cpus-per-task=32 --mem=48G --time=${WT[$level]})
  echo "cleanhc_${level}: array $id ($(grep -c . "$P") tasks) -> $BASE/${level}"
done
