#!/bin/bash
# Handcrafted ASM_RESISTANT reports (resistant vs non-resistant WITHIN epilepsy),
# corrected earliest-per-condition labels. Same driver/analysis as the epilepsy
# handcrafted (feature_importance + summary/SFS + l1_select + per-sensor + HTML,
# pooled metrics), target=asm_resistant, restricted to epilepsy==1. Groups:
# all/age/sex/comorbidity (no 'drug' -- controls-vs-drug is undefined within epilepsy).
# One array per level (mem/walltime differ); task = group x condition.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
BASE=/home/mat/scratch/results/clean_results/handcrafted/asm_resistant
declare -A SUB=( [all]="" [age]="/age" [sex]="/sex" [comorbidity]="/comorbidities" )
declare -A WT=( [subject]="08:00:00" [epoch]="23:59:00" )
declare -A MEM=( [subject]="32G" [epoch]="48G" )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg;"

for level in subject epoch; do
  P=$SLURM/params/hcasm_${level}.params; : > "$P"
  for group in all age sex comorbidity; do for cond in EO EC; do
    out="$BASE/${level}_level${SUB[$group]}"
    echo "hcasm_${level}_${group}_${cond}|$ENV mkdir -p $out; python $CA/run_handcrafted_cohorts.py --level ${level} --cohort-group ${group} --target-col asm_resistant --restrict-col epilepsy --condition ${cond}_baseline --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir $out" >> "$P"
  done; done
  id=$($SLURM/_submit_array.sh hcasm_${level} "$P" --cpus-per-task=32 --mem=${MEM[$level]} --time=${WT[$level]})
  echo "hcasm_${level}: array $id ($(grep -c . "$P") tasks) -> $BASE"
done
