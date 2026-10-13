#!/bin/bash
# Handcrafted nested-Optuna fleet: subject array (now) + epoch array (chained
# afterany on the whole subject array), 32 CPU / 32G.
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/probe_nested_optuna.py
BASE=/home/mat/scratch/results/embedding_performance/hand_crafted/baseline
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
declare -A WT=( [subject]="16:00:00" [epoch]="23:59:00" )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

for level in subject epoch; do
  P=$SLURM/params/hc_nested_${level}.params; : > "$P"
  for cond in EO EC; do
    echo "hc_nested_${level}_${cond}|$ENV python $DRV --model handcrafted --condition ${cond} --level ${level} --cohort-group everything --label-csv ${LBL[$cond]} --out-dir $BASE/runs/${level} --heads logreg rf dummy --n-trials 40 --inner-folds 3" >> "$P"
  done
done

SID=$($SLURM/_submit_array.sh hc_nested_subject $SLURM/params/hc_nested_subject.params \
      --cpus-per-task=32 --mem=32G --time=${WT[subject]})
echo "hc_nested_subject: array $SID"
EID=$(DEP=afterany:$SID $SLURM/_submit_array.sh hc_nested_epoch $SLURM/params/hc_nested_epoch.params \
      --cpus-per-task=32 --mem=32G --time=${WT[epoch]})
echo "hc_nested_epoch: array $EID (afterany:$SID)"
