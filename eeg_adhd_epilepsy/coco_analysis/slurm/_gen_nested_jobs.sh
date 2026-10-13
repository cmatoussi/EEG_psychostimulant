#!/bin/bash
# Nested-Optuna head HPO on frozen embeddings, corrected earliest-per-condition
# labels. Phase 1 = subject-level (averaged_epochs) array for all models; Phase 2 =
# epoch-level array chained afterany on the WHOLE phase-1 array.
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/probe_nested_optuna.py
BASE=/home/mat/scratch/results/embedding_performance/baseline
MODELS="bendr biot cbramod eegpt labram luna reve signaljepa moirai neurolm"
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
declare -A WT=( [subject]="12:00:00" [epoch]="23:59:00" )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

emit () {  # emit <level>  -> writes the params file for that level
  local level=$1
  local P=$SLURM/params/nested_${level}.params
  : > "$P"
  for model in $MODELS; do for cond in EO EC; do
    echo "nested_${level}_${model}_${cond}|$ENV python $DRV --model ${model} --condition ${cond} --level ${level} --cohort-group everything --label-csv ${LBL[$cond]} --out-dir $BASE/${level} --n-trials 40 --inner-folds 3" >> "$P"
  done; done
}

emit subject
SID=$($SLURM/_submit_array.sh nested_subject $SLURM/params/nested_subject.params \
      --cpus-per-task=32 --mem=32G --time=${WT[subject]})
echo "nested_subject: array $SID (20 tasks)"

emit epoch
EID=$(DEP=afterany:$SID $SLURM/_submit_array.sh nested_epoch $SLURM/params/nested_epoch.params \
      --cpus-per-task=32 --mem=32G --time=${WT[epoch]})
echo "nested_epoch: array $EID (20 tasks, afterany:$SID)"
