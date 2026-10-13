#!/bin/bash
# Fill the baseline heatmap blanks: rerun reve + eegpt (which timed out on SVM),
# now WITHOUT svm so they finish fast. Fixed default heads (baseline), full cohort
# sweep, subject + epoch, EO + EC. One array, 128 CPU / 3h / rrg-kjerbi.
set -u
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/probe_nested_optuna.py
BASE=/home/mat/scratch/results/clean_results/embeddings_classification/epilepsy/baseline
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8;"

P=$SLURM/params/fillbase.params; : > "$P"
for model in reve eegpt; do for level in subject epoch; do for cond in EO EC; do
  echo "fillbase_${level}_${model}_${cond}|$ENV python $DRV --model ${model} --condition ${cond} --level ${level} --fixed --cohort-group everything --label-csv ${LBL[$cond]} --heads dummy logreg histgb rf --out-dir $BASE/${level}" >> "$P"
done; done; done
id=$($SLURM/_submit_array.sh fillbase "$P" --account=rrg-kjerbi --cpus-per-task=128 --mem=64G --time=3:00:00)
echo "fillbase: array $id ($(grep -c . "$P") tasks, reve+eegpt no svm)"
