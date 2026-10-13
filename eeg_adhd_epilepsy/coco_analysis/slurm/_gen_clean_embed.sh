#!/bin/bash
# FAST fixed-head embedding classification (NO finetuning, NO nested Optuna) on
# frozen FM embeddings, corrected earliest-per-condition labels. Same head set as
# before (dummy/logreg/histgb/rf/svm), cohort breakdown (all/sex/age/comorbidity),
# subject + epoch. One job array (40 tasks, all run in parallel).
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/probe_nested_optuna.py
BASE=/home/mat/scratch/results/clean_results/embeddings_classification/epilepsy/baseline
MODELS="bendr biot cbramod eegpt labram luna reve signaljepa moirai neurolm"
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

P=$SLURM/params/cleanemb.params; : > "$P"
for level in subject epoch; do for model in $MODELS; do for cond in EO EC; do
  echo "cleanemb_${level}_${model}_${cond}|$ENV python $DRV --model ${model} --condition ${cond} --level ${level} --fixed --cohort-group everything --label-csv ${LBL[$cond]} --out-dir $BASE/${level}" >> "$P"
done; done; done
id=$($SLURM/_submit_array.sh cleanemb "$P" --cpus-per-task=32 --mem=32G --time=12:00:00)
echo "cleanemb: array $id ($(grep -c . "$P") tasks) -> $BASE"
