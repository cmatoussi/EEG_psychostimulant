#!/bin/bash
# FAST fixed-head embedding classification predicting ASM_RESISTANT (drug-resistance)
# instead of epilepsy. Same models/heads/metrics/labels (corrected earliest-per-condition)
# as the epilepsy clean run, but cohort-group=all ONLY (no sex/age/comorbidity split).
# subject + epoch, EO + EC. One job array (40 tasks).
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/probe_nested_optuna.py
BASE=/home/mat/scratch/results/clean_results/embeddings_classification/asm_resistant/baseline
MODELS="bendr biot cbramod eegpt labram luna reve signaljepa moirai neurolm"
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

P=$SLURM/params/cleanasm.params; : > "$P"
for level in subject epoch; do for model in $MODELS; do for cond in EO EC; do
  echo "cleanasm_${level}_${model}_${cond}|$ENV python $DRV --model ${model} --condition ${cond} --level ${level} --fixed --target-col asm_resistant --restrict-col epilepsy --cohort-group all --label-csv ${LBL[$cond]} --out-dir $BASE/${level}" >> "$P"
done; done; done
id=$($SLURM/_submit_array.sh cleanasm "$P" --cpus-per-task=32 --mem=32G --time=12:00:00)
echo "cleanasm: array $id ($(grep -c . "$P") tasks) -> $BASE"
