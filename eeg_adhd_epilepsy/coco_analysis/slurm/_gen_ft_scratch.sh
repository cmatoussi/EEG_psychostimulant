#!/bin/bash
# From-scratch small-CNN baselines (eegnet, shallow): no pretraining, no LoRA,
# whole net trained on the task — does a compact supervised CNN match the FMs?
# Same folds/seeds/metrics as the FT sweeps; results under strategy "scratch".
# epoch level -> GPU array; subject level -> CPU array (tiny data, fast).
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
FINAL=/home/mat/scratch/results/clean_results/finetuning
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"

for level in epoch subject; do
  P=$SLURM/params/scratch_${level}.params; : > "$P"
  for model in eegnet shallow; do for cond in EO EC; do
    echo "scratch_${model}_${level}_${cond}|$ENV python $CA/run_train_scratch.py --model ${model} --condition ${cond}_baseline --level ${level} --label-csv ${LBL[$cond]} --out-dir $FINAL" >> "$P"
  done; done
done

eid=$($SLURM/_submit_array.sh scratch_epoch $SLURM/params/scratch_epoch.params \
      --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=6 --mem=32G --time=06:00:00)
echo "scratch_epoch: array $eid (4 GPU tasks) -> $FINAL"
sid=$($SLURM/_submit_array.sh scratch_subject $SLURM/params/scratch_subject.params \
      --account=rrg-kjerbi --cpus-per-task=32 --mem=32G --time=03:00:00)
echo "scratch_subject: array $sid (4 CPU tasks) -> $FINAL"
