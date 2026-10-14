#!/bin/bash
# Pretraining ablation: every LoRA-FT model with a RANDOM-INIT backbone — same
# architecture, same LoRA budget, same recipe/folds/seeds; only the weights differ.
# Results under {model}_rand next to the pretrained runs, so the paired
# permutation test quantifies the pretraining edge per model.
# epoch -> GPU array (24h; reve/neurolm needed 7-10h pretrained),
# subject -> CPU array (24h; cbramod/luna exceeded 12h pretrained).
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
FINAL=/home/mat/scratch/results/clean_results/finetuning
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
pp_moirai="/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:/home/mat/.local/lib/python3.11/site-packages"
pp_neurolm="/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM"
COMMON="export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"

for level in epoch subject; do
  P=$SLURM/params/ftrand_${level}.params; : > "$P"
  for model in cbramod reve luna biot moirai neurolm; do
   for strat in ft_only lp_ft; do for cond in EO EC; do
     if [ "$model" = "moirai" ]; then
       env="$COMMON export PYTHONPATH=$pp_moirai HF_HOME=/home/mat/scratch/hf_cache;"
       cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --random-init --label-csv ${LBL[$cond]} --out-dir $FINAL"
     elif [ "$model" = "neurolm" ]; then
       env="$COMMON export PYTHONPATH=$pp_neurolm;"
       cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --random-init --label-csv ${LBL[$cond]} --out-dir $FINAL"
     else
       env="$COMMON export PYTHONPATH=/home/mat/projects/coco-pipe;"
       cmd="python $CA/run_finetune_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --random-init --label-csv ${LBL[$cond]} --out-dir $FINAL"
     fi
     echo "ftrand_${model}_${strat}_${level}_${cond}|$env $cmd" >> "$P"
   done; done
  done
done

eid=$($SLURM/_submit_array.sh ftrand_epoch $SLURM/params/ftrand_epoch.params \
      --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=6 --mem=32G --time=23:59:00)
echo "ftrand_epoch: array $eid (24 GPU tasks) -> $FINAL"
sid=$($SLURM/_submit_array.sh ftrand_subject $SLURM/params/ftrand_subject.params \
      --account=rrg-kjerbi --cpus-per-task=64 --mem=48G --time=23:59:00)
echo "ftrand_subject: array $sid (24 CPU tasks) -> $FINAL"
