#!/bin/bash
# Finetune jobs for MOIRAI / NeuroLM LoRA finetuning. Per model: 8 finetunes
# (ft_only/lp_ft x epoch/subject x EO/EC). Fixed LoRA config (no tuning stage --
# the old Optuna search leaked across folds). One GPU array per level (walltimes
# differ: subject 8h, epoch 24h).
SLURM=/home/mat/scratch/results/slurm_scripts
DRV=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis/finetune_tslm_lora.py
RT=/home/mat/scratch/results/fine_tune/ray_tuned
declare -A DIR=( [ft_only_epoch]=ft_epoch [ft_only_subject]=ft_subject [lp_ft_epoch]=lp_ft_epoch [lp_ft_subject]=lp_ft_subject )
declare -A WT=( [subject]="08:00:00" [epoch]="23:59:00" )

# per-model PYTHONPATH (MOIRAI needs uni2ts + ~/.local jax/hydra; NeuroLM must NOT
# have ~/.local so the venv's transformers-compatible huggingface-hub wins).
pp_moirai="/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:/home/mat/.local/lib/python3.11/site-packages"
pp_neurolm="/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM"
COMMON="export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True;"

for level in epoch subject; do
  P=$SLURM/params/tslm_ft_${level}.params; : > "$P"
  for model in moirai neurolm; do
    if [ "$model" = "moirai" ]; then
      env="$COMMON export PYTHONPATH=$pp_moirai HF_HOME=/home/mat/scratch/hf_cache;"
    else
      env="$COMMON export PYTHONPATH=$pp_neurolm;"
    fi
    for strat in ft_only lp_ft; do for cond in EO EC; do
      out=$RT/${DIR[${strat}_${level}]}
      echo "tslm_ft_${model}_${strat}_${level}_${cond}|$env mkdir -p $out; /home/mat/ep/bin/python $DRV --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --out-dir $out" >> "$P"
    done; done
  done
  id=$($SLURM/_submit_array.sh tslm_ft_${level} "$P" \
       --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=8 --mem=64G --time=${WT[$level]})
  echo "tslm_ft_${level}: array $id ($(grep -c . "$P") GPU tasks)"
done
