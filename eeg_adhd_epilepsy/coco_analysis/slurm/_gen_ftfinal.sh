#!/bin/bash
# Finetuning WITH cohort breakdown, corrected earliest-per-condition labels, best
# config per model. 6 LoRA models x {ft_only,lp_ft} x epoch x {EO,EC} = 24 GPU tasks
# in ONE array (subject-level runs on CPU via _gen_ftcpu.sh).
# driver builds {FINAL}/{level}/{strategy}/{group}/results_{model}_{cond}.json itself
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
FINAL=/home/mat/scratch/results/clean_results/finetuning
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )

# per-model PYTHONPATH (MOIRAI needs uni2ts + ~/.local jax/hydra; NeuroLM must NOT
# have ~/.local so the venv's transformers-compatible huggingface-hub wins).
pp_moirai="/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:/home/mat/.local/lib/python3.11/site-packages"
pp_neurolm="/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM"
COMMON="export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"

P=$SLURM/params/ftfinal.params; : > "$P"
for model in cbramod reve luna biot moirai neurolm; do
 for strat in ft_only lp_ft; do for cond in EO EC; do
   level=epoch
   if [ "$model" = "moirai" ]; then
     env="$COMMON export PYTHONPATH=$pp_moirai HF_HOME=/home/mat/scratch/hf_cache;"
     cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --label-csv ${LBL[$cond]} --out-dir $FINAL"
   elif [ "$model" = "neurolm" ]; then
     env="$COMMON export PYTHONPATH=$pp_neurolm;"
     cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --label-csv ${LBL[$cond]} --out-dir $FINAL"
   else
     env="$COMMON export PYTHONPATH=/home/mat/projects/coco-pipe;"
     cmd="python $CA/run_finetune_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --label-csv ${LBL[$cond]} --out-dir $FINAL"
   fi
   echo "ftfinal_${model}_${strat}_${level}_${cond}|$env mkdir -p $FINAL; $cmd" >> "$P"
 done; done
done
id=$(DEP=${DEP:-} $SLURM/_submit_array.sh ftfinal "$P" \
     --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=6 --mem=32G --time=06:00:00)
echo "ftfinal: array $id ($(grep -c . "$P") GPU tasks) -> $FINAL"
