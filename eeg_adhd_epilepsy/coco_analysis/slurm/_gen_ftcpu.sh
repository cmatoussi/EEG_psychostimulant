#!/bin/bash
# ALL 6 LoRA models, subject-level (averaged epochs) finetune on CPU. No GPU gres
# -> device auto-detects to CPU. Writes to clean_results/finetuning/subject/...
# (same paths as the GPU jobs) so the GPU subject-level jobs must be cancelled first.
# One array, 24 tasks, 64 CPU / rrg-kjerbi.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
FINAL=/home/mat/scratch/results/clean_results/finetuning
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
pp_moirai="/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:/home/mat/.local/lib/python3.11/site-packages"
pp_neurolm="/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM"
COMMON="export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"

P=$SLURM/params/ftcpu.params; : > "$P"
for model in cbramod reve luna biot moirai neurolm; do
 for strat in ft_only lp_ft; do for cond in EO EC; do
   if [ "$model" = "moirai" ]; then
     env="$COMMON export PYTHONPATH=$pp_moirai HF_HOME=/home/mat/scratch/hf_cache;"
     cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level subject --batch 16 --label-csv ${LBL[$cond]} --out-dir $FINAL"
   elif [ "$model" = "neurolm" ]; then
     env="$COMMON export PYTHONPATH=$pp_neurolm;"
     cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level subject --batch 16 --label-csv ${LBL[$cond]} --out-dir $FINAL"
   else
     env="$COMMON export PYTHONPATH=/home/mat/projects/coco-pipe;"
     cmd="python $CA/run_finetune_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level subject --label-csv ${LBL[$cond]} --out-dir $FINAL"
   fi
   echo "ftcpu_${model}_${strat}_subject_${cond}|$env mkdir -p $FINAL; $cmd" >> "$P"
 done; done
done
id=$($SLURM/_submit_array.sh ftcpu "$P" --account=rrg-kjerbi --cpus-per-task=64 --mem=48G --time=12:00:00)
echo "ftcpu: array $id ($(grep -c . "$P") CPU subject-level finetunes)"
