#!/bin/bash
# ASM-resistant finetuning (drug-resistance within epilepsy). Uses relabeled CSVs
# (labels_asm_{cond}.csv: epilepsy patients, 'epilepsy' col holds asm_resistant).
# 6 models x {ft_only,lp_ft} x {EO,EC}: subject -> CPU/rrg array (24 tasks),
# epoch -> GPU/def array (24 tasks).
# Driver builds {OUT}/{level}/{strategy}/{group}/results_{model}_{cond}.json.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
OUT=/home/mat/scratch/results/clean_results/finetuning/asm_resistant/baseline
declare -A LBL=( [EO]=/home/mat/scratch/labels_asm_EO.csv [EC]=/home/mat/scratch/labels_asm_EC.csv )
pp_moirai="/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:/home/mat/.local/lib/python3.11/site-packages"
pp_neurolm="/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM"
COMMON="export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"

for level in subject epoch; do
  P=$SLURM/params/ftasm_${level}.params; : > "$P"
  for model in cbramod reve luna biot moirai neurolm; do
   for strat in ft_only lp_ft; do for cond in EO EC; do
     if [ "$model" = "moirai" ]; then
       env="$COMMON export PYTHONPATH=$pp_moirai HF_HOME=/home/mat/scratch/hf_cache;"
       cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --label-csv ${LBL[$cond]} --out-dir $OUT"
     elif [ "$model" = "neurolm" ]; then
       env="$COMMON export PYTHONPATH=$pp_neurolm;"
       cmd="python $CA/finetune_tslm_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --batch 16 --label-csv ${LBL[$cond]} --out-dir $OUT"
     else
       env="$COMMON export PYTHONPATH=/home/mat/projects/coco-pipe;"
       cmd="python $CA/run_finetune_lora.py --model ${model} --condition ${cond}_baseline --strategy ${strat} --level ${level} --label-csv ${LBL[$cond]} --out-dir $OUT"
     fi
     echo "ftasm_${model}_${strat}_${level}_${cond}|$env mkdir -p $OUT; $cmd" >> "$P"
   done; done
  done
done

# subject -> CPU/rrg-kjerbi ; epoch -> GPU/def-kjerbi
sid=$($SLURM/_submit_array.sh ftasm_subject $SLURM/params/ftasm_subject.params \
      --account=rrg-kjerbi --cpus-per-task=64 --mem=48G --time=12:00:00)
echo "ftasm_subject: array $sid (24 CPU tasks) -> $OUT"
eid=$($SLURM/_submit_array.sh ftasm_epoch $SLURM/params/ftasm_epoch.params \
      --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=6 --mem=32G --time=06:00:00)
echo "ftasm_epoch: array $eid (24 GPU tasks) -> $OUT"
