#!/bin/bash
# Re-extract NeuroLM embeddings with the fixed VQ checkpoint loader (the old run
# silently used a random encoder: torch.compile "_orig_mod.VQ." key prefix meant
# strict=False matched nothing; the loader now strips the prefix and raises if
# encoder keys are missing). Overwrites the invalid CSVs. One GPU array, EO + EC.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
OUT=/home/mat/scratch/results/extraceted_embeddings_files/neurolm
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"

P=$SLURM/params/extract_neurolm.params; : > "$P"
for cond in EO EC; do
  echo "extract_neurolm_${cond}|$ENV python $CA/extract_neurolm_embeddings.py --condition ${cond}_baseline --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir $OUT" >> "$P"
done
id=$($SLURM/_submit_array.sh extract_neurolm "$P" \
     --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=6 --mem=32G --time=06:00:00)
echo "extract_neurolm: array $id (2 GPU tasks) -> $OUT"
