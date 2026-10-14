#!/bin/bash
# Pretraining ablation, probe pathway: extract RANDOM-encoder embeddings for
# moirai + neurolm (reservoir features), then run the fixed-head probe sweep on
# them. Probe array chains afterok on the extraction array; results merge into
# the baseline embeddings_classification tree as {model}_rand.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
EMB=/home/mat/scratch/results/extraceted_embeddings_files
BASE=/home/mat/scratch/results/clean_results/embeddings_classification/epilepsy/baseline
declare -A LBL=( [EO]=/home/mat/scratch/labels_earliest_EO.csv [EC]=/home/mat/scratch/labels_earliest_EC.csv )
pp_moirai="/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:/home/mat/.local/lib/python3.11/site-packages"
pp_neurolm="/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM"
COMMON="export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;"
PENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

P=$SLURM/params/extract_rand.params; : > "$P"
for cond in EO EC; do
  echo "extract_moirai_random_encoder_${cond}|$COMMON export PYTHONPATH=$pp_moirai HF_HOME=/home/mat/scratch/hf_cache; python $CA/extract_moirai_embeddings.py --condition ${cond}_baseline --random-init --label-csv ${LBL[$cond]} --out-dir $EMB/moirai_random_encoder" >> "$P"
  echo "extract_neurolm_random_encoder_${cond}|$COMMON export PYTHONPATH=$pp_neurolm; python $CA/extract_neurolm_embeddings.py --condition ${cond}_baseline --random-init --label-csv ${LBL[$cond]} --out-dir $EMB/neurolm_random_encoder" >> "$P"
done
XID=$($SLURM/_submit_array.sh extract_rand "$P" \
      --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 --cpus-per-task=6 --mem=32G --time=06:00:00)
echo "extract_rand: array $XID (4 GPU tasks) -> $EMB/{moirai_random_encoder,neurolm_random_encoder}"

P2=$SLURM/params/cleanemb_rand.params; : > "$P2"
for level in subject epoch; do for model in moirai_random_encoder neurolm_random_encoder; do for cond in EO EC; do
  echo "cleanemb_${level}_${model}_${cond}|$PENV python $CA/probe_nested_optuna.py --model ${model} --condition ${cond} --level ${level} --fixed --cohort-group everything --label-csv ${LBL[$cond]} --out-dir $BASE/${level}" >> "$P2"
done; done; done
PID=$(DEP=afterok:$XID $SLURM/_submit_array.sh cleanemb_rand "$P2" --cpus-per-task=32 --mem=32G --time=12:00:00)
echo "cleanemb_rand: array $PID (8 tasks, afterok:$XID) -> $BASE"
