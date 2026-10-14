#!/bin/bash
# Dim-reduction reports on the extracted FM embeddings (PCA/UMAP/Isomap x components),
# per model, per cohort (all + sex + age + comorbidity), corrected earliest-per-condition
# labels. Pooled separation metric (patched in coco-pipe). One array (16 tasks),
# rrg-kjerbi / 64 CPU.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
OUT=/home/mat/scratch/results/clean_results/dim_reduction/epilepsy/baseline/embed_extraction
MODELS="signaljepa labram cbramod luna biot bendr reve eegpt"
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MPLBACKEND=Agg;"

P=$SLURM/params/dimred.params; : > "$P"
for model in $MODELS; do for cond in EO EC; do
  echo "dimred_${model}_${cond}|$ENV python $CA/run_dimred_embeddings.py --model ${model} --condition ${cond}_baseline --levels both --cohort-group everything --target-col epilepsy --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir $OUT" >> "$P"
done; done
id=$($SLURM/_submit_array.sh dimred "$P" --account=rrg-kjerbi --cpus-per-task=64 --mem=64G --time=12:00:00)
echo "dimred: array $id ($(grep -c . "$P") tasks) -> $OUT"
