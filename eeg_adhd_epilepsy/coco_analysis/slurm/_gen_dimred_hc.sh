#!/bin/bash
# Dim-reduction reports on the HANDCRAFTED sensor features (PCA/UMAP/Isomap x
# components), per cohort (all + sex + age + comorbidity), corrected
# earliest-per-condition labels. One array (2 tasks: EO, EC), rrg-kjerbi / 64 CPU.
SLURM=/home/mat/scratch/results/slurm_scripts
CA=/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis
OUT=/home/mat/scratch/results/clean_results/dim_reduction/epilepsy/baseline/handcrafted
ENV="export PYTHONPATH=/home/mat/projects/coco-pipe OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MPLBACKEND=Agg;"

P=$SLURM/params/dimred_hc.params; : > "$P"
for cond in EO EC; do
  echo "dimred_hc_${cond}|$ENV python $CA/run_dimred_cohorts.py --cohort-group everything --condition ${cond}_baseline --target-col epilepsy --label-csv /home/mat/scratch/labels_earliest_${cond}.csv --out-dir $OUT" >> "$P"
done
id=$($SLURM/_submit_array.sh dimred_hc "$P" --account=rrg-kjerbi --cpus-per-task=64 --mem=64G --time=12:00:00)
echo "dimred_hc: array $id (2 tasks) -> $OUT"
