#!/bin/bash
# Generic SLURM array task shared by all sweeps. The submitting helper passes
# PARAMS (file with one "tag|command" line per task) and LOGDIR via --export.
# Each task runs its own line and logs to logs/<sweep>/<tag>_<arrayjobid>.out.
set -u
SLURM=/home/mat/scratch/results/slurm_scripts
fail() { echo "task ${SLURM_ARRAY_TASK_ID:-?}: $*" >> "$SLURM/logs/ERROR_${SLURM_JOB_ID:-unknown}.out"; exit 1; }

[ -f "${PARAMS:-}" ] || fail "params file missing: ${PARAMS:-unset}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS")
[ -n "$line" ] || fail "no line $((SLURM_ARRAY_TASK_ID + 1)) in $PARAMS"
tag=${line%%|*}
cmd=${line#*|}

mkdir -p "$LOGDIR"
exec > "$LOGDIR/${tag}_${SLURM_ARRAY_JOB_ID}.out" 2>&1
echo "== $tag  (array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, $(date)) =="
echo "$cmd"
set +u   # drop -u before activate/eval: activate scripts touch unset vars
source /home/mat/ep/bin/activate
eval "$cmd"
