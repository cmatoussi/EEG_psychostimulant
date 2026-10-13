#!/bin/bash
# Submit a sweep as ONE job array instead of one .slurm file per combo.
#   usage: _submit_array.sh <sweep_name> <params_file> <sbatch options...>
#   params file: one task per line, "tag|command" (tag names the log file).
# Env:  DRY=1  print the sbatch command instead of submitting.
#       DEP=<spec>  add --dependency=<spec> (e.g. afterany:12345 covers a whole array).
# Prints the array job id on stdout (or "DRY").
set -u
SLURM=/home/mat/scratch/results/slurm_scripts
sweep=$1; params=$2; shift 2
ntasks=$(grep -c . "$params")
[ "$ntasks" -ge 1 ] || { echo "no tasks in $params" >&2; exit 1; }
logdir=$SLURM/logs/$sweep
mkdir -p "$logdir"
cmd=(sbatch --parsable --account=def-kjerbi --job-name="$sweep" --array="0-$((ntasks - 1))"
     --output=/dev/null --export=ALL,PARAMS="$params",LOGDIR="$logdir"
     ${DEP:+--dependency=$DEP} "$@" "$SLURM/_array_runner.sh")
if [ "${DRY:-0}" = "1" ]; then
    echo "DRY ($ntasks tasks): ${cmd[*]}" >&2
    echo "DRY"
else
    "${cmd[@]}"
fi
