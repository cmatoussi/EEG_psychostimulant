# SLURM orchestration

Each sweep is submitted as **one SLURM job array** instead of one script per combo.

- `_submit_array.sh <sweep> <params_file> <sbatch opts...>` — counts the tasks in
  the params file, submits `--array=0-(N-1)`, prints the array job id.
  `DRY=1` prints the sbatch command instead of submitting;
  `DEP=afterany:<id>` chains the array on a previous one.
- `_array_runner.sh` — the generic task script shared by all sweeps: reads line
  `$SLURM_ARRAY_TASK_ID + 1` of the params file (`tag|command`), logs to
  `logs/<sweep>/<tag>_<arrayjobid>.out`, activates the venv, runs the command.
- `_mon.sh <sweep> [n_tail]` — queue + sacct task states, error scan of the
  sweep's logs, tail of the newest log.
- `_gen_*.sh` — one generator per sweep. Each writes `params/<sweep>.params`
  (one `tag|command` line per task; per-model env is an `export ...;` prefix in
  the command) and calls `_submit_array.sh`. Sweeps mixing resource classes
  (CPU vs GPU, different walltimes) submit one array per class.

The runtime directory (params files, logs, archives) is
`/home/mat/scratch/results/slurm_scripts/`, which symlinks back to the scripts
here. Run a sweep with `bash _gen_<sweep>.sh` (prefix `DRY=1` to preview).
Tags include model/strategy/level/condition so logs stay identifiable, e.g.
`logs/ftfinal/ftfinal_cbramod_ft_only_epoch_EO_<jobid>.out`.
