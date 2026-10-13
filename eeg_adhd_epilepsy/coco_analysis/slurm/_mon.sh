#!/bin/bash
# Monitor any array sweep: task states + newest log tail.
#   usage: _mon.sh <sweep> [n_tail]      e.g. _mon.sh ftfinal 20
# <sweep> = the job-name / logs subdir used by _submit_array.sh.
set -u
SLURM=/home/mat/scratch/results/slurm_scripts
sweep=${1:?usage: _mon.sh <sweep> [n_tail]}
ntail=${2:-15}

echo "=== queue (squeue, job-name=$sweep) ==="
squeue -u "$USER" -h -n "$sweep" -o "%18i %9T %10M %R" | sort || true

echo
echo "=== recent task states (sacct, today+yesterday) ==="
sacct -u "$USER" --name="$sweep" -S "$(date -d yesterday +%F)" \
      -o JobID%-18,State%12,Elapsed%10,MaxRSS%10 -n 2>/dev/null \
  | grep -vE "\.batch|\.extern" | tail -30 || true

echo
echo "=== failures in logs/$sweep ==="
grep -ril "error\|traceback" "$SLURM/logs/$sweep" 2>/dev/null | head -10 || true

new=$(ls -t "$SLURM/logs/$sweep"/*.out 2>/dev/null | head -1)
if [ -n "${new:-}" ]; then
  echo
  echo "=== newest log: $new ==="
  tail -n "$ntail" "$new"
else
  echo "(no logs yet in logs/$sweep)"
fi
