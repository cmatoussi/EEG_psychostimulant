"""Build the asm_resistant finetune label CSVs from the earliest-recording labels.

labels_asm_{EO,EC}.csv = labels_earliest_{cond}.csv restricted to treated epilepsy
with a known resistance label (restrict_for_asm_resistant), with the `epilepsy`
column overwritten by `asm_resistant` -- the finetune driver decodes the `epilepsy`
column, so the swap makes asm_resistant the target without touching the driver.
One recording per patient is inherited from the earliest files, so subject-grouped
CV stays patient-level.

Usage:
    python build_asm_labels.py [--out-dir /home/mat/scratch]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/mat/scratch")
    args = ap.parse_args()
    for cond in ["EO", "EC"]:
        d = ra.normalize_label_df(pd.read_csv(ra.resolve_label_csv(cond)))
        d = ra.restrict_for_asm_resistant(d)
        d["epilepsy"] = pd.to_numeric(d["asm_resistant"], errors="coerce").astype(int)
        out = Path(args.out_dir) / f"labels_asm_{cond}.csv"
        d.to_csv(out, index=False)
        print(f"{cond}: {len(d)} patients -> {out} "
              f"(resistant={int((d.epilepsy == 1).sum())}, responsive={int((d.epilepsy == 0).sum())})",
              flush=True)


if __name__ == "__main__":
    main()
