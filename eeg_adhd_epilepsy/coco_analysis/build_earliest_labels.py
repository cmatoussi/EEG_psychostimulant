"""Build per-condition label CSVs using the earliest-available recording.

For each patient, per condition (EO/EC), select the study_id of the EARLIEST
recording (by eeg_date) that actually HAS that condition's data; fall back to the
later recording only for a condition the earliest recording is missing. This
dedups the 91 dual-study patients (one recording per patient per condition) while
not discarding a condition that only exists on the second recording.

Writes labels_earliest_EO.csv and labels_earliest_EC.csv (full metadata rows,
one study_id per patient for that condition).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

FULL = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
EMB = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
       "signal_features/eeg_foundation_embeddings/combined")


def _avail(cond):
    """study_ids that have <cond> data (from a reference embeddings CSV)."""
    f = f"{EMB}/bendr_{cond}_baseline_epoch_embeddings.csv"
    c = pd.read_csv(f, nrows=1).columns
    ic = "subject" if "subject" in c else "study_id"
    return set(pd.read_csv(f, usecols=[ic])[ic].astype(str))


def select_for_condition(df, avail):
    """One study_id per patient: earliest (by date) recording that has the condition."""
    keep = []
    for _, grp in df.sort_values("eeg_dt").groupby("patient_id", sort=False):
        for sid in grp["study_id"].astype(str):
            if sid in avail:
                keep.append(sid)
                break
    return set(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/mat/scratch")
    args = ap.parse_args()
    df = pd.read_csv(FULL)
    df["eeg_dt"] = pd.to_datetime(df["eeg_date"], errors="coerce")
    df["study_id"] = df["study_id"].astype(str)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for cond in ("EO", "EC"):
        avail = _avail(cond)
        sel = select_for_condition(df, avail)
        sub = df[df["study_id"].isin(sel)].copy()
        p = out / f"labels_earliest_{cond}.csv"
        sub.to_csv(p, index=False)
        # sanity: exactly one study_id per patient, all have the condition
        dup = sub["patient_id"].duplicated().sum()
        print(f"{cond}: {len(sub)} recordings, {sub['patient_id'].nunique()} patients "
              f"(dupe-patient rows={dup}); all in avail={sub['study_id'].isin(avail).all()} "
              f"-> {p}", flush=True)


if __name__ == "__main__":
    main()
