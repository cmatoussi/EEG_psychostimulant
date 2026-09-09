"""Shared balanced-cohort construction, reused by every analysis script.

Uniform quota matching: within a cohort, group subjects into cells by every
demographic axis NOT already fixed by the cohort filter itself (sex x
age_group for 'all'/'comorbidity' cohorts, age_group only for 'sex' cohorts,
sex only for 'age' cohorts). Find the smallest min(control,case) count across
all cells, use that as a uniform per-cell-per-class quota, and sample exactly
that many controls and that many cases from every cell. Result: control and
case end up with IDENTICAL sex/age (/comorbidity) composition, not just equal
totals. A cohort is dropped if the matched total falls under MIN_N=30, or if
any required cell has zero overlap (quota collapses to 0).

No data leakage: this filters label_df (subject-level metadata) BEFORE any
embedding/feature/epoch loading and before any train/test split. It only uses
demographic columns (sex, age_group, comorbidity flags) already used to define
the cohort itself -- never the target label's downstream features -- and
GroupKFold (already used everywhere in this codebase) still prevents a
subject's rows from spanning train and test after this filter is applied.

age_group == '0-4' is always excluded first (established separately as too
small a bin to use anywhere in this project).
"""
import numpy as np
import pandas as pd

MIN_N = 30
SEED = 42


def strata_for(cohort_group):
    """Demographic axes to jointly balance, given which axis the cohort itself
    already fixes."""
    if cohort_group == "sex":
        return ["age_group"]
    if cohort_group == "age":
        return ["sex"]
    return ["sex", "age_group"]  # 'all', 'comorbidity'


def build_balanced(df, target_col, cohort_group, seed=SEED):
    """Returns (balanced_df, n_total, drop_reason_or_None)."""
    df = df[df["age_group"] != "0-4"].copy()
    strata_cols = strata_for(cohort_group)
    cells = {}
    for key, cell in df.groupby(strata_cols, dropna=False):
        n0 = int((cell[target_col] == 0).sum())
        n1 = int((cell[target_col] == 1).sum())
        cells[key] = (cell, n0, n1)
    if not cells:
        return df.iloc[0:0], 0, "no data in cohort"
    quota = min(min(n0, n1) for _, n0, n1 in cells.values())
    if quota == 0:
        bottleneck = min(cells.items(), key=lambda kv: min(kv[1][1], kv[1][2]))
        return df.iloc[0:0], 0, f"zero overlap in stratum {bottleneck[0]} (ctrl={bottleneck[1][1]}, case={bottleneck[1][2]})"
    keep_idx = []
    for cell, n0, n1 in cells.values():
        c0 = cell[cell[target_col] == 0]
        c1 = cell[cell[target_col] == 1]
        keep_idx += list(c0.sample(quota, random_state=seed).index)
        keep_idx += list(c1.sample(quota, random_state=seed).index)
    balanced = df.loc[keep_idx].copy()
    if len(balanced) < MIN_N:
        return balanced, len(balanced), f"matched total {len(balanced)} < {MIN_N}"
    return balanced, len(balanced), None
