"""Single source of truth for the demographic cohort definitions.

The masks live here once; consumers attach their own labels/filenames via
subdir_groups() (handcrafted/dimred) and embed_groups() (embedding sweep), so
existing output names stay stable while the definitions can no longer drift
between files.
"""
from __future__ import annotations
import pandas as pd


def _true(d):
    return pd.Series(True, index=d.index)


# canonical masks on the normalized label_df
MASKS = {
    "all": {"all": _true},
    "sex": {
        "F": lambda d: d.sex == "F",
        "M": lambda d: d.sex == "M",
        "ALL": _true,
    },
    "age": {  # uses the pre-binned age_group column
        "0-4": lambda d: d.age_group == "0-4",
        "5-8": lambda d: d.age_group == "5-8",
        "9-12": lambda d: d.age_group == "9-12",
        "13-18": lambda d: d.age_group == "13-18",
    },
    "comorbidity": {
        "none": lambda d: (d.autism == 0) & (d.adhd == 0),
        "asd": lambda d: (d.autism == 1) & (d.adhd == 0),
        "adhd": lambda d: (d.autism == 0) & (d.adhd == 1),
        "both": lambda d: (d.autism == 1) & (d.adhd == 1),
    },
}

# always skipped: tiny age bin, undecodable ASD comorbidity, redundant sex/ALL
SKIP = {("age", "0-4"), ("comorbidity", "asd"), ("sex", "ALL")}

# (group, key) -> output subdir used by the handcrafted / dim-reduction sweeps
SUBDIR = {
    ("all", "all"): "all",
    ("sex", "F"): "female", ("sex", "M"): "male", ("sex", "ALL"): "all",
    ("age", "0-4"): "age_0_4", ("age", "5-8"): "age_5_8",
    ("age", "9-12"): "age_9_12", ("age", "13-18"): "age_13_18",
    ("comorbidity", "none"): "none", ("comorbidity", "asd"): "asd",
    ("comorbidity", "adhd"): "adhd", ("comorbidity", "both"): "both",
}

# embedding sweep: (group, canonical key) -> (its cohort key, its csv filename)
_EMBED = [
    ("comorbidity", "none", "no_comorbidity", "results_embedding_no_comorbidity.csv"),
    ("comorbidity", "asd", "with_asd", "results_embedding_with_asd.csv"),
    ("comorbidity", "adhd", "with_adhd", "results_embedding_with_adhd.csv"),
    ("comorbidity", "both", "with_adhd_and_asd", "results_embedding_with_adhd_and_asd.csv"),
    ("sex", "F", "F", "results_embedding_female.csv"),
    ("sex", "M", "M", "results_embedding_male.csv"),
    ("sex", "ALL", "ALL", "results_embedding_all.csv"),
    ("age", "0-4", "0-4", "results_embedding_age_0_4.csv"),
    ("age", "5-8", "5-8", "results_embedding_age_5_8.csv"),
    ("age", "9-12", "9-12", "results_embedding_age_9_12.csv"),
    ("age", "13-18", "13-18", "results_embedding_age_13_18.csv"),
]


def subdir_groups():
    """{group: {key: (subdir, mask)}} for the handcrafted / dim-reduction sweeps."""
    return {g: {k: (SUBDIR[(g, k)], fn) for k, fn in ks.items()} for g, ks in MASKS.items()}


def embed_groups():
    """{group: {embed_key: (csv_name, mask)}} for the embedding sweep (no 'all' group)."""
    out = {}
    for g, k, ek, csv in _EMBED:
        out.setdefault(g, {})[ek] = (csv, MASKS[g][k])
    return out
