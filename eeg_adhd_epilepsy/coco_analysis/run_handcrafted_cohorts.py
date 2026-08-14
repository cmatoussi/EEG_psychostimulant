"""
Handcrafted-feature epilepsy decoding (sensor-level), 'all' cohort.

For each condition (EO, EC): decode Epilepsy from the SENSOR descriptor features
(19 sensors x 152 descriptors) three ways:
  1. all    - multivariate on the full feature set
  2. sensor - one model per channel -> per-sensor AUC -> MNE topomap
  3. sfs    - forward feature selection (LogReg) on the full set
Models: LogReg (L2, L1), RandomForest, Dummy. CV: StratifiedGroupKFold(5),
subject-grouped. Writes summary CSVs, feature-importance CSVs, and a
self-contained HTML report (offline-viewable) per condition.

Usage:
    python run_handcrafted_cohorts.py --level subject --label-csv <csv> --out-dir <dir>
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import SequentialFeatureSelector  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402
import pooled_metrics  # noqa: E402

FEATDIR = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
           "signal_features/descriptors/combined")
LABEL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"
CONDITIONS = ["EO_baseline", "EC_baseline"]
N_SPLITS = 5
SFS_N = 20
N_JOBS = 4  # bounded parallelism (unbounded -1 broke the process pool on shared nodes)
# nested SFS refits the wrapper 5x (once per outer fold); at epoch level (30-50k rows)
# that is intractable, so the SFS step alone runs on a subject-grouped subsample of
# at most this many rows. RF / per-sensor / full-model still use ALL rows.
SFS_MAX_ROWS = 8000
MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

MODELS = {
    "logreg_l2": lambda: make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=1000, class_weight="balanced")),
    "logreg_l1": lambda: make_pipeline(StandardScaler(), LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=1000, class_weight="balanced")),
    "rf": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=42, n_jobs=N_JOBS),
    "dummy": lambda: DummyClassifier(strategy="stratified", random_state=42),
}

# Cohorts to always skip (too small / undecodable): tiny age bin and ASD comorbidity.
SKIP_COHORTS = {("age", "0-4"), ("comorbidity", "asd"), ("sex", "ALL")}

# Per-drug anti-seizure-medication flag columns (used to derive monotherapy).
ASM_COLS = ["LEV", "LTG", "LCS", "CLB", "CBZ", "VPA", "ETH", "TPM", "RUF",
            "BRV", "STP", "OXZ", "CBM"]
# Canonical metadata (has source_dataset) — merged in only to source-filter the
# drug cohorts; source is NEVER a feature in the sensor decoding, so this is safe.
CANONICAL_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"

# Binary label column to decode (set from --target-col in main); "epilepsy" by
# default. For asm_resistant the positive class is asm_resistant==1, control==0.
_TARGET_COL = "epilepsy"
# Directory for the HTML reports; None -> alongside the result CSVs (out_dir).
_REPORTS_DIR = None


def add_drug_flags(d):
    """Attach `_n_asm` (count of distinct ASMs) and ensure a `source_dataset`
    column (merged from the canonical CSV when the label CSV dropped it), so the
    drug cohorts can be defined and kept within a single source."""
    d = d.copy()
    present = [c for c in ASM_COLS if c in d.columns]
    d["_n_asm"] = d[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    for c in ["asm", "LEV", "VPA", "asm_resistant"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    if "source_dataset" not in d.columns:
        src = pd.read_csv(CANONICAL_CSV, usecols=["study_id", "source_dataset"])
        src["study_id"] = src["study_id"].astype(str)
        d["study_id"] = d["study_id"].astype(str)
        d = d.merge(src.rename(columns={"study_id": "study_id"}), on="study_id", how="left")
    return d


def _ctrl(d):
    return d.epilepsy == 0


def _epi(d):
    return d.epilepsy == 1


def _adhd(d):
    # keep positives within the adhd study so "controls vs drug-X" is not a
    # controls-vs-other-study (source) contrast in disguise.
    return d.source_dataset == "adhd"


# cohort-group -> {cohort key: (subdir label, mask fn on the normalized label_df)}.
# Demographic groups subset the sample and decode epilepsy-vs-control within it.
# The `drug` group keeps the SHARED controls and swaps only the epilepsy+ positive
# subset, so per-medication feature importance is directly comparable.
COHORT_GROUPS = {
    "all": {"all": ("all", lambda d: pd.Series(True, index=d.index))},
    "sex": {
        "F":   ("female", lambda d: d.sex == "F"),
        "M":   ("male",   lambda d: d.sex == "M"),
        "ALL": ("all",    lambda d: pd.Series(True, index=d.index)),
    },
    "age": {
        "0-4":   ("age_0_4",   lambda d: d.age_group == "0-4"),
        "5-8":   ("age_5_8",   lambda d: d.age_group == "5-8"),
        "9-12":  ("age_9_12",  lambda d: d.age_group == "9-12"),
        "13-18": ("age_13_18", lambda d: d.age_group == "13-18"),
    },
    "comorbidity": {
        "none": ("none", lambda d: (d.autism == 0) & (d.adhd == 0)),
        "asd":  ("asd",  lambda d: (d.autism == 1) & (d.adhd == 0)),
        "adhd": ("adhd", lambda d: (d.autism == 0) & (d.adhd == 1)),
        "both": ("both", lambda d: (d.autism == 1) & (d.adhd == 1)),
    },
    # controls (epilepsy=0) vs each epilepsy+ medication subset. Non-resistant
    # positives are kept within the adhd source (controls are all adhd); the
    # `resistant` cohort deliberately spans both sources (per request), so it
    # carries the study/source confound and its AUC is inflated accordingly.
    "drug": {
        "none":      ("none",      lambda d: _ctrl(d) | (_epi(d) & _adhd(d) & (d.asm == 0))),
        "LEV_only":  ("LEV_only",  lambda d: _ctrl(d) | (_epi(d) & _adhd(d) & (d.LEV == 1) & (d._n_asm == 1))),
        "VPA_only":  ("VPA_only",  lambda d: _ctrl(d) | (_epi(d) & _adhd(d) & (d.VPA == 1) & (d._n_asm == 1))),
        "ASM_other": ("ASM_other", lambda d: _ctrl(d) | (_epi(d) & _adhd(d) & (d.asm == 1)
                                                         & ~((d.LEV == 1) & (d._n_asm == 1))
                                                         & ~((d.VPA == 1) & (d._n_asm == 1)))),
        "ASM_any":   ("ASM_any",   lambda d: _ctrl(d) | (_epi(d) & _adhd(d) & (d.asm == 1))),
        "resistant": ("resistant", lambda d: _ctrl(d) | (_epi(d) & (d.asm_resistant == 1))),
    },
}


def _sensor_of(col):
    m = re.search(r"_ch-([A-Za-z0-9]+)$", col)
    return m.group(1) if m else None


def _family_of(col):
    return re.sub(r"_ch-[A-Za-z0-9]+$", "", col)


def load_sensor_data(level, condition, cohort_df):
    path = f"{FEATDIR}/sensor_{level}_features.csv"
    df = pd.read_csv(path, low_memory=False)
    df["study_id"] = df["study_id"].astype(str)
    if "condition" in df.columns:
        df = df[df["condition"] == condition]
    feats = json.load(open(f"{FEATDIR}/sensor_{level}_features_feature_columns.json"))
    feats = [c for c in feats if c in df.columns]
    lab = cohort_df.copy()
    lab["study_id"] = lab["study_id"].astype(str)
    # rename the label to avoid clashing with the feature CSV's own target
    # column (a plain merge would suffix both to <col>_x / <col>_y).
    lab = (lab[["study_id", _TARGET_COL]].drop_duplicates("study_id")
           .rename(columns={_TARGET_COL: "_label"}))
    df = df.merge(lab, on="study_id", how="inner")
    y = df["_label"].astype(int).values
    groups = df["study_id"].values
    X = df[feats].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    if np.isnan(X).any():
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isnan(cm), 0.0, cm)
        X = np.where(np.isnan(X), cm, X)
    sensors = sorted({_sensor_of(c) for c in feats if _sensor_of(c)})
    return X, y, groups, feats, sensors


def cv_auc(clf_fn, X, y, g):
    """Collect every fold's held-out predictions, then score them POOLED (shared
    helper: concat + score once, stratified-bootstrap std) instead of averaging
    per-fold. No single-class-fold dropping — degenerate folds still contribute
    their predictions to the pool."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fa = []
    for tr, te in sgkf.split(X, y, g):
        clf = clf_fn(); clf.fit(X[tr], y[tr])
        fa.append((y[te], clf.predict_proba(X[te])[:, 1]))
    r = pooled_metrics.pooled(fa, calibrated=False)
    nan = float("nan")
    if r is None:
        return {"roc_auc_mean": nan, "roc_auc_std": nan, "balanced_accuracy_mean": nan,
                "balanced_accuracy_std": nan, "weighted_f1_mean": nan, "weighted_f1_std": nan,
                "n_folds": 0}
    return {"roc_auc_mean": r["roc_auc"]["mean"], "roc_auc_std": r["roc_auc"]["std"],
            "balanced_accuracy_mean": r["balanced_accuracy"]["mean"],
            "balanced_accuracy_std": r["balanced_accuracy"]["std"],
            "weighted_f1_mean": r["weighted_f1"]["mean"], "weighted_f1_std": r["weighted_f1"]["std"],
            "n_folds": r["n_folds"]}


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _minmax_vlim(vals):
    """Scale from data min to max, floored/ceiled to 1 decimal for clean bounds."""
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if v.size == 0:
        return (0.4, 0.7)
    lo = np.floor(v.min() * 10) / 10
    hi = np.ceil(v.max() * 10) / 10
    if lo == hi:
        hi = lo + 0.1
    return (float(lo), float(hi))


def topomap_png(sensor_auc, sensor_bacc, sensors):
    try:
        import mne
        names = [MODERN.get(s, s) for s in sensors]
        info = mne.create_info(names, sfreq=200.0, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         on_missing="ignore")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
        panels = [
            (axes[0], sensor_auc, "Per-sensor decoding AUC (topomap)", "ROC-AUC"),
            (axes[1], sensor_bacc,
             "Per-sensor decoding balanced accuracy (topomap)", "balanced accuracy"),
        ]
        for ax, d, title, cbar in panels:
            vals = np.array([d.get(s, np.nan) for s in sensors])
            im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                         cmap="RdBu_r", vlim=_minmax_vlim(vals),
                                         contours=4)
            fig.colorbar(im, ax=ax, shrink=0.7, label=cbar)
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return _fig_b64(fig)
    except Exception as e:  # noqa: BLE001
        return f"<p>topomap failed: {e}</p>"


def _feature_importance(X, y, feats, perm_top=80):
    """Rank all features by RF Gini (instant) + L1 |coef|; run the more honest
    (but expensive) permutation importance only on the top-`perm_top` Gini
    features so this stays fast even on epoch-level matrices (~36k rows)."""
    Xs = StandardScaler().fit_transform(X)
    l1 = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000,
                            class_weight="balanced").fit(Xs, y)
    coef = np.abs(l1.coef_.ravel())
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=N_JOBS).fit(X, y)
    gini = rf.feature_importances_
    perm = np.full(len(feats), np.nan)
    top = np.argsort(gini)[::-1][:min(perm_top, len(feats))]
    rf_top = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=42, n_jobs=N_JOBS).fit(X[:, top], y)
    perm[top] = permutation_importance(rf_top, X[:, top], y, n_repeats=3,
                                       random_state=42, scoring="roc_auc",
                                       n_jobs=N_JOBS).importances_mean
    return pd.DataFrame({
        "feature": feats, "sensor": [_sensor_of(c) for c in feats],
        "family": [_family_of(c) for c in feats],
        "logreg_l1_abscoef": coef, "rf_gini_importance": gini,
        "rf_perm_importance": perm,
    }).sort_values("rf_gini_importance", ascending=False)


def _importance_png(imp, top=25):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(t["feature"], t["rf_gini_importance"], color="#4C78A8")
    ax.set_xlabel("RF Gini importance")
    ax.set_title(f"Top {top} features")
    return _fig_b64(fig)


def _sensor_feature_heatmap(imp, top_fam=20):
    top = imp.groupby("family")["rf_gini_importance"].mean().nlargest(top_fam).index
    piv = (imp[imp.family.isin(top)]
           .pivot_table(index="family", columns="sensor",
                        values="rf_gini_importance", aggfunc="mean"))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean RF importance")
    ax.set_title("Feature-family x sensor importance")
    return _fig_b64(fig)


def run_condition(condition, cohort_df, level, out_dir, label="all"):
    cshort = condition.replace("_baseline", "")
    X, y, g, feats, sensors = load_sensor_data(level, condition, cohort_df)
    n_sub = len(np.unique(g))
    if X.shape[0] < 20 or len(np.unique(y)) < 2 or n_sub < N_SPLITS:
        print(f"  {cshort}: SKIP (n={X.shape[0]}, subj={n_sub})", flush=True); return
    print(f"  {cshort}: {X.shape[0]} rows, {n_sub} subj, {X.shape[1]} feats, "
          f"epi={np.bincount(y).tolist()}", flush=True)

    rows, images = [], {}
    # 1) ALL
    for mname, mfn in MODELS.items():
        rows.append({"analysis": "all", "model": mname, "condition": cshort, **cv_auc(mfn, X, y, g)})
    imp = _feature_importance(X, y, feats)
    imp.to_csv(out_dir / f"feature_importance_{cshort}.csv", index=False)
    images["feature_importance"] = _importance_png(imp)
    images["sensor_feature_heatmap"] = _sensor_feature_heatmap(imp)

    # 2) SENSOR -> topomap
    sensor_auc, sensor_bacc = {}, {}
    for s in sensors:
        cols = [i for i, c in enumerate(feats) if _sensor_of(c) == s]
        res = cv_auc(MODELS["logreg_l2"], X[:, cols], y, g)
        sensor_auc[s] = res["roc_auc_mean"]
        sensor_bacc[s] = res["balanced_accuracy_mean"]
        rows.append({"analysis": "sensor", "model": "logreg_l2", "unit": s,
                     "condition": cshort, **res})
    images["topomap"] = topomap_png(sensor_auc, sensor_bacc, sensors)

    # 3) SFS  (filter->wrapper: ANOVA-F top-200, then forward SFS on those).
    #    NESTED: the whole selection pipeline is refit inside each outer CV fold
    #    (via cv_auc's clf_fn), so the prefilter+SFS never see the test fold ->
    #    honest score. The old in-sample variant (select on all data, then CV on
    #    the same folds) is also reported as `sfs_insample` to show the inflation.
    try:
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import Pipeline
        kbest = min(200, X.shape[1])
        nsel = min(SFS_N, kbest - 1)

        # subject-grouped subsample for the (expensive, 5x-refit) SFS step only
        Xf, yf, gf = X, y, g
        if X.shape[0] > SFS_MAX_ROWS:
            rng = np.random.RandomState(42)
            uniq = rng.permutation(np.unique(g))
            keep, tot = [], 0
            for s in uniq:                       # add whole subjects until the cap
                idx = np.where(g == s)[0]
                keep.append(idx); tot += len(idx)
                if tot >= SFS_MAX_ROWS:
                    break
            sel_rows = np.concatenate(keep)
            Xf, yf, gf = X[sel_rows], y[sel_rows], g[sel_rows]
            print(f"    sfs: subsampled {Xf.shape[0]}/{X.shape[0]} rows "
                  f"({len(np.unique(gf))} subj) for selection", flush=True)

        def _sfs_pipe():
            return Pipeline([
                ("scale", StandardScaler()),
                ("prefilter", SelectKBest(f_classif, k=kbest)),
                ("sfs", SequentialFeatureSelector(
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                    n_features_to_select=nsel, direction="forward",
                    scoring="roc_auc", cv=3, n_jobs=N_JOBS)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])

        res = cv_auc(_sfs_pipe, Xf, yf, gf)       # <- honest nested-CV score (subsampled)
        rows.append({"analysis": "sfs", "model": "logreg_l2", "condition": cshort,
                     "n_selected": nsel, "prefilter_k": kbest, "unit": Xf.shape[0], **res})

        # descriptive only (NOT scored): which features SFS keeps when fit on the
        # (subsampled) data, for the report / interpretation.
        full = _sfs_pipe().fit(Xf, yf)
        cand = np.where(full.named_steps["prefilter"].get_support())[0]
        sel = [feats[i] for i in cand[full.named_steps["sfs"].get_support()]]
        json.dump(sel, open(out_dir / f"sfs_selected_{cshort}.json", "w"))
        print(f"    sfs nested bacc={res['balanced_accuracy_mean']:.3f}; picks {sel[:5]}...", flush=True)

        # subject level only: tuned-L1 embedded selection (sweep C via inner CV),
        # nested, NO subsample. L1 chooses the feature count adaptively -> a better
        # selection model than a hard-coded 20 when features >> samples. Rows=subjects
        # at subject level, so the inner C-tuning CV needs no grouping. (epoch keeps
        # the forward-SFS above.)
        if level == "subject":
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import LogisticRegressionCV

            def _l1_pipe():
                return Pipeline([
                    ("scale", StandardScaler()),
                    ("select", SelectFromModel(
                        LogisticRegressionCV(penalty="l1", solver="liblinear",
                            Cs=np.logspace(-3, 2, 12), cv=3, scoring="roc_auc",
                            class_weight="balanced", max_iter=2000),
                        threshold=1e-8)),               # keep every non-zero L1 coef
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ])

            res_l1 = cv_auc(_l1_pipe, X, y, g)          # honest nested; inner CV tunes C
            full_l1 = _l1_pipe().fit(X, y)
            sel_l1 = [feats[i] for i in np.where(full_l1.named_steps["select"].get_support())[0]]
            rows.append({"analysis": "l1_select", "model": "logreg_l1->l2", "condition": cshort,
                         "n_selected": len(sel_l1), "prefilter_k": X.shape[1],
                         "unit": X.shape[0], **res_l1})
            json.dump(sel_l1, open(out_dir / f"l1_selected_{cshort}.json", "w"))
            print(f"    l1_select nested bacc={res_l1['balanced_accuracy_mean']:.3f}; "
                  f"kept {len(sel_l1)} feats {sel_l1[:5]}...", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"    sfs error: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"summary_{cshort}.csv", index=False)
    rpt_dir = _REPORTS_DIR if _REPORTS_DIR is not None else out_dir
    rpt_dir.mkdir(parents=True, exist_ok=True)
    rpt = rpt_dir / f"report_{label}_{cshort}.html"
    _write_report(rpt, cshort, level, df, images, label)
    print(f"  {cshort}: report -> {rpt}", flush=True)


def _write_report(path, cond, level, df, images, label="all"):
    def img(k):
        v = images.get(k, "")
        return f'<img src="data:image/png;base64,{v}">' if v and not v.startswith("<") else v
    tbl = df.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:20px}}
img{{max-width:100%;border:1px solid #ddd;margin:8px 0}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:3px 6px}}</style></head><body>
<h1>Handcrafted decoding — {label} / {cond} / {level}-level</h1>
<h2>Scores (all / per-sensor / SFS)</h2>{tbl}
<h2>Per-sensor decoding (topomaps)</h2>{img('topomap')}
<h2>Top features</h2>{img('feature_importance')}
<h2>Feature-family × sensor importance</h2>{img('sensor_feature_heatmap')}
</body></html>"""
    Path(path).write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["epoch", "subject"])
    ap.add_argument("--label-csv", default=LABEL_CSV)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cohort-group", default="all", choices=list(COHORT_GROUPS))
    ap.add_argument("--cohort", default=None,
                    help="single cohort key within the group; omit for all in the group.")
    ap.add_argument("--target-col", default="epilepsy",
                    help="binary label column to decode (e.g. epilepsy, asm_resistant). "
                         "positive class == 1, control == 0.")
    ap.add_argument("--reports-dir", default=None,
                    help="write HTML reports here instead of alongside the result CSVs.")
    ap.add_argument("--condition", default=None, choices=["EO_baseline", "EC_baseline"],
                    help="run only this condition (default: both). Use with the matching "
                         "per-condition label CSV (labels_earliest_EO/EC).")
    ap.add_argument("--restrict-col", default=None,
                    help="keep only subjects where this column==1 before cohorting "
                         "(e.g. epilepsy for asm_resistant: resistant-vs-non-resistant WITHIN epilepsy)")
    args = ap.parse_args()
    conds = [args.condition] if args.condition else CONDITIONS

    global _TARGET_COL, _REPORTS_DIR
    _TARGET_COL = args.target_col
    _REPORTS_DIR = Path(args.reports_dir) if args.reports_dir else None

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    if args.restrict_col:                # e.g. asm_resistant restricted to epilepsy==1
        keep = pd.to_numeric(label_df[args.restrict_col], errors="coerce").fillna(0) == 1
        label_df = label_df[keep].copy()
        print(f"restricted to {args.restrict_col}==1: {label_df['study_id'].nunique()} subjects", flush=True)
    if args.cohort_group == "drug":
        label_df = add_drug_flags(label_df)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cohorts = COHORT_GROUPS[args.cohort_group]
    keys = [args.cohort] if args.cohort else list(cohorts)
    for key in keys:
        if key not in cohorts:
            raise SystemExit(f"Unknown cohort {key!r}; options: {list(cohorts)}")
        if (args.cohort_group, key) in SKIP_COHORTS:
            print(f"=== {args.cohort_group}/{key}: SKIP (too small / undecodable) ===", flush=True)
            continue
        subdir, mask_fn = cohorts[key]
        cohort_df = label_df[mask_fn(label_df)].copy()
        cout = out / subdir
        cout.mkdir(parents=True, exist_ok=True)
        n_epi = int((cohort_df[_TARGET_COL] == 1).sum())
        n_ctrl = int((cohort_df[_TARGET_COL] == 0).sum())
        print(f"=== {args.cohort_group}/{key} -> {subdir} "
              f"({cohort_df['study_id'].nunique()} subj: {n_epi} {_TARGET_COL}+, {n_ctrl} ctrl), "
              f"{args.level}-level ===", flush=True)
        for cond in conds:
            run_condition(cond, cohort_df, args.level, cout, label=subdir)


if __name__ == "__main__":
    main()
