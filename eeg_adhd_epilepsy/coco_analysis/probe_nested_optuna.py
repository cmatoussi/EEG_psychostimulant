"""Nested-CV Optuna HPO of the classifier head on frozen FM embeddings.

coco-pipe does nested tuning but only grid/random (no Optuna) and its `grids` are
list-typed (can't express continuous distributions), so we implement a small
sklearn-compatible OptunaSearchCV and run the nested CV here.

Nested CV:  outer 5-fold StratifiedGroupKFold (subject-grouped, unbiased score)
            x inner k-fold Optuna (TPE) search per head, refit-best on outer-train,
            score outer-test.

Levels / aggregations (subject-level is the reported metric):
  subject : averaged_predictions (fit heads on epochs, average per-subject proba)
            + averaged_epochs   (mean-pool embeddings per subject, then classify)
  epoch   : per_epoch           (score each epoch, no pooling)

Balanced accuracy at three operating points: default(0.5) / honest(out-of-fold
threshold) / oracle(test-optimal). Output = the embedding_performance COLUMNS
schema, one CSV per (cohort-group) invocation.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score

import run_analysis as ra
import pooled_metrics
import cohort_balance
from run_embedding_cohorts import COHORT_GROUPS

EMB_FM = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
          "signal_features/eeg_foundation_embeddings/combined")
EMB_TS = "/home/mat/scratch/results/extraceted_embeddings_files"  # moirai / neurolm
N_OUTER = 5
_GRID = np.linspace(0.05, 0.95, 91)

COLUMNS = ["fm_model", "condition", "cohort_group", "cohort", "aggregation", "head", "status",
           "n_subjects", "n_windows", "adhd", "epilepsy", "autism",
           "accuracy_mean", "accuracy_std",
           "balanced_accuracy_mean", "balanced_accuracy_std",
           "balanced_accuracy_calibrated_mean", "balanced_accuracy_calibrated_std",
           "weighted_f1_mean", "weighted_f1_std", "roc_auc_mean", "roc_auc_std"]


# ------------- heads + Optuna search spaces -------------
def _base(head):
    if head == "logreg":
        # saga solver supports both l1 and l2 (tuned); needs more iters to converge on high-dim embeddings
        return make_pipeline(StandardScaler(),
                             LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced"))
    if head == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced"))
    if head == "rf":
        return make_pipeline(StandardScaler(), RandomForestClassifier(class_weight="balanced", n_jobs=1))
    if head == "histgb":
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(class_weight="balanced"))
    if head == "dummy":
        return make_pipeline(StandardScaler(), DummyClassifier(strategy="stratified", random_state=42))
    raise ValueError(head)


def _clf_key(pipe):  # the classifier step name in the pipeline
    return pipe.steps[-1][0]


# optimized-head mode: {head: {param: value}} loaded from an hp_search summary CSV.
_TUNED: dict[str, dict] = {}


def _head(head):
    """Base pipeline for `head`, with hp_search-tuned params applied if available."""
    p = _base(head)
    tp = _TUNED.get(head)
    if tp:
        clf = _clf_key(p)
        p = p.set_params(**{f"{clf}__{k}": v for k, v in tp.items()})
    return p


# space entries: (kind, *args). kind in {float, int} ; float has optional log flag.
SPACES = {
    "logreg": {"C": ("float", 1e-3, 1e2, True), "penalty": ("cat", ["l1", "l2"])},
    "svm":    {"C": ("float", 1e-2, 1e2, True), "gamma": ("float", 1e-4, 1e0, True)},
    "rf":     {"n_estimators": ("int", 100, 600), "max_depth": ("int", 3, 30),
               "min_samples_leaf": ("int", 1, 20)},
    "histgb": {"learning_rate": ("float", 1e-2, 0.5, True), "max_depth": ("int", 2, 15),
               "l2_regularization": ("float", 1e-6, 1e0, True)},
    "dummy":  {},
}


def _suggest(trial, name, spec):
    if spec[0] == "float":
        return trial.suggest_float(name, spec[1], spec[2], log=(len(spec) > 3 and spec[3]))
    if spec[0] == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if spec[0] == "cat":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(spec)


class OptunaSearchCV:
    """Minimal sklearn-style Optuna search with an inner grouped CV (maximise roc_auc)."""
    def __init__(self, base, space, n_trials, inner_folds, seed=42):
        self.base, self.space, self.n_trials, self.inner_folds, self.seed = \
            base, space, n_trials, inner_folds, seed

    def fit(self, X, y, groups):
        clf = _clf_key(self.base)
        if not self.space:                      # dummy: nothing to tune
            self.best_params_, self.best_estimator_ = {}, clone(self.base).fit(X, y)
            self.best_value_, self.trials_ = None, []
            return self
        inner = StratifiedGroupKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.seed)
        splits = [(tr, te) for tr, te in inner.split(X, y, groups) if len(np.unique(y[te])) > 1]

        def objective(trial):
            params = {f"{clf}__{n}": _suggest(trial, n, s) for n, s in self.space.items()}
            est = clone(self.base).set_params(**params)
            sc = []
            for tr, te in splits:
                e = clone(est).fit(X[tr], y[tr])
                sc.append(roc_auc_score(y[te], e.predict_proba(X[te])[:, 1]))
            return float(np.mean(sc))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params_ = {f"{clf}__{k}": v for k, v in study.best_params.items()}
        self.best_value_ = float(study.best_value)                        # best inner roc_auc
        self.trials_ = [{"params": t.params, "inner_roc": (None if t.value is None else float(t.value))}
                        for t in study.trials]
        self.best_estimator_ = clone(self.base).set_params(**self.best_params_).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


from run_hp_search import refine_1d  # noqa: E402  reuse the 1-D zoom search

_GRID_PTS = (5, 3, 3)   # lighter than the standalone search; runs once per outer fold


def _inner_cv_bal(base, params, X, y, g, folds, seed=42):
    """Inner grouped-CV pooled balanced accuracy for one hyperparameter setting."""
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fa = []
    for tr, te in skf.split(X, y, g):
        if len(np.unique(y[te])) < 2:
            continue
        est = clone(base).set_params(**params).fit(X[tr], y[tr])
        fa.append((y[te], est.predict_proba(X[te])[:, 1]))
    r = pooled_metrics.pooled(fa, calibrated=False)
    return r["balanced_accuracy"]["mean"] if r else -1.0


def grid_search_head(head, X, y, g, folds):
    """Coarse-to-fine search on the outer-train fold; returns tuned param dict."""
    base = _base(head); clf = _clf_key(base)
    if head == "logreg":
        c, _, _ = refine_1d(lambda v: _inner_cv_bal(base, {f"{clf}__C": v}, X, y, g, folds),
                            1e-3, 1e3, log_space=True, points=_GRID_PTS)
        return {f"{clf}__C": c}
    if head == "rf":
        n, _, _ = refine_1d(lambda v: _inner_cv_bal(base, {f"{clf}__n_estimators": int(v)}, X, y, g, folds),
                            50, 600, log_space=False, is_int=True, points=_GRID_PTS)
        d, _, _ = refine_1d(lambda v: _inner_cv_bal(base, {f"{clf}__n_estimators": int(n), f"{clf}__max_depth": int(v)}, X, y, g, folds),
                            3, 30, log_space=False, is_int=True, points=_GRID_PTS)
        return {f"{clf}__n_estimators": int(n), f"{clf}__max_depth": int(d)}
    if head == "histgb":
        lr, _, _ = refine_1d(lambda v: _inner_cv_bal(base, {f"{clf}__learning_rate": v}, X, y, g, folds),
                             1e-2, 0.5, log_space=True, points=_GRID_PTS)
        d, _, _ = refine_1d(lambda v: _inner_cv_bal(base, {f"{clf}__learning_rate": lr, f"{clf}__max_depth": int(v)}, X, y, g, folds),
                            2, 15, log_space=False, is_int=True, points=_GRID_PTS)
        return {f"{clf}__learning_rate": lr, f"{clf}__max_depth": int(d)}
    return {}   # dummy


# handcrafted sensor-feature matrices (separate feature sets per level)
FEATDIR_HC = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/derivatives/"
              "signal_features/descriptors/combined")


# ------------- data -------------
def _load_embeddings(model, condition, label_df, level="epoch", target="epilepsy"):
    """Return X, y, groups(study_id) merged with the (corrected) labels. `target`
    is the label column to predict (epilepsy | asm_resistant | ...). For
    model='handcrafted' loads the sensor feature matrix for the given level
    (subject file for subject-level, epoch file for epoch-level)."""
    if model == "handcrafted":
        lvl = "subject" if level == "subject" else "epoch"
        base = f"{FEATDIR_HC}/sensor_{lvl}_features"
        df = pd.read_csv(f"{base}.csv", low_memory=False)
        df = df[df["condition"] == condition].copy()
        feats = [c for c in json.load(open(f"{base}_feature_columns.json")) if c in df.columns]
        df["study_id"] = df["study_id"].astype(str)
        df = df.drop(columns=[c for c in (target,) if c in df.columns])  # take label from label_df
        lab = label_df[["study_id", target]].drop_duplicates("study_id").copy()
        lab["study_id"] = lab["study_id"].astype(str)
        df = df.merge(lab, on="study_id", how="inner")
        X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        cm = np.nanmean(X, axis=0); cm = np.where(np.isnan(cm), 0.0, cm)   # mean-impute
        X = np.where(np.isnan(X), cm, X).astype(np.float32)
        return X, df[target].astype(int).to_numpy(), df["study_id"].to_numpy()
    if model in ("moirai", "neurolm"):
        f = Path(EMB_TS) / model / f"{model}_{condition}_epoch_embeddings.csv"
        df = pd.read_csv(f); idc = "study_id"
    else:
        f = Path(EMB_FM) / f"{model}_{condition}_epoch_embeddings.csv"
        df = pd.read_csv(f); idc = "subject" if "subject" in df.columns else "study_id"
    emb = [c for c in df.columns if c.startswith("embedding_")]
    # moirai/neurolm CSVs ship stale label columns -> drop any that would collide
    # with the merged target so the label comes from the corrected label_df.
    df = df.drop(columns=[c for c in (target, "epilepsy") if c in df.columns])
    df[idc] = df[idc].astype(str)
    lab = label_df.copy(); lab["study_id"] = lab["study_id"].astype(str)
    lab = lab[["study_id", target]].drop_duplicates("study_id")
    df = df.merge(lab, left_on=idc, right_on="study_id", how="inner")
    X = df[emb].to_numpy(np.float32)
    y = df[target].astype(int).to_numpy()
    g = df[idc].astype(str).to_numpy()
    return X, y, g


def _avg_subject(X, y, g):
    u = np.unique(g)
    return (np.stack([X[g == s].mean(0) for s in u]).astype(np.float32),
            np.array([int(y[g == s][0]) for s in u]), u)


# ------------- nested CV + metrics -------------
def _aggregate_by_group(y, p, g):
    u = np.unique(g)
    return (np.array([int(round(float(y[g == s].mean()))) for s in u]),
            np.array([float(p[g == s].mean()) for s in u]))


def _best_thr(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    return float(_GRID[np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in _GRID])])


def nested_eval(head, X, y, g, n_trials, inner_folds, pool_subject, tune=True, search="grid"):
    """Outer 5-fold StratifiedGroupKFold. Returns (fold_arrays, fold_configs).
    tune=True  -> nested Optuna (inner TPE search per fold, refit-best).
    tune=False -> fixed default head fit directly on outer-train (fast baseline,
                  no inner search); cfgs record 'fixed_default'."""
    outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
    fa, cfgs = [], []
    for k, (tr, te) in enumerate(outer.split(X, y, g)):
        if len(np.unique(y[te])) < 2:
            continue
        if tune and search == "grid":
            params = grid_search_head(head, X[tr], y[tr], g[tr], inner_folds)
            est = clone(_base(head)).set_params(**params).fit(X[tr], y[tr])
            cfgs.append({"outer_fold": k, "best_params": params, "trials": []})
        elif tune:
            est = OptunaSearchCV(_base(head), SPACES[head], n_trials, inner_folds).fit(X[tr], y[tr], g[tr])
            cfgs.append({"outer_fold": k, "best_params": est.best_params_,
                         "best_inner_roc": est.best_value_, "trials": est.trials_})
        else:
            est = clone(_head(head)).fit(X[tr], y[tr])
            cfgs.append({"outer_fold": k, "best_params": (_TUNED.get(head) or "fixed_default"),
                         "trials": []})
        p = np.nan_to_num(est.predict_proba(X[te])[:, 1], nan=0.5)
        yt, pt = (y[te], p)
        if pool_subject:
            yt, pt = _aggregate_by_group(y[te], p, g[te])
        if len(np.unique(yt)) > 1:
            fa.append((yt, pt))
    return fa, cfgs


def _metrics(fa):
    """POOLED scoring via the shared helper: concatenate the held-out fold
    predictions and score once (stratified-bootstrap std), instead of averaging
    tiny per-fold scores. Returns (mean, std) per metric for the CSV COLUMNS;
    None if the pooled predictions have <2 classes."""
    r = pooled_metrics.pooled(fa, calibrated=True)
    if r is None:
        return None
    g = lambda k: (r[k]["mean"], r[k].get("std", 0.0))
    return {"accuracy": (r["accuracy"]["mean"], 0.0), "balanced_accuracy": g("balanced_accuracy"),
            "balanced_accuracy_calibrated": g("balanced_accuracy_calibrated"),
            "weighted_f1": g("weighted_f1"), "roc_auc": g("roc_auc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["EO", "EC"])
    ap.add_argument("--level", required=True, choices=["subject", "epoch"])
    ap.add_argument("--cohort-group", default="all",
                    choices=list(COHORT_GROUPS) + ["all", "everything"])
    ap.add_argument("--label-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    # fast heads first so incremental writes bank them before slow RF/SVM (which,
    # for averaged_predictions/epoch, fit on 35-50k windows) risk timing out.
    ap.add_argument("--heads", nargs="+", default=["dummy", "logreg", "histgb", "rf", "svm"])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--search", default="grid", choices=["grid", "optuna"],
                    help="nested inner search: 'grid' = coarse-to-fine (matches run_hp_search), "
                         "'optuna' = TPE. Both run inside each outer fold (leak-free).")
    ap.add_argument("--subject-aggs", nargs="+", default=None,
                    choices=["averaged_epochs", "averaged_predictions"],
                    help="limit subject-level aggregations (default both). Use averaged_epochs "
                         "alone to keep nested search tractable.")
    ap.add_argument("--balanced", action="store_true",
                    help="uniform sex x age (x comorbidity) matched case/control cohort per group "
                         "(via cohort_balance.build_balanced) before the nested CV.")
    ap.add_argument("--fixed", action="store_true",
                    help="skip nested Optuna; fit default heads directly (fast baseline)")
    ap.add_argument("--tuned-params-dir", default=None,
                    help="dir of hp_search summaries; load per-head best params from "
                         "hp_search_{model}_{condition}.csv and fit those tuned heads "
                         "(implies fixed mode; output tag = 'optimized')")
    ap.add_argument("--target-col", default="epilepsy",
                    help="label column to predict (epilepsy | asm_resistant | ...)")
    ap.add_argument("--restrict-col", default=None,
                    help="keep only subjects where this column==1 before cohorting "
                         "(e.g. epilepsy for asm_resistant: resistant-vs-non-resistant WITHIN epilepsy)")
    args = ap.parse_args()

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    label_df["study_id"] = label_df["study_id"].astype(str)
    cond = f"{args.condition}_baseline"
    # optimized-head mode: load hp_search best params for this model+condition into _TUNED.
    if args.tuned_params_dir:
        f = Path(args.tuned_params_dir) / f"hp_search_{args.model}_{args.condition}.csv"
        if f.exists():
            row = pd.read_csv(f).iloc[0]
            for head in args.heads:
                col = f"{head}_best_params"
                if col in row and pd.notna(row[col]):
                    _TUNED[head] = json.loads(row[col])
            print(f"loaded tuned params from {f.name}: {_TUNED}", flush=True)
        else:
            print(f"WARNING: no tuned params at {f}; falling back to default heads", flush=True)
    fixed_mode = args.fixed or bool(args.tuned_params_dir)
    tag = "optimized" if args.tuned_params_dir else ("fixed" if args.fixed else "nested")
    Xep, yep, gep = _load_embeddings(args.model, cond, label_df, args.level, target=args.target_col)
    print(f"{args.model}/{args.condition}/{args.level}: loaded {Xep.shape} "
          f"classes={np.bincount(yep).tolist()} subj={len(np.unique(gep))}", flush=True)
    if args.restrict_col:   # e.g. asm_resistant restricted to epilepsy==1 subjects
        keep_ids = set(label_df[label_df[args.restrict_col] == 1]["study_id"])
        m = np.isin(gep, list(keep_ids))
        Xep, yep, gep = Xep[m], yep[m], gep[m]
        print(f"  restricted to {args.restrict_col}==1: {Xep.shape} "
              f"classes={np.bincount(yep).tolist()} subj={len(np.unique(gep))}", flush=True)

    # subject-level embeddings = averaged_epochs (mean-pool per subject) AND
    # averaged_predictions (classify epochs, average proba per subject) -> separate
    # avg_epochs/ + avg_predictions/ folders. handcrafted subject features are ALREADY
    # per-subject descriptors, so a single aggregation (no agg subfolder). epoch = per_epoch.
    if args.level == "subject":
        aggs = args.subject_aggs or (["averaged_epochs"] if args.model == "handcrafted"
                                     else ["averaged_epochs", "averaged_predictions"])
    else:
        aggs = ["per_epoch"]
    multi_agg = len(aggs) > 1
    AGG_DIR = {"averaged_epochs": "avg_epochs", "averaged_predictions": "avg_predictions", "per_epoch": ""}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    groups = ["all", "sex", "age", "comorbidity"] if args.cohort_group == "everything" else [args.cohort_group]
    cohorts = {}
    for gname in groups:
        if gname == "all":   # whole cohort -- not a key in run_embedding_cohorts.COHORT_GROUPS
            cohorts["all:all"] = ("all", lambda d: pd.Series(True, index=d.index))
            continue
        for ckey, spec in COHORT_GROUPS[gname].items():
            if (gname, ckey) in {("age", "0-4"), ("comorbidity", "asd"), ("sex", "ALL")}:
                continue  # skip small/redundant (matches the other sweeps)
            cohorts[f"{gname}:{ckey}"] = spec
    rows, trials = [], []

    def _flush():
        # subject: out/{avg_epochs|avg_predictions}/{cohort_group}/{cond}/results_{tag}_{model}.csv
        # epoch:   out/{cohort_group}/{cond}/results_{tag}_{model}.csv
        df = pd.DataFrame(rows, columns=COLUMNS)
        for (agg, gname), sub in df.groupby(["aggregation", "cohort_group"]):
            aggdir = AGG_DIR.get(agg, agg) if multi_agg else ""   # subfolder only when >1 agg
            d = (out / aggdir / gname / args.condition) if aggdir else (out / gname / args.condition)
            d.mkdir(parents=True, exist_ok=True)
            sub.to_csv(d / f"results_{tag}_{args.model}.csv", index=False)
        if trials and not fixed_mode:   # Optuna history only makes sense for a real search
            (out / f"trials_{tag}_{args.model}_{args.condition}.json").write_text(
                json.dumps({"model": args.model, "condition": args.condition,
                            "n_trials": args.n_trials, "inner_folds": args.inner_folds,
                            "search_spaces": SPACES, "runs": trials}, indent=1))

    for ckey, (subdir, mask_fn) in cohorts.items():
        gname = ckey.split(":", 1)[0]
        ids = {"fm_model": args.model, "condition": args.condition,
               "cohort_group": gname, "cohort": subdir}
        cohort_df = label_df[mask_fn(label_df)]
        if args.balanced:   # uniform sex x age (x comorbidity) matched case/control
            cohort_df, n_bal, drop = cohort_balance.build_balanced(cohort_df, args.target_col, gname)
            if drop:
                print(f"  cohort {ckey}: BALANCED SKIP ({drop})", flush=True); continue
        keep = set(cohort_df["study_id"])
        m = np.isin(gep, list(keep))
        Xc, yc, gc = Xep[m], yep[m], gep[m]
        if len(np.unique(yc)) < 2 or len(np.unique(gc)) < N_OUTER:
            print(f"  cohort {ckey}: SKIP (subj={len(np.unique(gc))})", flush=True); continue
        sub = label_df[label_df["study_id"].isin(set(gc))].drop_duplicates("study_id")
        counts = {"n_subjects": len(np.unique(gc)), "n_windows": int(Xc.shape[0]),
                  "adhd": int((sub.adhd == 1).sum()), "epilepsy": int((sub.epilepsy == 1).sum()),
                  "autism": int((sub.autism == 1).sum())}
        for agg in aggs:
            if agg == "averaged_epochs":
                Xa, ya, ga = _avg_subject(Xc, yc, gc); pool = False
            else:                                     # averaged_predictions / per_epoch
                Xa, ya, ga = Xc, yc, gc
                pool = (agg == "averaged_predictions")
            for head in args.heads:
                try:
                    fa, cfgs = nested_eval(head, Xa, ya, ga, args.n_trials, args.inner_folds,
                                           pool, tune=not fixed_mode, search=args.search)
                except Exception as e:  # noqa: BLE001 -- one head must never kill the sweep
                    print(f"  {ckey}/{agg}/{head}: ERROR {repr(e)[:120]}", flush=True)
                    rows.append({**ids, "aggregation": agg, "head": head, "status": "error", **counts,
                                 **{c: np.nan for c in COLUMNS if c.endswith(("_mean", "_std"))}})
                    continue
                trials.append({"cohort": ckey, "aggregation": agg, "head": head, "folds": cfgs})
                mt = _metrics(fa)
                if mt is None:                       # pooled preds had <2 classes
                    rows.append({**ids, "aggregation": agg, "head": head, "status": "degenerate",
                                 **counts,
                                 **{c: np.nan for c in COLUMNS if c.endswith(("_mean", "_std"))}})
                    continue
                rec = {**ids, "aggregation": agg, "head": head, "status": "success", **counts}
                for k, (mean, std) in mt.items():
                    rec[f"{k}_mean"] = mean; rec[f"{k}_std"] = std
                rows.append(rec)
                print(f"  {ckey}/{agg}/{head}: bal_acc={mt['balanced_accuracy'][0]} "
                      f"calibrated={mt['balanced_accuracy_calibrated'][0]}", flush=True)
                _flush()   # flush after every head
    _flush()
    print(f"--> wrote {out}/{{group}}/{args.condition}/results_{tag}_{args.model}.csv "
          f"({len(rows)} rows total)", flush=True)


if __name__ == "__main__":
    main()
