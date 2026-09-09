"""Fill in the EC probe AUCs for MOIRAI/NeuroLM whose extraction jobs timed out
*after* writing the embedding CSV (only the post-hoc probe was lost). Reads the
saved CSV, runs the same logreg/rf 5-fold grouped CV epoch + subject-mean probe.
SVM is skipped (it was the slow step that caused the wall-time timeout)."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def _probe(X, y, groups, name):
    heads = {"logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
             "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for hn, clf in heads.items():
        aucs, bas = [], []
        for tr, te in sgkf.split(X, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            pipe = make_pipeline(StandardScaler(), clf)
            pipe.fit(X[tr], y[tr])
            p1 = pipe.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p1))
            bas.append(balanced_accuracy_score(y[te], (p1 >= 0.5).astype(int)))
        print(f"  [{name}/{hn}] roc_auc={np.mean(aucs):.3f} bal_acc={np.mean(bas):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    E = df[emb_cols].to_numpy(np.float32)
    y = df["epilepsy"].to_numpy(int)
    groups = df["study_id"].astype(str).to_numpy()
    print(f"{args.name}: {E.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)
    print("=== epoch-level ===", flush=True)
    _probe(E, y, groups, f"{args.name}/epoch")
    u = np.unique(groups)
    Xs = np.stack([E[groups == s].mean(0) for s in u])
    ys = np.array([int(y[groups == s][0]) for s in u])
    print("=== subject-level (mean-embedding) ===", flush=True)
    _probe(Xs, ys, u, f"{args.name}/subject")


if __name__ == "__main__":
    main()
