"""
Adversarial domain-invariance (DANN) on frozen FM embeddings — the "fancier"
subject/domain alignment that aligns nuisance variation WHILE jointly training
the epilepsy classifier, so (unlike crude per-subject normalisation) it is
constrained not to delete class-discriminative directions.

A small feature MLP feeds two heads: an epilepsy classifier, and a domain
discriminator behind a Gradient-Reversal Layer (GRL). Minimising domain loss
through the GRL pushes the features to be domain-invariant; the class loss keeps
them epilepsy-discriminative. lambda=0 is the plain baseline classifier.

  --domain subject : make features subject-invariant (the literal "patient A
                     looks like patient B" request; likely collapses because
                     epilepsy IS a between-subject property)
  --domain source  : make features site-invariant (confound removal; the more
                     principled nuisance to align)

Reports epoch- and subject-level ROC-AUC per lambda vs the lambda=0 baseline,
for every model, so we see if adversarial alignment improves or hits the same
wall as the crude version.

Usage:
    python test_dann_alignment.py --condition EO_baseline --domain source --out-json <p>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402

MODELS = ["reve", "cbramod", "luna", "bendr", "labram", "biot"]
N_SPLITS = 5
EPOCHS = 25
H = 128
BATCH = 256
LAMBDAS = [0.0, 0.3, 1.0]
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"
SOURCE_CSV = "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/csv/patients_metadata_clean.csv"


def _grl():
    import torch

    class _GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = float(lambd)
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            return -ctx.lambd * g, None
    return _GRL


def _make_dann(D, n_dom, dropout=0.5):
    import torch.nn as nn
    GRL = _grl()

    class DANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(D, H), nn.BatchNorm1d(H), nn.ReLU(), nn.Dropout(dropout))
            self.cls = nn.Linear(H, 2)
            self.dom = nn.Linear(H, n_dom)

        def forward(self, x, lambd=0.0):
            f = self.feat(x)
            return self.cls(f), self.dom(GRL.apply(f, lambd))

        def clsf(self, x):
            return self.cls(self.feat(x))

    return DANN()


def _fit_eval(Xtr, ytr, dtr, Xte, yte, gte, n_dom, lambd, dev):
    import torch, torch.nn as nn
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import roc_auc_score
    net = _make_dann(Xtr.shape[1], n_dom).to(dev)
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    ce_c = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    ce_d = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    dt = torch.tensor(dtr, dtype=torch.long); n = len(Xt)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xt[idx].to(dev); yb = yt[idx].to(dev); db = dt[idx].to(dev)
            cl, dl = net(xb, lambd)
            loss = ce_c(cl, yb) + (ce_d(dl, db) if lambd > 0 else 0.0 * dl.sum())
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net.clsf(torch.tensor(Xte, dtype=torch.float32).to(dev)), 1)[:, 1].cpu().numpy()
    ep = float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")
    u = np.unique(gte)
    sy = np.array([int(round(float(yte[gte == k].mean()))) for k in u])
    sp = np.array([float(p[gte == k].mean()) for k in u])
    sj = float(roc_auc_score(sy, sp)) if len(np.unique(sy)) > 1 else float("nan")
    return ep, sj


def main():
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--domain", required=True, choices=["subject", "source"])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    src_map = {}
    if args.domain == "source":
        sdf = ra.normalize_label_df(pd.read_csv(SOURCE_CSV))
        col = "source_dataset" if "source_dataset" in sdf.columns else None
        if col is None:
            raise SystemExit("no source_dataset column for --domain source")
        src_map = dict(zip(sdf["study_id"].astype(str), sdf[col].astype("category").cat.codes))

    results = {}
    print(f"domain={args.domain}  {'model':9s} | " + " ".join(f"L={l}" for l in LAMBDAS), flush=True)
    for model in MODELS:
        acfg = {"model_key": model, "target_col": "epilepsy", "embedding_level": "epoch"}
        try:
            X, y, groups = ra.load_precomputed_embeddings(acfg, {"paths": {}}, label_df, {"conditions": [args.condition]})
        except Exception as e:
            print(f"{model:9s} MISSING: {e}", flush=True); continue
        groups = np.asarray(groups).astype(str)
        results[model] = {}
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        for lambd in LAMBDAS:
            ep_folds, sj_folds = [], []
            for tr, te in sgkf.split(X, y, groups):
                if len(np.unique(y[te])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
                if args.domain == "source":
                    dtr = np.array([src_map.get(g, 0) for g in groups[tr]]); n_dom = 2
                else:  # subject: encode train subjects to 0..K-1
                    uniq = {g: i for i, g in enumerate(np.unique(groups[tr]))}
                    dtr = np.array([uniq[g] for g in groups[tr]]); n_dom = len(uniq)
                ep, sj = _fit_eval(Xtr, y[tr], dtr, Xte, y[te], groups[te], n_dom, lambd, dev)
                ep_folds.append(ep); sj_folds.append(sj)
            results[model][str(lambd)] = {"epoch_roc": float(np.nanmean(ep_folds)),
                                          "subj_roc": float(np.nanmean(sj_folds))}
        base = results[model][str(LAMBDAS[0])]["epoch_roc"]
        line = "  ".join(f"{results[model][str(l)]['epoch_roc']:.3f}" for l in LAMBDAS)
        best = max(results[model][str(l)]["epoch_roc"] for l in LAMBDAS[1:])
        verdict = "IMPROVES" if best > base + 0.005 else ("HURTS" if best < base - 0.005 else "no change")
        print(f"{model:9s} | {line}   base={base:.3f} best_adv={best:.3f} [{verdict}]", flush=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(
        {"condition": args.condition.replace("_baseline", ""), "domain": args.domain,
         "lambdas": LAMBDAS, "results": results}, indent=2))
    print(f"--> wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
