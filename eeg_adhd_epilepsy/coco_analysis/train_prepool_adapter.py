"""
Train a trainable TEMPORAL adapter on cached BENDR encoder feature maps
(pre-pooling), backbone frozen. Step 2 of the receptive-field-collapse test:
can a learned adapter beat the fixed mean/std/max pooling probe (0.565 EO)?

The encoder output is (n, 512 feature-channels, T'~26); electrodes are already
mixed by BENDR's first conv, so the exploitable axis is temporal. The adapter is
  input-BatchNorm -> [Conv1d + BN + GELU + Dropout] x n_layers
  -> attention-pool + mean-pool + max-pool (concat) -> Dropout -> Linear
Attention pooling lets the model learn WHICH timesteps matter (vs fixed pooling).

Runs a small config grid, 5-fold StratifiedGroupKFold (subject-grouped), reports
epoch-level and subject-level (averaged predictions) metrics, and saves the best.
Baselines (logreg / MLP on fixed mean/std/max pooling) are included for reference.

Usage:
    python train_prepool_adapter.py --cache prepool_maps_EO.npz \
        --out-json prepool_adapter_EO.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from coco_pipe.decoding._metrics import _balanced_accuracy_optimal_score  # noqa: E402

N_SPLITS = 5
MAX_EPOCHS = 40
BATCH = 256

# curated grid over the adapter design space (kept small; signal is weak)
CONFIGS = [
    {"width": 128, "kernel": 3, "dropout": 0.3, "n_layers": 1, "lr": 1e-3},
    {"width": 128, "kernel": 5, "dropout": 0.5, "n_layers": 2, "lr": 1e-3},
    {"width": 256, "kernel": 5, "dropout": 0.5, "n_layers": 2, "lr": 1e-3},
    {"width": 256, "kernel": 7, "dropout": 0.5, "n_layers": 2, "lr": 5e-4},
    {"width": 256, "kernel": 3, "dropout": 0.4, "n_layers": 2, "lr": 1e-3},
    {"width": 384, "kernel": 5, "dropout": 0.6, "n_layers": 2, "lr": 5e-4},
]


def _make_adapter(in_ch, cfg, n_out=2):
    import torch.nn as nn
    layers = [nn.BatchNorm1d(in_ch)]
    c = in_ch
    for _ in range(cfg["n_layers"]):
        layers += [nn.Conv1d(c, cfg["width"], cfg["kernel"], padding=cfg["kernel"] // 2),
                   nn.BatchNorm1d(cfg["width"]), nn.GELU(), nn.Dropout(cfg["dropout"])]
        c = cfg["width"]

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Sequential(*layers)
            self.attn = nn.Conv1d(cfg["width"], 1, 1)
            self.head = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(cfg["width"] * 3, n_out))

        def forward(self, x):
            import torch
            h = self.temporal(x)
            a = torch.softmax(self.attn(h), dim=-1)
            z = torch.cat([(h * a).sum(-1), h.mean(-1), h.amax(-1)], dim=1)
            return self.head(z)

    return Adapter()


def _metrics(yt, p1, groups=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    pred = (p1 >= 0.5).astype(int)
    out = {"accuracy": float(accuracy_score(yt, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
           "balanced_accuracy_optimal": _balanced_accuracy_optimal_score(yt, p1),
           "roc_auc": float(roc_auc_score(yt, p1))}
    if groups is not None:
        uniq = np.unique(groups)
        sy = np.array([int(round(float(yt[groups == g].mean()))) for g in uniq])
        sp = np.array([float(p1[groups == g].mean()) for g in uniq])
        spred = (sp >= 0.5).astype(int)
        out["subj_roc_auc"] = float(roc_auc_score(sy, sp)) if len(np.unique(sy)) > 1 else float("nan")
        out["subj_balanced_accuracy"] = float(balanced_accuracy_score(sy, spred))
        out["subj_balanced_accuracy_optimal"] = _balanced_accuracy_optimal_score(sy, sp) if len(np.unique(sy)) > 1 else float("nan")
    return out


def _train_fold(Xtr, ytr, Xte, cfg, dev):
    import torch, torch.nn as nn
    from schedulefree import AdamWScheduleFree
    from sklearn.utils.class_weight import compute_class_weight
    net = _make_adapter(Xtr.shape[1], cfg).to(dev)
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    opt = AdamWScheduleFree(net.parameters(), lr=cfg["lr"], weight_decay=0.01)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    n = len(Xtr_t)
    net.train(); opt.train()
    for ep in range(MAX_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr_t[idx].to(dev); yb = ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward(); opt.step()
    net.eval(); opt.eval()

    def _proba(arr):
        import torch
        out = []
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32)
            for i in range(0, len(t), BATCH):
                out.append(torch.softmax(net(t[i:i + BATCH].to(dev)), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)
    return _proba(Xtr), _proba(Xte)


def _agg_subject(y, p, g):
    """Mean probability per subject -> (subject labels, subject probs)."""
    uniq = np.unique(g)
    sy = np.array([int(round(float(y[g == u].mean()))) for u in uniq])
    sp = np.array([float(p[g == u].mean()) for u in uniq])
    return sy, sp


def _calibrated_bal_acc(y_tr, p_tr, y_te, p_te):
    """Pick the balanced-accuracy-maximizing threshold on TRAIN, apply to TEST.
    An honest, deployable operating point (no test-label peeking)."""
    from sklearn.metrics import balanced_accuracy_score
    best_t, best = 0.5, -1.0
    for t in np.unique(p_tr):
        ba = balanced_accuracy_score(y_tr, (p_tr >= t).astype(int))
        if ba > best:
            best, best_t = ba, t
    return float(balanced_accuracy_score(y_te, (p_te >= best_t).astype(int)))


def _run_cv(X, y, groups, cfg, dev):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_metrics = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        p_tr, p_te = _train_fold(X[tr], y[tr], X[te], cfg, dev)
        m = _metrics(y[te], p_te, groups[te])
        # honest calibrated balanced accuracy: threshold chosen on train, used on test
        m["balanced_accuracy_calibrated"] = _calibrated_bal_acc(y[tr], p_tr, y[te], p_te)
        sy_tr, sp_tr = _agg_subject(y[tr], p_tr, groups[tr])
        sy_te, sp_te = _agg_subject(y[te], p_te, groups[te])
        m["subj_balanced_accuracy_calibrated"] = (
            _calibrated_bal_acc(sy_tr, sp_tr, sy_te, sp_te)
            if len(np.unique(sy_te)) > 1 else float("nan"))
        fold_metrics.append(m)
    agg = {}
    for k in fold_metrics[0]:
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return agg


def _baselines(X, y, groups):
    """Fixed mean|std|max pooling + logreg / MLP, for reference (probe was 0.565)."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    Xp = np.concatenate([X.mean(-1), X.std(-1), X.max(-1)], axis=1).astype(np.float32)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    res = {}
    for name, clf in [("logreg_fixedpool", LogisticRegression(max_iter=1000, class_weight="balanced")),
                      ("mlp_fixedpool", MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, alpha=1e-3))]:
        fm = []
        for tr, te in sgkf.split(Xp, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            pipe = make_pipeline(StandardScaler(), clf)
            pipe.fit(Xp[tr], y[tr])
            p1 = pipe.predict_proba(Xp[te])[:, 1]
            fm.append(_metrics(y[te], p1, groups[te]))
        res[name] = {k: {"mean": float(np.nanmean([m[k] for m in fm])),
                         "std": float(np.nanstd([m[k] for m in fm]))} for k in fm[0]}
        print(f"[{name}] roc_auc={res[name]['roc_auc']['mean']:.3f} "
              f"subj_roc_auc={res[name]['subj_roc_auc']['mean']:.3f}", flush=True)
    return res


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--avg-epochs", action="store_true",
                    help="average each subject's cached feature maps into one input "
                         "(subject-level averaged-epochs) before CV")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X, y, groups = d["X"].astype(np.float32), d["y"].astype(int), d["groups"]
    if args.avg_epochs:
        uniq = np.unique(groups)
        X = np.stack([X[groups == g].mean(axis=0) for g in uniq]).astype(np.float32)
        y = np.array([int(y[groups == g][0]) for g in uniq])
        groups = uniq
        print(f"[avg-epochs] averaged to one map per subject -> {X.shape}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cache {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))} dev={dev}", flush=True)

    results = {"baselines": _baselines(X, y, groups), "adapter_configs": []}
    best = None
    for cfg in CONFIGS:
        agg = _run_cv(X, y, groups, cfg, dev)
        row = {"config": cfg, "metrics": agg}
        results["adapter_configs"].append(row)
        auc = agg["roc_auc"]["mean"]; sauc = agg["subj_roc_auc"]["mean"]
        print(f"[adapter {cfg}] roc_auc={auc:.3f} subj_roc_auc={sauc:.3f} "
              f"bacc_opt={agg['balanced_accuracy_optimal']['mean']:.3f}", flush=True)
        if best is None or auc > best["metrics"]["roc_auc"]["mean"]:
            best = row
    results["best"] = best
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    b = best["metrics"]
    print(f"\nBEST cfg={best['config']}\n  epoch  : roc_auc={b['roc_auc']['mean']:.3f} "
          f"bacc@0.5={b['balanced_accuracy']['mean']:.3f} "
          f"bacc_opt(oracle)={b['balanced_accuracy_optimal']['mean']:.3f} "
          f"bacc_calib(honest)={b['balanced_accuracy_calibrated']['mean']:.3f}\n  "
          f"subject: roc_auc={b['subj_roc_auc']['mean']:.3f} "
          f"bacc@0.5={b['subj_balanced_accuracy']['mean']:.3f} "
          f"bacc_opt(oracle)={b['subj_balanced_accuracy_optimal']['mean']:.3f} "
          f"bacc_calib(honest)={b['subj_balanced_accuracy_calibrated']['mean']:.3f}", flush=True)
    print(f"--> wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
