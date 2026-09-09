"""LoRA finetuning of MOIRAI / NeuroLM via a self-contained driver.

These two models are NOT in coco-pipe's backend registry, and peft/transformers
is version-broken on ComputeCanada (transformers pins huggingface-hub<1.0 but the
cluster ships 1.24), so we inject LoRA MANUALLY (a few lines, zero extra deps) and
run a plain PyTorch train loop with Schedule-Free AdamW -- mirroring the ray_tuned
tune->finetune flow used for the braindecode FMs.

Stages:
  --stage tune      Ray Tune Optuna over LoRA r/alpha/dropout/lr (one train/val
                    split, metric balanced_accuracy) ->
                    raytune/optuna/{model}/{model}_optuna_trials.csv
  --stage finetune  read the tuned config, 5-fold StratifiedGroupKFold,
                    {ft_only,lp_ft} x {epoch,subject} ->
                    ray_tuned/{strat}_{level}/results_{model}_{cond}.json

The wrapped module is  raw EEG epoch (B,19,T) -> model-specific tokenization ->
base encoder (LoRA-injected, grad flows through the adapters) -> mean-pool ->
Linear head -> logits.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402

LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"
DATA_ROOT = ("/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/BIDS/"
             "derivatives/preproc/")
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
        "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
MAX_EPOCHS = 15
LP_EPOCHS = 5
N_SPLITS = 5


# ------------------------- manual LoRA -------------------------
class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update B@A (scaled)."""
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / max(r, 1)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling


def inject_lora(model, r, alpha, dropout, skip=("head",)):
    """Replace every nn.Linear (whose qualified name avoids `skip`) with LoRALinear."""
    n = 0
    for mod_name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            qn = f"{mod_name}.{cname}"
            if isinstance(child, nn.Linear) and not any(s in qn for s in skip):
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout))
                n += 1
    return n


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def lora_params(model):
    """All LoRA A/B parameters (regardless of current requires_grad state)."""
    ps = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            ps += [m.A, m.B]
    return ps


# ------------------------- MOIRAI wrapper -------------------------
class MoiraiClassifier(nn.Module):
    PATCH = 128

    def __init__(self, r, alpha, dropout, n_out=2, model_name="Salesforce/moirai-1.0-R-small"):
        super().__init__()
        import extract_moirai_embeddings as em
        em._install_shims()
        from uni2ts.model.moirai.module import MoiraiModule
        self.em = em
        self.moirai = MoiraiModule.from_pretrained(model_name)
        for p in self.moirai.parameters():
            p.requires_grad_(False)
        self.d_model = self.moirai.d_model
        self.max_patch = max(self.moirai.patch_sizes)
        self.n_lora = inject_lora(self.moirai, r, alpha, dropout)
        self.head = nn.Linear(self.d_model, n_out)

    def reset_head(self, n_out=2):
        self.head = nn.Linear(self.d_model, n_out).to(next(self.parameters()).device)

    def forward(self, X):  # X: (B,19,T) float tensor
        from uni2ts.common.torch_util import mask_fill, packed_attention_mask
        dev = X.device
        b = self.em._pack_epochs(X.detach().cpu().numpy(), self.PATCH, self.max_patch)
        t = {k: torch.as_tensor(v, device=dev) for k, v in b.items()}
        loc, scale = self.moirai.scaler(t["target"],
                                        t["observed_mask"] * ~t["prediction_mask"].unsqueeze(-1),
                                        t["sample_id"], t["variate_id"])
        reprs = self.moirai.in_proj((t["target"] - loc) / scale, t["patch_size"])
        reprs = mask_fill(reprs, t["prediction_mask"], self.moirai.mask_encoding.weight)
        reprs = self.moirai.encoder(reprs, packed_attention_mask(t["sample_id"]),
                                    time_id=t["time_id"], var_id=t["variate_id"])
        return self.head(reprs.mean(dim=1))


# ------------------------- NeuroLM wrapper -------------------------
class NeuroLMClassifier(nn.Module):
    def __init__(self, r, alpha, dropout, n_out=2,
                 vq="/home/mat/scratch/neurolm_ckpt/checkpoints/VQ.pt"):
        super().__init__()
        sys.path.insert(0, "/home/mat/projects/NeuroLM")
        import extract_neurolm_embeddings as en
        from dataset import standard_1020
        self.en = en
        self.chan_idx = [standard_1020.index(c) for c in en.CH19_NL]
        self.enc = en._load_vq_encoder(vq, "cpu")
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.d_model = 768
        self.n_lora = inject_lora(self.enc, r, alpha, dropout)
        self.head = nn.Linear(self.d_model, n_out)

    def reset_head(self, n_out=2):
        self.head = nn.Linear(self.d_model, n_out).to(next(self.parameters()).device)

    def forward(self, X):  # X: (B,19,T)
        dev = X.device
        toks, ich, itime = self.en._epoch_tokens(X.detach().cpu().numpy(), self.chan_idx)
        xb = torch.as_tensor(toks, dtype=torch.float32, device=dev)
        ich_t = torch.as_tensor(ich, device=dev).unsqueeze(0).expand(xb.shape[0], -1)
        it_t = torch.as_tensor(itime, device=dev).unsqueeze(0).expand(xb.shape[0], -1)
        feats = self.enc.forward_features(xb, input_chans=ich_t, input_times=it_t)  # (B,768)
        return self.head(feats)


def build_model(model_key, cfg, n_out=2):
    if model_key == "moirai":
        return MoiraiClassifier(cfg["r"], cfg["alpha"], cfg["dropout"], n_out)
    if model_key == "neurolm":
        return NeuroLMClassifier(cfg["r"], cfg["alpha"], cfg["dropout"], n_out)
    raise ValueError(model_key)


# ------------------------- training -------------------------
def _sf_opt(params, lr):
    from schedulefree import AdamWScheduleFree
    return AdamWScheduleFree(params, lr=lr, weight_decay=0.01)


def train_eval(model_key, cfg, strategy, Xtr, ytr, Xte, yte, device, batch=16, max_epochs=MAX_EPOCHS):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    net = build_model(model_key, cfg).to(device)

    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))

    backbone = net.enc if model_key == "neurolm" else net.moirai

    def _phase(train_head_only, epochs, lr):
        for p in lora_params(backbone):
            p.requires_grad_(not train_head_only)   # LP: freeze LoRA; FT: unfreeze
        for p in net.head.parameters():
            p.requires_grad_(True)
        opt = _sf_opt([p for p in net.parameters() if p.requires_grad], lr)
        Xt = torch.as_tensor(Xtr, dtype=torch.float32)
        yt = torch.as_tensor(ytr, dtype=torch.long)
        net.train(); opt.train() if hasattr(opt, "train") else None
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), batch):
                idx = perm[i:i + batch]
                xb = Xt[idx].to(device); yb = yt[idx].to(device)
                opt.zero_grad()
                loss = crit(net(xb), yb)
                loss.backward(); opt.step()
        if hasattr(opt, "eval"):
            opt.eval()
        return opt

    if strategy == "lp_ft":
        _phase(True, LP_EPOCHS, cfg["lr"])          # linear probe: head only
    _phase(False, max_epochs, cfg["lr"])            # ft: LoRA (+head) unfrozen

    # inference
    net.eval()
    proba = []
    Xt = torch.as_tensor(Xte, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xb = Xt[i:i + batch].to(device)
            p = torch.softmax(net(xb), dim=1)[:, 1]
            proba.append(p.float().cpu().numpy())
    p1 = np.nan_to_num(np.concatenate(proba), nan=0.5, posinf=1.0, neginf=0.0)
    pred = (p1 >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(yte, p1)) if len(np.unique(yte)) > 1 else float("nan"),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "accuracy": float(accuracy_score(yte, pred))}, p1


def _bacc_opt(y, p):
    from sklearn.metrics import balanced_accuracy_score
    if len(np.unique(y)) < 2:
        return float("nan")
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid))


def _avg_by_subject(X, y, groups):
    u = np.unique(groups)
    Xs = np.stack([X[groups == g].mean(0) for g in u]).astype(np.float32)
    ys = np.array([int(y[groups == g][0]) for g in u])
    return Xs, ys, u


def load_data(condition, level, label_csv=LABEL_CSV):
    config = {"paths": {"data_root": DATA_ROOT},
              "signal": {"sfreq": 200.0, "conditions": [condition], "epoch_desc": "base",
                         "ch_names": CH19}}
    label_df = ra.normalize_label_df(__import__("pandas").read_csv(label_csv))
    X, y, groups = ra.load_eeg_epochs(config, label_df)
    if level == "subject":
        X, y, groups = _avg_by_subject(X, y, groups)
    return X.astype(np.float32), y.astype(int), np.asarray(groups)


def tuned_config(model):
    f = f"/home/mat/scratch/results/fine_tune/ray_tuned/raytune/optuna/{model}/{model}_optuna_trials.csv"
    import pandas as pd
    df = pd.read_csv(f)
    b = df.loc[df.balanced_accuracy.idxmax()]
    return {"r": int(b.r), "alpha": int(b.alpha), "dropout": float(b.dropout), "lr": float(b.lr)}


# ------------------------- stages -------------------------
def run_finetune(args):
    from sklearn.model_selection import StratifiedGroupKFold
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = tuned_config(args.model)
    cshort = args.condition.replace("_baseline", "")
    print(f"[{args.model}/{cshort}/{args.strategy}/{args.level}] cfg={cfg}", flush=True)
    X, y, groups = load_data(args.condition, args.level, args.label_csv)
    print(f"  data {X.shape} classes={np.bincount(y).tolist()} subj={len(np.unique(groups))}", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    metric_names = ["accuracy", "balanced_accuracy", "roc_auc"]   # balanced_accuracy is main; oracle removed
    folds = {m: [] for m in metric_names}
    fold_preds = []
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {k}: single-class test, skip", flush=True); continue
        res, p1 = train_eval(args.model, cfg, args.strategy, X[tr], y[tr], X[te], y[te],
                             dev, batch=args.batch)
        for m in metric_names:
            folds[m].append(res[m])
        fold_preds.append((y[te], p1, groups[te]))
        print(f"  fold {k}: acc={res['accuracy']:.3f} bacc={res['balanced_accuracy']:.3f} "
              f"auc={res['roc_auc']:.3f}", flush=True)
    metrics = {m: {"mean": float(np.nanmean(v)) if v else float("nan"),
                   "std": float(np.nanstd(v)) if v else float("nan"), "folds": v}
               for m, v in folds.items()}
    import cohort_ft_metrics
    cohort_ft_metrics.save_preds(args.out_dir, args.level, args.strategy, args.model, cshort, fold_preds)
    cohort_metrics = cohort_ft_metrics.compute(fold_preds, args.label_csv)
    for cname, cm in cohort_metrics.items():
        print(f"    cohort {cname}: honest_bal_acc={cm.get('balanced_accuracy_calibrated',{}).get('mean')} "
              f"(n={cm.get('n')})", flush=True)
    written = cohort_ft_metrics.write_split(args.out_dir, args.level, args.strategy,
                                            args.model, cshort, metrics, cohort_metrics)
    print(f"--> wrote {len(written)} group files under {args.out_dir}/{args.level}/{args.strategy}/ "
          f"(auc={metrics['roc_auc']['mean']:.3f})", flush=True)


def run_tune(args):
    import pandas as pd
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from sklearn.model_selection import StratifiedGroupKFold
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, groups = load_data(args.condition, "subject" if args.tune_level == "subject" else "epoch")
    # single train/val split for tuning
    tr, va = next(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(X, y, groups))
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    model_key = args.model
    batch = args.batch

    def trainable(cfg):
        res, _ = train_eval(model_key, cfg, "ft_only", Xtr, ytr, Xva, yva, dev, batch=batch,
                            max_epochs=MAX_EPOCHS)
        tune.report({"balanced_accuracy": res["balanced_accuracy"], "roc_auc": res["roc_auc"]})

    space = {"r": tune.choice([4, 8, 16, 32]), "alpha": tune.choice([8, 16, 32, 64]),
             "dropout": tune.choice([0.0, 0.05, 0.1]), "lr": tune.loguniform(1e-5, 1e-2)}
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"gpu": 1}),
        param_space=space,
        tune_config=tune.TuneConfig(metric="balanced_accuracy", mode="max",
                                    num_samples=args.num_samples,
                                    search_alg=OptunaSearch(metric="balanced_accuracy", mode="max")),
        run_config=tune.RunConfig(name=f"{model_key}_optuna",
                                  storage_path=str(Path(args.out_dir) / "ray")),
    )
    res = tuner.fit()
    rows = [{**r.config, "balanced_accuracy": r.metrics.get("balanced_accuracy"),
             "roc_auc": r.metrics.get("roc_auc")} for r in res]
    out = Path(f"/home/mat/scratch/results/fine_tune/raytune/optuna/{model_key}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).to_csv(
        out / f"{model_key}_optuna_trials.csv", index=False)
    print(f"--> wrote {out}/{model_key}_optuna_trials.csv ({len(rows)} trials)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["moirai", "neurolm"])
    ap.add_argument("--stage", required=True, choices=["tune", "finetune"])
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--strategy", default="ft_only", choices=["ft_only", "lp_ft"])
    ap.add_argument("--level", default="epoch", choices=["epoch", "subject"])
    ap.add_argument("--tune-level", default="subject", choices=["epoch", "subject"])
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--label-csv", default=LABEL_CSV)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=16)
    args = ap.parse_args()
    if args.stage == "tune":
        run_tune(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    main()
