"""
Fine-tune an EEG FM with the Ray-Tune-selected LoRA hyperparameters, using
Schedule-Free AdamW (same optimizer the tuning used, so the tuned lr transfers),
full 5-fold StratifiedGroupKFold CV, on the ALL cohort.

Four variants (strategy x level):
  ft_only / lp_ft            - single-stage LoRA vs linear-probe -> LoRA (two phase)
  epoch   / subject          - train on all epochs vs on per-subject AVERAGED epochs
                               (raw epochs meaned per subject -> one input each;
                               note: averaging non-phase-locked EEG attenuates
                               oscillations, mirroring the embedding-averaging case).

Config (r/alpha/dropout/lr) is read from the model's optuna trials.csv (best ROC-AUC row).

Usage:
    python run_finetune_tuned.py --model reve --condition EO_baseline \
        --strategy lp_ft --level subject --out-dir .../ray_tuned/lp_ft_subject
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from coco_pipe.decoding._metrics import _balanced_accuracy_optimal_score  # noqa: E402
from run_analysis import load_eeg_epochs, normalize_label_df  # noqa: E402
from tune_lora import _prep_model_data  # noqa: E402  (model-specific preprocessing)

N_SPLITS = 5
MAX_EPOCHS = 15
LP_EPOCHS = 5
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"
BATCH = 32
_HEAD = ("final_layer", "classifier", "head")


def tuned_config(model):
    f = f"/home/mat/scratch/results/fine_tune/ray_tuned/raytune/optuna/{model}/{model}_optuna_trials.csv"
    df = pd.read_csv(f).rename(columns={
        "config/r": "r", "config/alpha": "alpha", "config/dropout": "dropout", "config/lr": "lr"})
    b = df.loc[df.roc_auc.idxmax()]
    return {"r": int(b.r), "alpha": int(b.alpha), "dropout": float(b.dropout), "lr": float(b.lr)}


def _average_by_subject(X, y, groups):
    uniq = np.unique(groups)
    Xs = np.stack([X[groups == g].mean(axis=0) for g in uniq]).astype(np.float32)
    ys = np.array([int(y[groups == g][0]) for g in uniq])
    return Xs, ys, uniq


def _sf_net(backend, y_tr, lr, max_epochs):
    import torch, torch.nn as nn
    from schedulefree import AdamWScheduleFree
    from skorch import NeuralNetClassifier
    from skorch.callbacks import Callback
    from sklearn.utils.class_weight import compute_class_weight

    class _SFTrainMode(Callback):
        # Schedule-Free requires optimizer.train() before the first step.
        def on_train_begin(self, net, **kw):
            opt = getattr(net, "optimizer_", None)
            if opt is not None and hasattr(opt, "train"):
                opt.train()

    out_dim = int(np.unique(y_tr).size)
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_t = torch.as_tensor(cw, dtype=torch.float32).to(backend._device)
    return NeuralNetClassifier(
        module=backend._get_skorch_module(), module__backend=backend, module__output_dim=out_dim,
        device=backend._device, max_epochs=max_epochs, lr=lr, batch_size=BATCH,
        optimizer=AdamWScheduleFree, optimizer__weight_decay=0.01,
        criterion=nn.CrossEntropyLoss, criterion__weight=cw_t,
        train_split=None, callbacks=[_SFTrainMode()], verbose=0)


def _fit(backend, X_tr, y_tr, cfg, strategy):
    """Fit with Schedule-Free AdamW; lp_ft does the two-phase LP->LoRA schedule."""
    out_dim = int(np.unique(y_tr).size)
    try:
        backend.reset_head(out_dim)          # once, before any phase
    except NotImplementedError:
        pass

    if strategy == "lp_ft":
        # Phase 1: freeze backbone LoRA (keep head + head-LoRA trainable), train head only.
        frozen = []
        for src in (getattr(backend, "_model", None), getattr(backend, "_backbone", None)):
            if src is not None:
                for n, p in src.named_parameters():
                    if "lora_" in n and p.requires_grad and not any(h in n for h in _HEAD):
                        p.requires_grad = False; frozen.append(p)
                break
        saved = backend._train_mode; backend._train_mode = "frozen"
        lp = _sf_net(backend, y_tr, cfg["lr"], LP_EPOCHS)
        lp.fit(X_tr, y_tr)
        if hasattr(lp.optimizer_, "eval"):
            lp.optimizer_.eval()             # bake averaged head weights before phase 2
        for p in frozen:
            p.requires_grad = True
        backend._train_mode = saved

    net = _sf_net(backend, y_tr, cfg["lr"], MAX_EPOCHS)
    net.fit(X_tr, y_tr)
    if hasattr(net.optimizer_, "eval"):
        net.optimizer_.eval()                # schedule-free: eval weights for inference
    return net


def main():
    global BATCH
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score)
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--strategy", required=True, choices=["ft_only", "lp_ft"])
    ap.add_argument("--level", required=True, choices=["epoch", "subject"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="batch size (lower for LaBraM's 3000-sample windows to avoid OOM)")
    args = ap.parse_args()
    BATCH = args.batch

    cfg = tuned_config(args.model)
    cshort = args.condition.replace("_baseline", "")
    print(f"[{args.model}/{cshort}/{args.strategy}/{args.level}] tuned config: {cfg}", flush=True)

    # data (ALL cohort)
    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base",
                         "ch_names": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                                      "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]}}
    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = load_eeg_epochs(config, label_df)
    X, y, groups, ch_names = _prep_model_data(X, y, groups, args.model, config["signal"])
    sfreq = float(config["signal"]["sfreq"])
    if args.level == "subject":
        X, y, groups = _average_by_subject(X, y, groups)
    print(f"  data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    bkw = {"lora_r": cfg["r"], "lora_alpha": cfg["alpha"], "lora_dropout": cfg["dropout"],
           "lora_target_modules": "all-linear"}
    if args.model in {"labram", "bendr"}:
        bkw["interpolate_channels"] = True

    METRIC_NAMES = ["accuracy", "balanced_accuracy", "roc_auc"]   # balanced_accuracy is main; oracle removed
    folds = {m: [] for m in METRIC_NAMES}
    fold_preds = []   # (y_te, proba1, groups_te) per fold, for the cohort breakdown
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {k}: single-class test, skip", flush=True); continue
        prepared = prepare_backend(args.model, X=X[tr], backend="auto", n_outputs=2,
                                   device="auto", train_mode="lora", sfreq=sfreq,
                                   ch_names=ch_names, backend_kwargs=bkw)
        backend = prepared.backend
        # prepared.adapt() resamples/interpolates but does NOT apply the fixed-montage
        # channel construction (e.g. BIOT's 19->16 bipolar derivation) -- that lives in
        # backend.fit()/predict_proba() which this driver bypasses. Apply it here so the
        # model sees its native channel count (fixes BIOT IndexError; no-op otherwise).
        _native = getattr(backend, "_construct_channels", lambda z: z)
        net = _fit(backend, _native(prepared.adapt(X[tr])), y[tr], cfg, args.strategy)
        Xte = _native(prepared.adapt(X[te]))
        pred = net.predict(Xte)
        proba1 = net.predict_proba(Xte)[:, 1]
        folds["accuracy"].append(float(accuracy_score(y[te], pred)))
        folds["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], pred)))
        folds["roc_auc"].append(float(roc_auc_score(y[te], proba1)))
        fold_preds.append((y[te], proba1, groups[te]))
        print(f"  fold {k}: acc={folds['accuracy'][-1]:.3f} "
              f"bacc={folds['balanced_accuracy'][-1]:.3f} auc={folds['roc_auc'][-1]:.3f}", flush=True)

    metrics = {m: {"mean": float(np.mean(v)) if v else float("nan"),
                   "std": float(np.std(v)) if v else float("nan"), "folds": v}
               for m, v in folds.items()}
    # per-cohort breakdown (all/sex/age/comorbidity) from the held-out predictions
    import cohort_ft_metrics
    cohort_ft_metrics.save_preds(args.out_dir, args.level, args.strategy, args.model, cshort, fold_preds)
    cohort_metrics = cohort_ft_metrics.compute(fold_preds, args.label_csv)
    for cname, cm in cohort_metrics.items():
        hb = cm.get("balanced_accuracy_calibrated", {}).get("mean")
        print(f"    cohort {cname}: honest_bal_acc={hb} (n={cm.get('n')})", flush=True)
    written = cohort_ft_metrics.write_split(args.out_dir, args.level, args.strategy,
                                            args.model, cshort, metrics, cohort_metrics)
    print(f"--> wrote {len(written)} group files under {args.out_dir}/{args.level}/{args.strategy}/ "
          f"(bacc={metrics['balanced_accuracy']['mean']:.3f} auc={metrics['roc_auc']['mean']:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
