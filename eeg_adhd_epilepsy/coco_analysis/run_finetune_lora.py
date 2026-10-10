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

LoRA config (r/alpha/dropout/lr) is fixed (see LORA). The old per-model Optuna
search tuned on a split that overlapped the reported folds, so its numbers were
optimistic; a fixed default avoids that leak with no real loss (the search was
noise-level).

Usage:
    python run_finetune_lora.py --model reve --condition EO_baseline \
        --strategy lp_ft --level subject --out-dir .../ray_tuned/lp_ft_subject
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import (  # noqa: E402
    load_eeg_epochs, normalize_label_df, resolve_label_csv, _prep_model_data)

MAX_EPOCHS = 15
LP_EPOCHS = 5
BATCH = 32
_HEAD = ("final_layer", "classifier", "head")

# Fixed LoRA config for all models (what moirai/neurolm/labram already defaulted to).
LORA = {"r": 8, "alpha": 16, "dropout": 0.05, "lr": 1e-3}


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
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend
    import ft_cv

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--strategy", required=True, choices=["ft_only", "lp_ft"])
    ap.add_argument("--level", required=True, choices=["epoch", "subject"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=None,
                    help="override; default = earliest-per-condition file.")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="batch size (lower for LaBraM's 3000-sample windows to avoid OOM)")
    args = ap.parse_args()
    args.label_csv = resolve_label_csv(args.condition, args.label_csv)
    BATCH = args.batch

    cfg = dict(LORA)
    cshort = args.condition.replace("_baseline", "")
    print(f"[{args.model}/{cshort}/{args.strategy}/{args.level}] LoRA config: {cfg}", flush=True)

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

    def fit_predict(X_tr, y_tr, X_te, strategy, k):
        prepared = prepare_backend(args.model, X=X_tr, backend="auto", n_outputs=2,
                                   device="auto", train_mode="lora", sfreq=sfreq,
                                   ch_names=ch_names, backend_kwargs=bkw)
        backend = prepared.backend
        # prepared.adapt() resamples/interpolates but does NOT apply the fixed-montage
        # channel construction (e.g. BIOT's 19->16 bipolar derivation) -- that lives in
        # backend.fit()/predict_proba() which this driver bypasses. Apply it here so the
        # model sees its native channel count (fixes BIOT IndexError; no-op otherwise).
        _native = getattr(backend, "_construct_channels", lambda z: z)
        net = _fit(backend, _native(prepared.adapt(X_tr)), y_tr, cfg, strategy)
        return net.predict_proba(_native(prepared.adapt(X_te)))[:, 1]

    ft_cv.run_cv(args.model, cshort, args.strategy, args.level, args.out_dir,
                 args.label_csv, X, y, groups, fit_predict)


if __name__ == "__main__":
    main()
