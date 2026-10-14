"""
Train a small supervised EEG net FROM SCRATCH (no pretraining, no LoRA) as a
baseline for the foundation-model comparisons: does a compact task-trained CNN
match the fine-tuned FMs on this data?

Models: eegnet (EEGNetv4) and shallow (ShallowFBCSPNet), both from braindecode.
Same data pipeline, CV folds, seeds, class weights, optimizer family
(Schedule-Free AdamW) and scoring harness (ft_cv) as the LoRA fine-tunes; the
whole network is trainable and gets a larger fixed epoch budget (no tuning).
Results land under strategy "scratch": {out}/{level}/scratch/{group}/results_{model}_{cond}.json

Usage:
    python run_train_scratch.py --model eegnet --condition EO_baseline \
        --level subject --out-dir .../clean_results/finetuning
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import (  # noqa: E402
    load_eeg_epochs, normalize_label_df, resolve_label_csv, smoke_subset)
from run_finetune_lora import _average_by_subject  # noqa: E402

MAX_EPOCHS = 50     # from-scratch nets need more epochs than the LoRA fine-tunes
BATCH = 32
LR = 1e-3


def _build(model_key, n_chans, n_times, n_out):
    from braindecode.models import EEGNetv4, ShallowFBCSPNet
    cls = {"eegnet": EEGNetv4, "shallow": ShallowFBCSPNet}[model_key]
    try:
        return cls(n_chans=n_chans, n_outputs=n_out, n_times=n_times)
    except TypeError:   # older braindecode argument names
        return cls(in_chans=n_chans, n_classes=n_out, input_window_samples=n_times,
                   final_conv_length="auto")


def _net(model, y_tr, max_epochs, device):
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

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_t = torch.as_tensor(cw, dtype=torch.float32).to(device)
    return NeuralNetClassifier(
        module=model, device=device, max_epochs=max_epochs, lr=LR, batch_size=BATCH,
        optimizer=AdamWScheduleFree, optimizer__weight_decay=0.01,
        criterion=nn.CrossEntropyLoss, criterion__weight=cw_t,
        train_split=None, callbacks=[_SFTrainMode()], verbose=0)


def main():
    global MAX_EPOCHS
    import torch
    import ft_cv

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["eegnet", "shallow"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--level", required=True, choices=["epoch", "subject"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=None,
                    help="override; default = earliest-per-condition file.")
    ap.add_argument("--smoke", action="store_true",
                    help="10-subject, 2-fold, 2-epoch crash check before submitting long jobs")
    args = ap.parse_args()
    args.label_csv = resolve_label_csv(args.condition, args.label_csv)
    n_splits = 5
    if args.smoke:
        MAX_EPOCHS, n_splits = 2, 2

    cshort = args.condition.replace("_baseline", "")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.model}/{cshort}/scratch/{args.level}] epochs={MAX_EPOCHS} lr={LR} dev={dev}", flush=True)

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base",
                         "ch_names": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                                      "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]}}
    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    if args.smoke:
        label_df = smoke_subset(label_df)
    X, y, groups = load_eeg_epochs(config, label_df)
    X = X.astype(np.float32)
    if args.level == "subject":
        X, y, groups = _average_by_subject(X, y, groups)
    print(f"  data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    def fit_predict(X_tr, y_tr, X_te, strategy, k):
        torch.manual_seed(42 + k)   # same per-fold seeds as the FT drivers
        model = _build(args.model, X_tr.shape[1], X_tr.shape[2], int(np.unique(y_tr).size))
        net = _net(model, y_tr, MAX_EPOCHS, dev)
        net.fit(X_tr, y_tr)
        if hasattr(net.optimizer_, "eval"):
            net.optimizer_.eval()   # schedule-free: eval weights for inference
        return net.predict_proba(X_te)[:, 1]

    ft_cv.run_cv(args.model, cshort, "scratch", args.level, args.out_dir,
                 args.label_csv, X, y, groups, fit_predict, n_splits=n_splits)


if __name__ == "__main__":
    main()
