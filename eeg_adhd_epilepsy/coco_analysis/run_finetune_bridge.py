"""
Channel-bridge fine-tuning for BENDR (and other conv/special-montage FMs that
LoRA cannot adapt).

BENDR is a convolutional encoder + transformer whose first conv is rigidly tied
to its pretrained channel count, and PEFT's ``all-linear`` LoRA finds no usable
targets — so the standard ``fm_lora`` path degenerates to a linear probe. Here we
instead FREEZE the backbone and inject a lightweight, trainable
``Conv1d(C, C, kernel_size=1)`` **channel bridge** between the model's fixed
montage adapter (BENDR reorder + Deep1010 norm + SCALE -> 20 channels) and the
frozen backbone. The bridge is initialised to identity, so training starts from
the known-good fixed montage and learns a task-specific spatial remix without
touching the pretrained temporal weights (no catastrophic forgetting). Trainable
params = bridge + classification head; everything else stays frozen.

Optimizer is Schedule-Free AdamW (same as run_finetune_tuned, no schedule to
tune). 5-fold StratifiedGroupKFold CV on the ALL cohort. Subject level averages
each subject's raw epochs into one input before the montage adapter.

Usage:
    python run_finetune_bridge.py --model bendr --condition EO_baseline \
        --level subject --out-dir .../bridge/subject
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from coco_pipe.decoding._metrics import _balanced_accuracy_optimal_score  # noqa: E402
from run_analysis import (  # noqa: E402
    load_eeg_epochs, normalize_label_df, _to_modern_nomenclature,
)
from tune_lora import _prep_model_data  # noqa: E402  (model-specific preprocessing)

N_SPLITS = 5
MAX_EPOCHS = 15
BATCH = 32
LR = 1e-3
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def _average_by_subject(X, y, groups):
    uniq = np.unique(groups)
    Xs = np.stack([X[groups == g].mean(axis=0) for g in uniq]).astype(np.float32)
    ys = np.array([int(y[groups == g][0]) for g in uniq])
    return Xs, ys, uniq


def _bridge_module_cls():
    """skorch module: trainable Conv1d(C,C,1) bridge (identity init) -> frozen backbone.

    The backbone is kept in eval() while training so its frozen BatchNorm/dropout
    do not drift; only the bridge and the (trainable) classification head learn.
    """
    import torch
    import torch.nn as nn

    class _BridgeModule(nn.Module):
        def __init__(self, backend, output_dim: int, n_ch: int) -> None:
            super().__init__()
            self._backend = backend
            self._output_dim = output_dim
            self.bridge = nn.Conv1d(n_ch, n_ch, kernel_size=1, bias=False)
            with torch.no_grad():                      # identity init
                self.bridge.weight.zero_()
                for i in range(n_ch):
                    self.bridge.weight[i, i, 0] = 1.0
            self.model = backend._model

        def forward(self, X):
            X = self.bridge(X)
            out = self.model(X)
            return out["logits"] if isinstance(out, dict) else out

        def train(self, mode: bool = True):
            super().train(mode)
            if mode:
                # keep the frozen backbone in eval; head (trainable) back to train
                self._backend._set_backbone_eval()
                self.bridge.train(True)
            return self

    return _BridgeModule


def _sf_net(backend, module_cls, y_tr, n_ch, out_dim, lr, max_epochs):
    import torch, torch.nn as nn
    from schedulefree import AdamWScheduleFree
    from skorch import NeuralNetClassifier
    from skorch.callbacks import Callback
    from sklearn.utils.class_weight import compute_class_weight

    class _SFTrainMode(Callback):
        def on_train_begin(self, net, **kw):
            opt = getattr(net, "optimizer_", None)
            if opt is not None and hasattr(opt, "train"):
                opt.train()

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_t = torch.as_tensor(cw, dtype=torch.float32).to(backend._device)
    return NeuralNetClassifier(
        module=module_cls, module__backend=backend, module__output_dim=out_dim,
        module__n_ch=n_ch,
        device=backend._device, max_epochs=max_epochs, lr=lr, batch_size=BATCH,
        optimizer=AdamWScheduleFree, optimizer__weight_decay=0.01,
        criterion=nn.CrossEntropyLoss, criterion__weight=cw_t,
        train_split=None, callbacks=[_SFTrainMode()], verbose=0)


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bendr")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--level", default="subject", choices=["epoch", "subject"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    cshort = args.condition.replace("_baseline", "")
    print(f"[{args.model}/{cshort}/bridge/{args.level}] lr={LR} epochs={MAX_EPOCHS}", flush=True)

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base",
                         "ch_names": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                                      "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]}}
    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = load_eeg_epochs(config, label_df)
    X, y, groups, ch_names = _prep_model_data(X, y, groups, args.model, config["signal"])
    # BENDR's montage plan (like BIOT's) resolves channels by modern 10-20 names,
    # so map the legacy labels T3/T4/T5/T6 -> T7/T8/P7/P8 (same electrodes).
    if args.model in {"bendr", "biot"}:
        ch_names = _to_modern_nomenclature(ch_names)
    sfreq = float(config["signal"]["sfreq"])
    if args.level == "subject":
        X, y, groups = _average_by_subject(X, y, groups)
    print(f"  data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    module_cls = _bridge_module_cls()
    METRIC_NAMES = ["accuracy", "balanced_accuracy", "balanced_accuracy_optimal", "roc_auc"]
    folds = {m: [] for m in METRIC_NAMES}
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {k}: single-class test, skip", flush=True); continue
        # Frozen backbone + trainable head; the fixed montage adapter is applied
        # here (adapt = resample/cast, then the reorder+SCALE channel adapter),
        # so the bridge sees the model's native 20-channel layout.
        prepared = prepare_backend(args.model, X=X[tr], backend="auto", n_outputs=2,
                                   device="auto", train_mode="frozen", sfreq=sfreq,
                                   ch_names=ch_names, backend_kwargs={})
        backend = prepared.backend

        def _construct(arr):
            a = prepared.adapt(arr)
            adapter = getattr(backend, "_channel_adapter", None)
            return adapter(a) if adapter is not None else a

        Xtr = _construct(X[tr]); Xte = _construct(X[te])
        n_ch = int(Xtr.shape[1])
        net = _sf_net(backend, module_cls, y[tr], n_ch, 2, LR, MAX_EPOCHS)
        net.fit(Xtr, y[tr])
        if hasattr(net.optimizer_, "eval"):
            net.optimizer_.eval()
        pred = net.predict(Xte)
        proba1 = net.predict_proba(Xte)[:, 1]
        folds["accuracy"].append(float(accuracy_score(y[te], pred)))
        folds["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], pred)))
        folds["balanced_accuracy_optimal"].append(_balanced_accuracy_optimal_score(y[te], proba1))
        folds["roc_auc"].append(float(roc_auc_score(y[te], proba1)))
        print(f"  fold {k}: acc={folds['accuracy'][-1]:.3f} "
              f"bacc={folds['balanced_accuracy'][-1]:.3f} auc={folds['roc_auc'][-1]:.3f} "
              f"(n_ch={n_ch})", flush=True)

    metrics = {m: {"mean": float(np.mean(v)) if v else float("nan"),
                   "std": float(np.std(v)) if v else float("nan"), "folds": v}
               for m, v in folds.items()}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "condition": cshort, "strategy": "channel_bridge",
               "level": args.level, "config": {"lr": LR, "max_epochs": MAX_EPOCHS, "bridge": "conv1d_1x1_identity"},
               "optimizer": "schedule_free_adamw",
               "results": {args.model: {"metrics": metrics}}}
    p = out / f"results_{args.model}_{cshort}.json"
    p.write_text(json.dumps(payload, indent=2))
    print(f"--> wrote {p}  (bacc={metrics['balanced_accuracy']['mean']:.3f} "
          f"auc={metrics['roc_auc']['mean']:.3f})", flush=True)


if __name__ == "__main__":
    main()
