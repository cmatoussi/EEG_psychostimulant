"""
Partial-layer adaptation for transformer FMs (LaBraM, BIOT) — an alternative to
LoRA that mirrors why frozen extraction works: freeze the backbone, keep a
trainable head, and (for extra capacity) unfreeze only the LAST-K transformer
blocks. No PEFT injection.

We reuse train_mode="frozen" (backbone frozen + head trainable), then unfreeze
the last K transformer blocks. The backend's skorch module already puts only the
trainable modules into train() mode (via _set_backbone_eval), so the tested
forward/training path handles partial adaptation with no changes. The model's
native pooled head is the readout (the same pooling that makes extraction work);
the channel adapter (e.g. BIOT's TCP-bipolar montage) IS applied here — the LoRA
finetune driver omitted it, a likely cause of BIOT's chance-level EO result.

Optimizer: Schedule-Free AdamW. 5-fold StratifiedGroupKFold. Reports epoch- and
subject-level metrics incl. an honest calibrated balanced accuracy
(threshold chosen on train, applied to test).

Usage:
    python run_finetune_partial.py --model labram --condition EO_baseline \
        --level subject --unfreeze-k 2 --out-dir .../partial/labram
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from coco_pipe.decoding._metrics import _balanced_accuracy_optimal_score  # noqa: E402
from run_analysis import load_eeg_epochs, normalize_label_df, _to_modern_nomenclature  # noqa: E402
from tune_lora import _prep_model_data  # noqa: E402

N_SPLITS = 5
MAX_EPOCHS = 15
LP_EPOCHS = 5
BATCH = 32
LR = 5e-4
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def _average_by_subject(X, y, groups):
    uniq = np.unique(groups)
    Xs = np.stack([X[groups == g].mean(axis=0) for g in uniq]).astype(np.float32)
    ys = np.array([int(y[groups == g][0]) for g in uniq])
    return Xs, ys, uniq


def _unfreeze_last_k_blocks(model, k):
    """Unfreeze the last K blocks of the model's largest transformer ModuleList."""
    import torch.nn as nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None or k <= 0:
        return 0, None
    name, blocks = best
    n = 0
    params = []
    for blk in list(blocks)[-k:]:
        for p in blk.parameters():
            p.requires_grad = True
            n += p.numel()
            params.append(p)
    return n, f"{name}[-{k}:] ({len(blocks)} total)", params


def _sf_net(backend, y_tr, lr, max_epochs, out_dim):
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
        module=backend._get_skorch_module(), module__backend=backend, module__output_dim=out_dim,
        device=backend._device, max_epochs=max_epochs, lr=lr, batch_size=BATCH,
        optimizer=AdamWScheduleFree, optimizer__weight_decay=0.01,
        criterion=nn.CrossEntropyLoss, criterion__weight=cw_t,
        train_split=None, callbacks=[_SFTrainMode()], verbose=0)


def _calib(y_tr, p_tr, y_te, p_te):
    from sklearn.metrics import balanced_accuracy_score
    bt, bb = 0.5, -1.0
    for t in np.unique(p_tr):
        ba = balanced_accuracy_score(y_tr, (p_tr >= t).astype(int))
        if ba > bb:
            bb, bt = ba, t
    return float(balanced_accuracy_score(y_te, (p_te >= bt).astype(int)))


def _subj(y, p, g):
    u = np.unique(g)
    return (np.array([int(round(float(y[g == k].mean()))) for k in u]),
            np.array([float(p[g == k].mean()) for k in u]))


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["labram", "biot", "luna", "cbramod", "reve", "eegpt", "bendr"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--level", default="subject", choices=["epoch", "subject"])
    ap.add_argument("--unfreeze-k", type=int, default=2)
    ap.add_argument("--strategy", default="ft_only", choices=["ft_only", "lp_ft"],
                    help="lp_ft: train head only (blocks frozen) for LP_EPOCHS, then unfreeze last-K + head")
    ap.add_argument("--full", action="store_true", help="full fine-tuning: unfreeze the entire backbone")
    ap.add_argument("--lr", type=float, default=LR, help="learning rate (use a lower value, e.g. 1e-4, for --full)")
    ap.add_argument("--batch", type=int, default=32, help="batch size (lower for LaBraM's 3000-sample windows)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()
    global BATCH
    BATCH = args.batch

    cshort = args.condition.replace("_baseline", "")
    _mode_tag = "full" if args.full else f"k{args.unfreeze_k}"
    print(f"[{args.model}/{cshort}/{_mode_tag}/{args.level}] lr={args.lr}", flush=True)

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base",
                         "ch_names": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                                      "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]}}
    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = load_eeg_epochs(config, label_df)
    X, y, groups, ch_names = _prep_model_data(X, y, groups, args.model, config["signal"])
    if args.model == "bendr":  # BENDR's montage plan resolves by modern 10-20 names
        ch_names = _to_modern_nomenclature(ch_names)
    sfreq = float(config["signal"]["sfreq"])
    if args.level == "subject":
        X, y, groups = _average_by_subject(X, y, groups)
    print(f"  data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    bkw = {}
    if args.model in {"labram", "bendr"}:
        bkw["interpolate_channels"] = True

    METRICS = ["accuracy", "balanced_accuracy", "balanced_accuracy_optimal", "roc_auc",
               "balanced_accuracy_calibrated", "subj_roc_auc", "subj_balanced_accuracy_calibrated"]
    folds = {m: [] for m in METRICS}
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {k}: single-class test, skip", flush=True); continue
        mode = "full" if args.full else "frozen"
        prepared = prepare_backend(args.model, X=X[tr], backend="auto", n_outputs=2,
                                   device="auto", train_mode=mode, sfreq=sfreq,
                                   ch_names=ch_names, backend_kwargs=bkw)
        backend = prepared.backend
        if args.full:
            for p in backend._model.parameters():
                p.requires_grad = True          # full FT: entire backbone + head
            if k == 0:
                n_tr = sum(p.numel() for p in backend._model.parameters() if p.requires_grad)
                print(f"  full FT: {n_tr:,} trainable params", flush=True)
        else:
            n_unf, where, blk_params = _unfreeze_last_k_blocks(backend._model, args.unfreeze_k)
            if k == 0:
                print(f"  unfroze last {args.unfreeze_k} blocks: {n_unf:,} params @ {where}", flush=True)

        def _construct(arr):
            a = prepared.adapt(arr)
            adapter = getattr(backend, "_channel_adapter", None)
            return adapter(a) if adapter is not None else a

        Xtr, Xte = _construct(X[tr]), _construct(X[te])
        # lp_ft: phase 1 trains the head only (last-K blocks frozen), then unfreeze.
        if args.strategy == "lp_ft" and not args.full:
            for p in blk_params:
                p.requires_grad = False
            lp = _sf_net(backend, y[tr], args.lr, LP_EPOCHS, 2)
            lp.fit(Xtr, y[tr])
            if hasattr(lp.optimizer_, "eval"):
                lp.optimizer_.eval()
            for p in blk_params:
                p.requires_grad = True
        net = _sf_net(backend, y[tr], args.lr, MAX_EPOCHS, 2)
        net.fit(Xtr, y[tr])
        if hasattr(net.optimizer_, "eval"):
            net.optimizer_.eval()
        # NaN-safe: unstable finetuning (e.g. BENDR contextualizer) can diverge to
        # NaN probabilities; map them to 0.5 so metrics report a degenerate score
        # instead of crashing the whole run.
        p_tr = np.nan_to_num(net.predict_proba(Xtr)[:, 1], nan=0.5, posinf=1.0, neginf=0.0)
        p_te = np.nan_to_num(net.predict_proba(Xte)[:, 1], nan=0.5, posinf=1.0, neginf=0.0)
        pred = (p_te >= 0.5).astype(int)
        folds["accuracy"].append(float(accuracy_score(y[te], pred)))
        folds["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], pred)))
        folds["balanced_accuracy_optimal"].append(_balanced_accuracy_optimal_score(y[te], p_te))
        folds["roc_auc"].append(float(roc_auc_score(y[te], p_te)))
        folds["balanced_accuracy_calibrated"].append(_calib(y[tr], p_tr, y[te], p_te))
        sytr, sptr = _subj(y[tr], p_tr, groups[tr]); syte, spte = _subj(y[te], p_te, groups[te])
        folds["subj_roc_auc"].append(float(roc_auc_score(syte, spte)) if len(np.unique(syte)) > 1 else float("nan"))
        folds["subj_balanced_accuracy_calibrated"].append(
            _calib(sytr, sptr, syte, spte) if len(np.unique(syte)) > 1 else float("nan"))
        print(f"  fold {k}: roc={folds['roc_auc'][-1]:.3f} bacc_calib={folds['balanced_accuracy_calibrated'][-1]:.3f} "
              f"subj_roc={folds['subj_roc_auc'][-1]:.3f}", flush=True)

    metrics = {m: {"mean": float(np.nanmean(v)) if v else float("nan"),
                   "std": float(np.nanstd(v)) if v else float("nan"), "folds": v}
               for m, v in folds.items()}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "condition": cshort, "strategy": f"partial_unfreeze_k{args.unfreeze_k}",
               "level": args.level, "optimizer": "schedule_free_adamw",
               "results": {args.model: {"metrics": metrics}}}
    p = out / f"results_{args.model}_{cshort}.json"
    p.write_text(json.dumps(payload, indent=2))
    print(f"--> wrote {p}  (roc={metrics['roc_auc']['mean']:.3f} "
          f"subj_roc={metrics['subj_roc_auc']['mean']:.3f} "
          f"bacc_calib={metrics['balanced_accuracy_calibrated']['mean']:.3f})", flush=True)


if __name__ == "__main__":
    main()
