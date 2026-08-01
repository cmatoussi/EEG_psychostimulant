"""
Ray Tune hyperparameter search for LoRA fine-tuning of an EEG foundation model.

Reuses the proven data-loading + backend-prep + skorch-fit path from
run_lr_finder.py, wrapped as a Ray Tune trainable: each trial trains LoRA with a
sampled {r, alpha, dropout, lr} on one GroupKFold train split and reports the
held-out validation balanced-accuracy / roc-auc (per epoch, so ASHA can prune).

Two search backends (Ray Tune is the runner in BOTH):
  random  - Ray's random search + ASHA early-stopping scheduler
  optuna  - Ray Tune with the OptunaSearch (Bayesian / TPE) sampler

Usage:
    python tune_lora.py --config combo.yaml --analysis-id fm_lora_cbramod \
        --label-csv labels.csv --search optuna --num-samples 20 \
        --out-dir /home/mat/scratch/results/fine_tune/raytune/optuna/cbramod
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis")

from run_analysis import (  # noqa: E402
    _concat_and_slice_epochs, _slice_epochs, _to_modern_nomenclature, load_eeg_epochs,
    normalize_label_df,
)

MAX_EPOCHS = 15
SEARCH_SPACE_DESC = "r{4,8,16,32} x alpha{8,16,32,64} x dropout{0,.05,.1} x lr~logU(1e-5,1e-2)"


def _prep_model_data(X, y, groups, model_key, signal_cfg):
    """Model-specific preprocessing (mirrors run_analysis.main / run_lr_finder)."""
    ch_names = signal_cfg.get("ch_names")
    if model_key == "biot":
        # Defer BIOT's TCP-bipolar montage to coco-pipe; just supply modern
        # channel names (T3/T4/T5/T6 -> T7/T8/P7/P8) so its montage resolves.
        ch_names = _to_modern_nomenclature(signal_cfg.get("ch_names", []))
    if model_key == "labram" and X.shape[-1] < 3000:
        X, y, groups = _concat_and_slice_epochs(X, y, groups, window=3000)
    if model_key == "signaljepa" and X.shape[-1] > 400:
        X, y, groups = _slice_epochs(X, y, groups, window=400)
    return X, y, groups, ch_names


def _build_net(backend, y_train, lr, batch_size, max_epochs, val):
    """Schedule-Free AdamW (Defazio 2024): no warmup / cosine schedule to tune,
    so the LR-schedule hyperparameters drop out of the search entirely. The
    optimizer must be in .train() mode while training and .eval() mode whenever
    we read out weights for inference (it keeps an averaged eval weight set)."""
    import torch
    import torch.nn as nn
    from schedulefree import AdamWScheduleFree
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from skorch import NeuralNetClassifier
    from skorch.callbacks import Callback
    from sklearn.utils.class_weight import compute_class_weight

    Xval, yval = val

    class _ScheduleFreeReport(Callback):
        def on_train_begin(self, net, **kw):
            opt = getattr(net, "optimizer_", None)
            if opt is not None and hasattr(opt, "train"):
                opt.train()

        def on_epoch_end(self, net, **kw):
            from ray import tune  # ray.train.report is deprecated inside Tune fns
            opt = getattr(net, "optimizer_", None)
            if opt is not None and hasattr(opt, "eval"):
                opt.eval()               # switch to averaged weights for eval
            p = net.predict_proba(Xval)[:, 1]
            if opt is not None and hasattr(opt, "train"):
                opt.train()              # resume training weights
            tune.report({
                "balanced_accuracy": float(balanced_accuracy_score(yval, (p >= 0.5).astype(int))),
                "roc_auc": float(roc_auc_score(yval, p)),
                "epoch": len(net.history),
            })

    out_dim = int(np.unique(y_train).size)
    try:
        backend.reset_head(out_dim)
    except NotImplementedError:
        pass
    mod = backend._get_skorch_module()
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_t = torch.as_tensor(cw, dtype=torch.float32).to(backend._device)
    return NeuralNetClassifier(
        module=mod, module__backend=backend, module__output_dim=out_dim,
        device=backend._device, max_epochs=max_epochs, lr=lr, batch_size=batch_size,
        optimizer=AdamWScheduleFree, optimizer__weight_decay=0.01,
        criterion=nn.CrossEntropyLoss, criterion__weight=cw_t,
        train_split=None, callbacks=[_ScheduleFreeReport()], verbose=0,
    )


def train_lora(config, data=None):
    """Ray Tune trainable: one LoRA fit, report val metric per epoch."""
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend
    Xtr, ytr, Xval, yval, model_key, sfreq, ch_names, batch_size = data

    backend_kwargs = {}
    if model_key in {"labram", "bendr"}:
        backend_kwargs["interpolate_channels"] = True
    backend_kwargs.update({
        "lora_r": int(config["r"]), "lora_alpha": int(config["alpha"]),
        "lora_dropout": float(config["dropout"]), "lora_target_modules": "all-linear",
    })
    prepared = prepare_backend(
        model_key, X=Xtr, backend="auto", n_outputs=2, device="auto",
        train_mode="lora", sfreq=sfreq, ch_names=ch_names, backend_kwargs=backend_kwargs)
    backend = prepared.backend
    # prepared.adapt() only resamples + casts; it does NOT apply the model's
    # fixed-montage channel construction (e.g. BIOT's 19->16 bipolar adapter),
    # which backend.fit() normally does via _construct_channels. Since we drive
    # skorch directly (bypassing backend.fit), apply that adapter here or the
    # model receives the wrong channel count. No-op when no adapter is set.
    def _construct(X):
        adapter = getattr(backend, "_channel_adapter", None)
        return adapter(X) if adapter is not None else X

    Xtr_a, Xval_a = _construct(prepared.adapt(Xtr)), _construct(prepared.adapt(Xval))
    net = _build_net(backend, ytr, float(config["lr"]), batch_size, MAX_EPOCHS, (Xval_a, yval))
    net.fit(Xtr_a, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--analysis-id", required=True)
    ap.add_argument("--label-csv", required=True)
    ap.add_argument("--search", choices=["random", "optuna"], required=True)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    with open(args.config) as f:
        config = yaml.safe_load(f)
    analysis_cfg = next(a for a in config["analyses"] if a["id"] == args.analysis_id)
    signal_cfg = config.get("signal", {})
    model_key = analysis_cfg["model_key"]
    sfreq = float(signal_cfg.get("sfreq", 200.0))
    batch_size = int(analysis_cfg.get("trainer", {}).get("batch_size", 32))

    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = load_eeg_epochs(config, label_df)
    X, y, groups, ch_names = _prep_model_data(X, y, groups, model_key, signal_cfg)
    print(f"[{model_key}] data {X.shape} classes={np.bincount(y).tolist()}", flush=True)

    from sklearn.model_selection import StratifiedGroupKFold
    tr, va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    data = (X[tr], y[tr], X[va], y[va], model_key, sfreq, ch_names, batch_size)
    print(f"[{model_key}] train={len(tr)} val={len(va)} "
          f"(val classes={np.bincount(y[va]).tolist()})", flush=True)

    space = {
        "r": tune.choice([4, 8, 16, 32]),
        "alpha": tune.choice([8, 16, 32, 64]),
        "dropout": tune.choice([0.0, 0.05, 0.1]),
        "lr": tune.loguniform(1e-5, 1e-2),
    }
    if args.search == "optuna":
        search_alg, scheduler = OptunaSearch(metric="balanced_accuracy", mode="max"), None
    else:
        search_alg = None
        scheduler = ASHAScheduler(max_t=MAX_EPOCHS, grace_period=3, reduction_factor=2)

    ray_tmp = os.environ.get("RAY_TMPDIR") or os.environ.get("SLURM_TMPDIR") or "/home/mat/scratch/raytmp"
    Path(ray_tmp).mkdir(parents=True, exist_ok=True)
    ray.init(num_gpus=1, num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
             _temp_dir=ray_tmp, include_dashboard=False, ignore_reinit_error=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # Keep Ray Tune's chatty experiment-state syncing on node-local disk — writing
    # it to the network scratch FS throttles the whole run (state-save grows to
    # 100s+). Only the final summary (best.json / trials.csv) is written to `out`.
    local_results = Path(os.environ.get("SLURM_TMPDIR", ray_tmp)) / "ray_results"
    local_results.mkdir(parents=True, exist_ok=True)

    # Safety net: persist each finished trial to the network dir as it completes,
    # so a job that dies before tuner.fit() returns doesn't lose the whole search.
    progress_csv = out / f"{model_key}_{args.search}_progress.csv"

    class _PersistTrials(tune.Callback):
        def __init__(self):
            self.best = {}

        def on_trial_result(self, iteration, trials, trial, result, **info):
            b = result.get("balanced_accuracy")
            if b is None:
                return
            cur = self.best.get(trial.trial_id)
            if cur is None or b > cur["balanced_accuracy"]:
                cfg = trial.config or {}
                self.best[trial.trial_id] = {
                    "trial": trial.trial_id, "r": cfg.get("r"), "alpha": cfg.get("alpha"),
                    "dropout": cfg.get("dropout"), "lr": cfg.get("lr"),
                    "balanced_accuracy": round(float(b), 4),
                    "roc_auc": round(float(result.get("roc_auc", float("nan"))), 4)}

        def on_trial_complete(self, iteration, trials, trial, **info):
            pd.DataFrame(sorted(self.best.values(),
                                key=lambda d: -d["balanced_accuracy"])).to_csv(progress_csv, index=False)

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train_lora, data=data), {"gpu": 1}),
        param_space=space,
        tune_config=tune.TuneConfig(metric="balanced_accuracy", mode="max",
                                    num_samples=args.num_samples,
                                    search_alg=search_alg, scheduler=scheduler),
        run_config=tune.RunConfig(name=f"{model_key}_{args.search}",
                                  storage_path=str(local_results),
                                  callbacks=[_PersistTrials()]),
    )
    results = tuner.fit()

    df = results.get_dataframe()
    df.to_csv(out / f"{model_key}_{args.search}_trials.csv", index=False)
    best = results.get_best_result("balanced_accuracy", "max")
    payload = {"model": model_key, "search": args.search,
               "search_space": SEARCH_SPACE_DESC, "num_samples": args.num_samples,
               "best_config": best.config,
               "best_balanced_accuracy": best.metrics.get("balanced_accuracy"),
               "best_roc_auc": best.metrics.get("roc_auc")}
    (out / f"{model_key}_{args.search}_best.json").write_text(json.dumps(payload, indent=2))
    print(f"BEST [{model_key}/{args.search}] bacc="
          f"{payload['best_balanced_accuracy']} cfg={best.config}", flush=True)


if __name__ == "__main__":
    main()
