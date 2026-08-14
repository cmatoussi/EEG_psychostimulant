"""
LR Range Test for LoRA fine-tuning.

Sweeps LR from lr_start to lr_end exponentially over num_iter batches
(single GroupKFold train split, no CV), records smoothed loss, and saves
a loss-vs-LR plot.  Read the plot: pick the LR just BEFORE the loss
starts climbing steeply (steepest-descent point is marked in red).

Usage:
    python run_lr_finder.py \
        --config /path/to/combo_XX.yaml \
        --analysis-id fm_lora_cbramod \
        --label-csv /path/to/labels.csv \
        --output-dir /home/mat/scratch/results/lr_finder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis")

from run_analysis import (
    _concat_and_slice_epochs,
    _slice_epochs,
    _to_biot_bipolar,
    load_eeg_epochs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ─── LR finder callback ────────────────────────────────────────────────────────

class _LRFinderStop(Exception):
    pass


def _make_lr_finder_callback(
    lr_start: float,
    lr_end: float,
    num_iter: int,
    beta: float,
    diverge_factor: float,
    lrs_out: list,
    losses_out: list,
):
    """Return a skorch Callback that implements the LR range test."""
    from skorch.callbacks import Callback

    lr_mult = (lr_end / lr_start) ** (1.0 / max(1, num_iter - 1))

    class _FinderCB(Callback):
        def initialize(self):
            self._batch_num = 0
            self._avg_loss = 0.0
            self._best_loss = float("inf")
            return self

        def on_batch_begin(self, net, batch=None, training=None, **kwargs):
            if not training:
                return
            lr = lr_start * (lr_mult ** self._batch_num)
            if hasattr(net, "optimizer_"):
                for g in net.optimizer_.param_groups:
                    g["lr"] = lr

        def on_batch_end(self, net, batch=None, training=None, **kwargs):
            if not training:
                return

            current_lr = lr_start * (lr_mult ** self._batch_num)

            try:
                loss = float(net.history[-1, "batches", -1, "train_loss"])
            except (KeyError, IndexError):
                self._batch_num += 1
                return

            # Bias-corrected exponential smoothing
            self._avg_loss = beta * self._avg_loss + (1 - beta) * loss
            smoothed = self._avg_loss / (1 - beta ** (self._batch_num + 1))

            lrs_out.append(current_lr)
            losses_out.append(smoothed)

            if smoothed < self._best_loss:
                self._best_loss = smoothed

            self._batch_num += 1
            logger.info(
                "[LR finder] batch %3d: lr=%.2e  loss=%.4f",
                self._batch_num, current_lr, smoothed,
            )

            if self._batch_num >= num_iter:
                raise _LRFinderStop(f"Done after {num_iter} batches")
            if self._batch_num > 10 and smoothed > diverge_factor * self._best_loss:
                raise _LRFinderStop(f"Diverged at lr={current_lr:.2e}")

    return _FinderCB()


# ─── Core finder ───────────────────────────────────────────────────────────────

def run_lr_finder(
    backend,
    X_adapted: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 32,
    lr_start: float = 1e-7,
    lr_end: float = 1e-1,
    num_iter: int = 100,
    beta: float = 0.98,
    diverge_factor: float = 4.0,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the LR range test; return (lrs, smoothed_losses)."""
    import torch
    import torch.nn as nn
    from skorch import NeuralNetClassifier

    lrs: list[float] = []
    losses: list[float] = []
    finder_cb = _make_lr_finder_callback(
        lr_start, lr_end, num_iter, beta, diverge_factor, lrs, losses
    )

    # Reset head for binary classification (some models like LUNA don't support
    # reset_head() post-load — they're already initialised with the right n_outputs)
    out_dim = int(np.unique(y_train).size)
    try:
        backend.reset_head(out_dim)
    except NotImplementedError:
        pass

    skorch_module_cls = backend._get_skorch_module()

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_tensor = torch.as_tensor(cw, dtype=torch.float32).to(backend._device)

    net = NeuralNetClassifier(
        module=skorch_module_cls,
        module__backend=backend,
        module__output_dim=out_dim,
        device=backend._device,
        max_epochs=9999,           # stopped by callback exception
        lr=lr_start,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=weight_decay,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=cw_tensor,
        callbacks=[finder_cb],
        train_split=None,          # no validation split during finder
        verbose=0,
    )

    try:
        net.fit(X_adapted, y_train)
    except _LRFinderStop as exc:
        logger.info("LR finder stopped: %s", exc)

    return np.array(lrs), np.array(losses)


# ─── Plot ──────────────────────────────────────────────────────────────────────

def plot_lr_finder(
    lrs: np.ndarray,
    losses: np.ndarray,
    model_key: str,
    output_path: Path,
) -> float | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(lrs) < 5:
        logger.warning("Too few points to plot (%d). Skipping.", len(lrs))
        return None

    # Steepest-descent point:
    # beta=0.98 EMA needs ~1/(1-0.98)=50 batches to fully settle, but bias
    # correction compensates most of it. Skip 15 batches to be safe.
    # Use nanargmin so NaN (diverged batches) don't corrupt the minimum search.
    losses_safe = np.where(np.isfinite(losses), losses, np.inf)
    warmup = min(15, len(lrs) // 6)
    post_warmup = losses_safe[warmup:]
    if np.all(np.isinf(post_warmup)):
        real_min_idx = warmup
    else:
        real_min_idx = warmup + int(np.nanargmin(post_warmup))
    search_end = max(real_min_idx + 1, warmup + 3)
    search_losses = losses_safe[warmup:search_end]
    if len(search_losses) < 3:
        search_losses = losses_safe
        warmup = 0
    grad = np.gradient(search_losses)
    best_idx = warmup + int(np.argmin(grad))
    suggested_lr = float(lrs[best_idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, linewidth=1.5, label="Smoothed loss")
    ax.axvline(
        suggested_lr,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Steepest descent: {suggested_lr:.2e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Smoothed Loss", fontsize=12)
    ax.set_title(f"LR Range Test — {model_key}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", output_path)
    return suggested_lr


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LR range test for LoRA fine-tuning")
    parser.add_argument("--config", required=True, help="Path to combo YAML")
    parser.add_argument("--analysis-id", required=True, help="Analysis block ID")
    parser.add_argument("--label-csv", required=True, help="Cohort label CSV")
    parser.add_argument("--output-dir", default="/home/mat/scratch/results/lr_finder")
    parser.add_argument("--lr-start", type=float, default=1e-7)
    parser.add_argument("--lr-end", type=float, default=1e-1)
    parser.add_argument("--num-iter", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.98,
                        help="EMA smoothing factor (lower = responds faster; 0.85 for short sweeps)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    analyses = config.get("analyses", [])
    analysis_cfg = next(
        (a for a in analyses if a["id"] == args.analysis_id), None
    )
    if analysis_cfg is None:
        raise ValueError(f"Analysis ID '{args.analysis_id}' not found in config.")

    signal_cfg = config.get("signal", {})
    model_key = analysis_cfg["model_key"]
    lora_raw = analysis_cfg.get("lora", {})
    trainer_raw = analysis_cfg.get("trainer", {})
    batch_size = int(trainer_raw.get("batch_size", 32))
    weight_decay = float(trainer_raw.get("weight_decay", 0.01))
    ch_names = signal_cfg.get("ch_names")
    sfreq = float(signal_cfg.get("sfreq", 200.0))

    # ── Load data ──
    label_df = pd.read_csv(args.label_csv)
    logger.info("Loaded %d subjects from %s", len(label_df), args.label_csv)

    X, y, groups = load_eeg_epochs(config, label_df)
    logger.info("Data: %s  classes=%s", X.shape, dict(zip(*np.unique(y, return_counts=True))))

    # Apply model-specific preprocessing (mirrors run_analysis.py main())
    if model_key == "biot":
        X, biot_ch_names = _to_biot_bipolar(X, signal_cfg.get("ch_names", []))
        signal_cfg = {**signal_cfg, "ch_names": biot_ch_names}
        ch_names = signal_cfg["ch_names"]
    if model_key == "labram" and X.shape[-1] < 3000:
        X, y, groups = _concat_and_slice_epochs(X, y, groups, window=3000)
    if model_key == "signaljepa" and X.shape[-1] > 400:
        X, y, groups = _slice_epochs(X, y, groups, window=400)

    logger.info("After preprocessing: %s", X.shape)

    # ── Take first GroupKFold train split ──
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    train_idx, _ = next(gkf.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    logger.info("Training fold: %d samples", len(y_train))

    # ── Load backend with LoRA ──
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    backend_kwargs: dict = {}
    if model_key in {"labram", "bendr"}:
        backend_kwargs["interpolate_channels"] = True
    backend_kwargs.update(
        {
            "lora_r": lora_raw.get("r", "auto"),
            "lora_alpha": lora_raw.get("alpha", "auto"),
            "lora_dropout": lora_raw.get("dropout", 0.05),
            "lora_target_modules": lora_raw.get("target_modules", "all-linear"),
        }
    )

    logger.info("Loading %s with LoRA …", model_key)
    prepared = prepare_backend(
        model_key,
        X=X_train,
        backend="auto",
        n_outputs=2,
        device="auto",
        train_mode="lora",
        sfreq=sfreq,
        ch_names=ch_names,
        backend_kwargs=backend_kwargs,
    )
    backend = prepared.backend
    X_adapted = prepared.adapt(X_train)
    logger.info("Backend ready. Adapted X: %s", X_adapted.shape)

    # ── Run LR finder ──
    logger.info(
        "Starting LR range test: %d batches, lr %s → %s, beta=%.2f",
        args.num_iter, args.lr_start, args.lr_end, args.beta,
    )
    lrs, losses = run_lr_finder(
        backend,
        X_adapted,
        y_train,
        batch_size=batch_size,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        num_iter=args.num_iter,
        beta=args.beta,
        weight_decay=weight_decay,
    )

    if len(lrs) == 0:
        logger.error("No batches recorded — check model loading.")
        sys.exit(1)

    # ── Save raw results ──
    suffix = f"_beta{args.beta:.2f}".replace(".", "") if args.beta != 0.98 else ""
    results_path = output_dir / f"lr_finder_{model_key}{suffix}.json"
    with open(results_path, "w") as f:
        json.dump({"model_key": model_key, "beta": args.beta,
                   "lrs": lrs.tolist(), "losses": losses.tolist()}, f)
    logger.info("Raw results → %s", results_path)

    # ── Plot ──
    plot_path = output_dir / f"lr_finder_{model_key}{suffix}.png"
    suggested_lr = plot_lr_finder(lrs, losses, model_key, plot_path)
    if suggested_lr is not None:
        logger.info(
            "═══════════════════════════════════════════════════\n"
            "  %s  →  suggested LR: %.2e\n"
            "  (use value ~10× smaller for stable training)\n"
            "═══════════════════════════════════════════════════",
            model_key, suggested_lr,
        )


if __name__ == "__main__":
    main()