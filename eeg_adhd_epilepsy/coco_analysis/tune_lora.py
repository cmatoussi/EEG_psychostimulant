"""Model-specific preprocessing for the LoRA finetune drivers.

This file was a Ray-Tune LoRA hyperparameter search; the search is retired -- it
tuned on a split that overlapped the reported CV folds (item 1.6) and its configs
were noise-level. Only `_prep_model_data` remains, imported by run_finetune_tuned.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, "/home/mat/projects/EEG_psychostimulant/eeg_adhd_epilepsy/coco_analysis")

from run_analysis import (  # noqa: E402
    _concat_and_slice_epochs, _slice_epochs, _to_modern_nomenclature,
)


def _prep_model_data(X, y, groups, model_key, signal_cfg):
    """Model-specific preprocessing (mirrors run_analysis.main / run_lr_finder)."""
    ch_names = signal_cfg.get("ch_names")
    if model_key == "biot":
        # Supply modern channel names (T3/T4/T5/T6 -> T7/T8/P7/P8) so BIOT's
        # TCP-bipolar montage resolves; the montage itself is applied by coco-pipe.
        ch_names = _to_modern_nomenclature(signal_cfg.get("ch_names", []))
    if model_key == "labram" and X.shape[-1] < 3000:
        X, y, groups = _concat_and_slice_epochs(X, y, groups, window=3000)
    if model_key == "signaljepa" and X.shape[-1] > 400:
        X, y, groups = _slice_epochs(X, y, groups, window=400)
    return X, y, groups, ch_names
