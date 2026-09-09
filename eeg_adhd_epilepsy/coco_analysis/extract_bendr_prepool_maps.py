"""
Extract BENDR ENCODER feature maps (pre-pooling) and cache them to disk.

BENDR's encoder mixes electrodes in its first conv, so the encoder output is
(n_epochs, 512 feature-channels, T'~26) -- electrodes are gone, the remaining
structure is TEMPORAL. We cache the full time axis (no pooling) so a downstream
trainable temporal adapter can learn which transient patterns matter, instead of
the fixed mean/std/max pooling used in the diagnostic probe.

Cache layout (npz, float16 for the maps): X (n, 512, T'), y (n,), groups (n,).

Usage:
    python extract_bendr_prepool_maps.py --condition EO_baseline \
        --out-npz /home/mat/scratch/results/fine_tune/.../prepool_maps_EO.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import (  # noqa: E402
    load_eeg_epochs, normalize_label_df, _to_modern_nomenclature,
)
from tune_lora import _prep_model_data  # noqa: E402

BATCH = 64
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def main():
    import torch
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base",
                         "ch_names": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                                      "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]}}
    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = load_eeg_epochs(config, label_df)
    X, y, groups, ch_names = _prep_model_data(X, y, groups, "bendr", config["signal"])
    ch_names = _to_modern_nomenclature(ch_names)
    print(f"data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    prepared = prepare_backend("bendr", X=X, backend="auto", n_outputs=2, device="auto",
                               train_mode="frozen", sfreq=200.0, ch_names=ch_names, backend_kwargs={})
    backend = prepared.backend
    model = backend._model; dev = backend._device
    adapter = getattr(backend, "_channel_adapter", None)

    maps = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = X[i:i + BATCH]
            xc = adapter(prepared.adapt(xb)) if adapter is not None else prepared.adapt(xb)
            t = torch.as_tensor(xc, dtype=torch.float32, device=dev)
            enc = model.encoder(t)                     # (B, 512, T') conv only, no transformer
            enc = enc if isinstance(enc, torch.Tensor) else enc[0]
            maps.append(enc.half().cpu().numpy())
    Xmap = np.concatenate(maps, 0)
    print(f"encoder maps: {Xmap.shape} dtype={Xmap.dtype}", flush=True)
    out = Path(args.out_npz); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=Xmap, y=y.astype(np.int8), groups=groups.astype(str))
    print(f"--> wrote {out} ({out.stat().st_size/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
