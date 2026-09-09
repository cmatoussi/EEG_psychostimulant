"""
Diagnostic: does ANY BENDR layer carry epilepsy signal, or only the final
(washed-out) pooled embedding?

We know the final pooled embedding decodes epilepsy at chance (~0.5) with convex
probes. This taps BENDR's ENCODER output (the convolutional features BEFORE the
transformer + final pooling) and runs the same convex logistic-regression probe.
The encoder output keeps far more temporal detail; we summarise it with
mean + std + max pooling over time (std/max preserve transient, high-amplitude
structure like spikes that plain mean-pooling erases).

Two probes are run on the same epochs so the comparison is apples-to-apples:
  encoder_prepool  - mean/std/max over the encoder feature map (the real test)
  final_embedding  - BENDR's normal pooled embedding (control; expect ~0.5)

If even a convex probe on the encoder features stays at ~0.5, no adapter (bridge,
GNN, intermediate) has anything to work with and BENDR is a genuine negative.
If it jumps above chance, an intermediate-layer adapter becomes worth building.

Usage:
    python probe_bendr_prepool.py --condition EO_baseline --out-json <path>
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
from tune_lora import _prep_model_data  # noqa: E402

N_SPLITS = 5
BATCH = 64
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def _pool_time(feats):
    """(B, C, T) tensor -> (B, 3C) numpy: mean|std|max over time."""
    import torch
    m = feats.mean(dim=-1)
    s = feats.std(dim=-1)
    mx = feats.amax(dim=-1)
    return torch.cat([m, s, mx], dim=1).cpu().numpy()


def _probe(X, y, groups, name):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = {m: [] for m in ["accuracy", "balanced_accuracy", "balanced_accuracy_optimal", "roc_auc"]}
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p1 = clf.predict_proba(X[te])[:, 1]
        pred = (p1 >= 0.5).astype(int)
        folds["accuracy"].append(float(accuracy_score(y[te], pred)))
        folds["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], pred)))
        folds["balanced_accuracy_optimal"].append(_balanced_accuracy_optimal_score(y[te], p1))
        folds["roc_auc"].append(float(roc_auc_score(y[te], p1)))
    out = {m: {"mean": float(np.mean(v)) if v else float("nan"),
               "std": float(np.std(v)) if v else float("nan")} for m, v in folds.items()}
    print(f"[{name}] roc_auc={out['roc_auc']['mean']:.3f}+/-{out['roc_auc']['std']:.3f} "
          f"bacc={out['balanced_accuracy']['mean']:.3f} "
          f"bacc_opt={out['balanced_accuracy_optimal']['mean']:.3f} "
          f"(X={X.shape})", flush=True)
    return out


def main():
    import torch
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    ap.add_argument("--max-subjects", type=int, default=0, help="0 = all")
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
    if args.max_subjects:
        keep = np.isin(groups, np.unique(groups)[:args.max_subjects])
        X, y, groups = X[keep], y[keep], groups[keep]
    print(f"data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    prepared = prepare_backend("bendr", X=X, backend="auto", n_outputs=2, device="auto",
                               train_mode="frozen", sfreq=200.0, ch_names=ch_names, backend_kwargs={})
    backend = prepared.backend
    model = backend._model
    dev = backend._device
    adapter = getattr(backend, "_channel_adapter", None)

    # capture encoder output (pre-pooling) via a forward hook
    cap = {}
    h = model.encoder.register_forward_hook(lambda m, i, o: cap.__setitem__("enc", o))

    enc_feats, fin_feats = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = X[i:i + BATCH]
            xc = adapter(prepared.adapt(xb)) if adapter is not None else prepared.adapt(xb)
            t = torch.as_tensor(xc, dtype=torch.float32, device=dev)
            out = model(t, return_features=True)      # runs full model; hook grabs encoder out
            fin = out["features"] if isinstance(out, dict) else out
            enc = cap["enc"]
            enc = enc if isinstance(enc, torch.Tensor) else enc[0]
            enc_feats.append(_pool_time(enc))
            fin = fin if fin.dim() == 2 else _pool_time(fin)  # final is already pooled (B, D)
            fin_feats.append(fin.detach().cpu().numpy() if hasattr(fin, "detach") else np.asarray(fin))
    h.remove()
    Xenc = np.concatenate(enc_feats, 0)
    Xfin = np.concatenate(fin_feats, 0)
    print(f"encoder pre-pool feats: {Xenc.shape} | final embedding feats: {Xfin.shape}", flush=True)

    results = {
        "encoder_prepool": _probe(Xenc, y, groups, "encoder_prepool"),
        "final_embedding": _probe(Xfin, y, groups, "final_embedding (control)"),
    }
    payload = {"model": "bendr", "condition": args.condition.replace("_baseline", ""),
               "target": "epilepsy", "probe": "logreg_grouped_5fold",
               "encoder_pool": "mean|std|max", "results": results}
    outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2))
    print(f"--> wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
