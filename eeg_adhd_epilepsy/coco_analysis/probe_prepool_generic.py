"""
Generic pre-pool probe — apply the BENDR "read the pre-pooling representation"
idea to the transformer FMs (LaBraM, BIOT, LUNA, EEGPT, SignalJEPA).

For transformers the pre-pool representation is the LAST transformer block's
token sequence (B, n_tokens, D), before the head pools it. Standard extraction
mean-pools it; here we pool it RICHLY (mean+std+max — std/max capture the
transient/peaky structure mean-pooling averages away) and compare a convex
logreg probe on:
  mean        (= standard extraction, control)
  mean|std|max (richer pre-pool readout)
If richer > mean, there is signal extraction discards -> a learned attention
adapter is worth building (the BENDR result). If not, the transformer's mean-pool
is already optimal and finetuning won't beat it via this route.

Token sequences are pooled ON THE FLY (never cached), so memory stays small.

Usage:
    python probe_prepool_generic.py --model labram --condition EO_baseline \
        --out-json <path>
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
BATCH = 64
LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"


def _last_block(model):
    """Return the last module of the model's largest transformer ModuleList."""
    import torch.nn as nn
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            if best is None or len(mod) > len(best):
                best = mod
    return best[-1] if best is not None else None


def _probe(X, y, groups, name):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    aucs, bopt = [], []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p1 = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], p1)))
        bopt.append(_balanced_accuracy_optimal_score(y[te], p1))
    out = {"roc_auc": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
           "balanced_accuracy_optimal": float(np.mean(bopt))}
    print(f"  [{name}] roc_auc={out['roc_auc']:.3f}+/-{out['roc_auc_std']:.3f} "
          f"bacc_opt={out['balanced_accuracy_optimal']:.3f} (dim={X.shape[1]})", flush=True)
    return out


def main():
    import torch
    from coco_pipe.decoding.foundation_models.estimators import prepare_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base",
                         "ch_names": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                                      "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]}}
    label_df = normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = load_eeg_epochs(config, label_df)
    X, y, groups, ch_names = _prep_model_data(X, y, groups, args.model, config["signal"])
    if args.model in {"biot", "bendr"}:
        ch_names = _to_modern_nomenclature(ch_names)
    print(f"data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    bkw = {"interpolate_channels": True} if args.model in {"labram", "bendr"} else {}
    prepared = prepare_backend(args.model, X=X, backend="auto", n_outputs=2, device="auto",
                               train_mode="frozen", sfreq=200.0, ch_names=ch_names, backend_kwargs=bkw)
    backend = prepared.backend
    model = backend._model; dev = backend._device
    adapter = getattr(backend, "_channel_adapter", None)
    fd = int(getattr(backend, "_feat_dim", 0) or 0)

    blk = _last_block(model)
    if blk is None:
        raise SystemExit(f"no transformer ModuleList found for {args.model}")
    cap = {}
    h = blk.register_forward_hook(lambda m, i, o: cap.__setitem__("t", o[0] if isinstance(o, tuple) else o))

    mean_f, rich_f = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = X[i:i + BATCH]
            xc = adapter(prepared.adapt(xb)) if adapter is not None else prepared.adapt(xb)
            model(torch.as_tensor(xc, dtype=torch.float32, device=dev))
            tok = cap["t"]
            if tok.dim() == 2:                       # already pooled -> no token axis
                mean_f.append(tok.cpu().numpy()); rich_f.append(tok.cpu().numpy()); continue
            # find the token axis (the one that is NOT the feature dim fd)
            seq = 1 if (tok.shape[-1] == fd or tok.shape[1] != fd) else 2
            mean = tok.mean(seq); std = tok.std(seq); mx = tok.amax(seq)
            mean_f.append(mean.cpu().numpy())
            rich_f.append(torch.cat([mean, std, mx], dim=-1).cpu().numpy())
    h.remove()
    Xm = np.concatenate(mean_f, 0); Xr = np.concatenate(rich_f, 0)
    print(f"last-block token feats: mean{Xm.shape} rich{Xr.shape}", flush=True)

    res = {"mean_pool (extraction control)": _probe(Xm, y, groups, "mean_pool"),
           "rich_pool (mean|std|max)": _probe(Xr, y, groups, "rich_pool")}
    gain = res["rich_pool (mean|std|max)"]["roc_auc"] - res["mean_pool (extraction control)"]["roc_auc"]
    print(f"==> {args.model} {args.condition}: rich-vs-mean gain = {gain:+.3f}", flush=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(
        {"model": args.model, "condition": args.condition.replace("_baseline", ""),
         "rich_minus_mean_roc": gain, "results": res}, indent=2))


if __name__ == "__main__":
    main()
