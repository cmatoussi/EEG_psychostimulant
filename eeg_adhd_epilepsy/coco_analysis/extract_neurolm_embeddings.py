"""
Isolated NeuroLM embedding extraction + classical-head probe (Option 1).

NeuroLM (Jiang et al., github 935963004/NeuroLM) is an EEG foundation model: a
VQ neural-transformer tokenizer + a GPT. For a frozen embedding + classical-head
comparison we tap the **VQ encoder** (its `forward_features`, mean-pooled) — the
same role LaBraM's encoder plays. This mirrors your other FM extraction so the
number drops straight into your comparison.

NeuroLM input format (from downstream_dataset.py):
  - scale EEG by 1/100 (microvolt normalisation)
  - split each channel into 1 s / 200-sample patches: (N_ch, A*200) -> (A*N_ch, 200)
    token order is (time-patch, channel)
  - input_chans = channel index into NeuroLM's standard_1020 list, per token
  - input_time  = time-patch index per token
  - encoder.forward_features(x, input_chans, input_times) -> mean-pooled (B, 768)

Channel names must be UPPERCASE + modern 10-20 (T3/T4/T5/T6 -> T7/T8/P7/P8).

Saves a `neurolm_{cond}_{level}_embeddings.csv` in the format
load_precomputed_embeddings expects, and runs a logreg/svm/rf 5-fold grouped CV
for an immediate comparable AUC.

Usage (main env has torch/einops/mne; add the NeuroLM repo to PYTHONPATH):
    PYTHONPATH=/home/mat/projects/coco-pipe:/home/mat/projects/NeuroLM \
    python extract_neurolm_embeddings.py --condition EO_baseline \
        --vq /home/mat/scratch/neurolm_ckpt/VQ.pt \
        --out-dir /home/mat/scratch/results/extraceted_embeddings_files/moirai/../neurolm
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/mat/projects/NeuroLM")
import run_analysis as ra  # noqa: E402

LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"
# our 19 channels -> NeuroLM standard_1020 names (uppercase, modern 10-20)
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
        "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
_NL = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
CH19_NL = [_NL.get(c, c).upper() for c in CH19]
BATCH = 64
PATCH = 200  # 1 s at 200 Hz


def _load_vq_encoder(vq_path, device):
    import torch
    from model.model_vq import VQ
    from model.model_neural_transformer import NTConfig
    enc_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.0, num_classes=0, in_chans=1, out_chans=16)
    dec_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.0, num_classes=0, in_chans=128)
    vq = VQ(NTConfig(**enc_args), NTConfig(**dec_args))
    sd = torch.load(vq_path, map_location="cpu", weights_only=False)  # trusted official NeuroLM ckpt
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = vq.load_state_dict(sd, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    print(f"  VQ loaded: {len(enc_missing)} encoder keys missing, {len(unexpected)} unexpected", flush=True)
    enc = vq.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def _epoch_tokens(X, chan_idx):
    """(n, 19, T) -> tokens (n, A*19, 200), input_chans (A*19,), input_time (A*19,).
    Token order (time-patch, channel), matching downstream_dataset."""
    from einops import rearrange
    n, nch, T = X.shape
    A = T // PATCH
    Xc = (X[:, :, : A * PATCH] / 100.0).astype(np.float32)
    toks = rearrange(Xc, "n c (a t) -> n (a c) t", t=PATCH)          # (n, A*nch, 200)
    input_chans = np.tile(np.asarray(chan_idx, dtype=np.int64), A)   # per token, (A*nch,)
    input_time = np.repeat(np.arange(A, dtype=np.int64), nch)        # per token
    return toks, input_chans, input_time


def _probe(X, y, groups, name):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    heads = {"logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
             "svm": SVC(kernel="rbf", probability=True, class_weight="balanced"),
             "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced")}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for hn, clf in heads.items():
        aucs, bas = [], []
        for tr, te in sgkf.split(X, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            pipe = make_pipeline(StandardScaler(), clf)
            pipe.fit(X[tr], y[tr])
            p1 = pipe.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p1))
            bas.append(balanced_accuracy_score(y[te], (p1 >= 0.5).astype(int)))
        print(f"  [{name}/{hn}] roc_auc={np.mean(aucs):.3f} bal_acc={np.mean(bas):.3f}", flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--vq", default="/home/mat/scratch/neurolm_ckpt/VQ.pt")
    ap.add_argument("--out-dir", default="/home/mat/scratch/results/extraceted_embeddings_files/neurolm")
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from dataset import standard_1020
    chan_idx = [standard_1020.index(c) for c in CH19_NL]  # errors loudly if a name is absent
    print(f"channel map -> NeuroLM indices: {dict(zip(CH19, chan_idx))}", flush=True)

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition], "epoch_desc": "base", "ch_names": CH19}}
    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = ra.load_eeg_epochs(config, label_df)
    print(f"data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    enc = _load_vq_encoder(args.vq, dev)
    toks, ich, itime = _epoch_tokens(X, chan_idx)
    ich_t = torch.as_tensor(ich, device=dev).unsqueeze(0)
    itime_t = torch.as_tensor(itime, device=dev).unsqueeze(0)

    embs = []
    with torch.no_grad():
        for i in range(0, len(toks), BATCH):
            xb = torch.as_tensor(toks[i:i + BATCH], dtype=torch.float32, device=dev)
            b = xb.shape[0]
            e = enc.forward_features(xb, input_chans=ich_t.expand(b, -1),
                                     input_times=itime_t.expand(b, -1))  # (b, 768) mean-pooled
            embs.append(e.float().cpu().numpy())
    E = np.concatenate(embs, 0)
    print(f"NeuroLM embeddings: {E.shape}", flush=True)

    # save in load_precomputed_embeddings CSV format
    cond = args.condition
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(E, columns=[f"embedding_{i}" for i in range(E.shape[1])])
    df.insert(0, "study_id", np.asarray(groups).astype(str))
    df.insert(1, "epilepsy", y.astype(int))
    csv = out / f"neurolm_{cond}_epoch_embeddings.csv"
    df.to_csv(csv, index=False)
    print(f"--> wrote {csv}", flush=True)

    print("=== epoch-level classical-head probe ===", flush=True)
    _probe(E, y, np.asarray(groups), "neurolm/epoch")
    # subject level: mean embedding per subject
    u = np.unique(groups)
    Xs = np.stack([E[np.asarray(groups) == s].mean(0) for s in u])
    ys = np.array([int(y[np.asarray(groups) == s][0]) for s in u])
    print("=== subject-level (mean-embedding) probe ===", flush=True)
    _probe(Xs, ys, u, "neurolm/subject")


if __name__ == "__main__":
    main()
