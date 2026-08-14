"""
Isolated MOIRAI embedding extraction + classical-head probe (Option 1).

MOIRAI (Woo et al., Salesforce uni2ts) is a masked-encoder time-series
foundation model.  For a frozen embedding + classical-head comparison we tap the
transformer `reprs` produced inside `MoiraiModule.forward` -- i.e. the token
representations *before* the distribution head (`param_proj`) -- exactly the
role the encoder plays in the other FMs.  We run it as a pure encoder:
`prediction_mask` is all-False so nothing is masked/forecast; every token is a
real observed patch and we mean-pool the resulting token reprs.

Multivariate packing (uses MOIRAI's cross-variate attention, its actual
strength): each EEG epoch (19 ch, T) is split into patches of size PATCH; token
t = one (channel, time-patch) pair, with variate_id = channel, time_id = patch
index.  With PATCH=128 and T=2000 -> 15 patches x 19 ch = 285 tokens <= 512
(max_seq_len).  Pooled repr -> 384-dim (d_model) per epoch.

uni2ts on ComputeCanada needs shims: its package __init__ pulls
finetune->lightning, and jaxtyping.PyTree lazily imports jax.  We stub jax
(backed by torch's pytree) and pre-empt the moirai subpackage __init__ -- the
model forward never touches either.

Saves a `moirai_{cond}_epoch_embeddings.csv` in the format
load_precomputed_embeddings expects, and runs a logreg/svm/rf 5-fold grouped CV
for an immediate comparable AUC (epoch + subject-mean).

Usage:
    PYTHONPATH=/home/mat/projects/coco-pipe:/home/mat/uni2ts/src:\
/home/mat/.local/lib/python3.11/site-packages \
    python extract_moirai_embeddings.py --condition EO_baseline
"""
from __future__ import annotations
import argparse, sys, types
from pathlib import Path
import numpy as np
import pandas as pd


def _install_shims():
    """jax stub (jaxtyping.PyTree) + pre-empt moirai __init__ (finetune->lightning)."""
    import torch.utils._pytree as tp
    jax = types.ModuleType("jax"); jtu = types.ModuleType("jax.tree_util")
    for n in ("tree_flatten", "tree_unflatten", "tree_map", "tree_leaves", "tree_structure"):
        setattr(jtu, n, getattr(tp, n))
    class _Arr:  # noqa: E701
        ...
    jax.Array = _Arr; jax.tree_util = jtu
    jnp = types.ModuleType("jax.numpy"); jnp.ndarray = _Arr; jax.numpy = jnp
    sys.modules.update({"jax": jax, "jax.tree_util": jtu, "jax.numpy": jnp})
    pkg = types.ModuleType("uni2ts.model.moirai")
    pkg.__path__ = ["/home/mat/uni2ts/src/uni2ts/model/moirai"]
    sys.modules["uni2ts.model.moirai"] = pkg


sys.path.insert(0, "/home/mat/projects/coco-pipe")
sys.path.insert(0, "/home/mat/uni2ts/src")
sys.path.insert(0, "/home/mat/.local/lib/python3.11/site-packages")
sys.path.insert(0, str(Path(__file__).parent))
import run_analysis as ra  # noqa: E402

LABEL_CSV = "/home/mat/scratch/patients_metadata_clean_without_source.csv"
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
        "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
BATCH = 32
PATCH = 128  # 15 patches/ch * 19 ch = 285 tokens <= 512 max_seq_len


def _pack_epochs(X, patch, max_patch):
    """(n, 19, T) -> packed forward inputs. Token t = (channel, time-patch).

    Returns dict of tensors (numpy), shapes (n, seq_len, ...). seq_len = A*V.
    """
    n, V, T = X.shape
    A = T // patch
    Xt = X[:, :, : A * patch].astype(np.float32)                  # (n, V, A*patch)
    # per token value block, padded to max_patch
    patches = Xt.reshape(n, V, A, patch)                          # (n, V, A, patch)
    patches = np.transpose(patches, (0, 2, 1, 3)).reshape(n, A * V, patch)  # token order (a, v)
    seq = A * V
    target = np.zeros((n, seq, max_patch), np.float32)
    target[:, :, :patch] = patches
    observed = np.zeros((n, seq, max_patch), bool)
    observed[:, :, :patch] = True
    var = np.tile(np.arange(V, dtype=np.int64), A)                # (seq,) token->channel
    tim = np.repeat(np.arange(A, dtype=np.int64), V)             # (seq,) token->patch index
    variate_id = np.broadcast_to(var, (n, seq)).copy()
    time_id = np.broadcast_to(tim, (n, seq)).copy()
    sample_id = np.ones((n, seq), np.int64)                       # single packed series/row
    prediction_mask = np.zeros((n, seq), bool)                   # encode only, nothing to predict
    patch_size = np.full((n, seq), patch, np.int64)
    return dict(target=target, observed_mask=observed, sample_id=sample_id,
                time_id=time_id, variate_id=variate_id,
                prediction_mask=prediction_mask, patch_size=patch_size)


def _encode(model, batch, device):
    """Replicate MoiraiModule.forward up to `reprs` (pre param_proj); mean-pool tokens."""
    import torch
    from uni2ts.common.torch_util import mask_fill, packed_attention_mask
    t = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
    loc, scale = model.scaler(t["target"],
                              t["observed_mask"] * ~t["prediction_mask"].unsqueeze(-1),
                              t["sample_id"], t["variate_id"])
    scaled = (t["target"] - loc) / scale
    reprs = model.in_proj(scaled, t["patch_size"])
    reprs = mask_fill(reprs, t["prediction_mask"], model.mask_encoding.weight)
    reprs = model.encoder(reprs, packed_attention_mask(t["sample_id"]),
                          time_id=t["time_id"], var_id=t["variate_id"])  # (b, seq, d_model)
    return reprs.mean(dim=1).float().cpu().numpy()                       # (b, d_model)


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
    _install_shims()
    import torch
    torch.set_num_threads(int(np.clip(torch.get_num_threads(), 1, 8)))
    from uni2ts.model.moirai.module import MoiraiModule

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="EO_baseline")
    ap.add_argument("--model", default="Salesforce/moirai-1.0-R-small")
    ap.add_argument("--out-dir", default="/home/mat/scratch/results/extraceted_embeddings_files/moirai")
    ap.add_argument("--label-csv", default=LABEL_CSV)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    config = {"paths": {"data_root": "/home/mat/projects/rrg-kjerbi/shared/eeg-adhdh-epilepsy/"
                        "BIDS/derivatives/preproc/"},
              "signal": {"sfreq": 200.0, "conditions": [args.condition],
                         "epoch_desc": "base", "ch_names": CH19}}
    label_df = ra.normalize_label_df(pd.read_csv(args.label_csv))
    X, y, groups = ra.load_eeg_epochs(config, label_df)
    print(f"data {X.shape} classes={np.bincount(y).tolist()} subjects={len(np.unique(groups))}", flush=True)

    model = MoiraiModule.from_pretrained(args.model).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    max_patch = max(model.patch_sizes)

    embs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            batch = _pack_epochs(X[i:i + BATCH], PATCH, max_patch)
            embs.append(_encode(model, batch, dev))
    E = np.concatenate(embs, 0)
    print(f"MOIRAI embeddings: {E.shape}", flush=True)

    cond = args.condition
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(E, columns=[f"embedding_{i}" for i in range(E.shape[1])])
    df.insert(0, "study_id", np.asarray(groups).astype(str))
    df.insert(1, "epilepsy", y.astype(int))
    csv = out / f"moirai_{cond}_epoch_embeddings.csv"
    df.to_csv(csv, index=False)
    print(f"--> wrote {csv}", flush=True)

    print("=== epoch-level classical-head probe ===", flush=True)
    _probe(E, y, np.asarray(groups), "moirai/epoch")
    u = np.unique(groups)
    Xs = np.stack([E[np.asarray(groups) == s].mean(0) for s in u])
    ys = np.array([int(y[np.asarray(groups) == s][0]) for s in u])
    print("=== subject-level (mean-embedding) probe ===", flush=True)
    _probe(Xs, ys, u, "moirai/subject")


if __name__ == "__main__":
    main()
