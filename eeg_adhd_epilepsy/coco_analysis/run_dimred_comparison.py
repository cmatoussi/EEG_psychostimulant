"""Embeddings-vs-handcrafted dim-reduction comparison report.

Compares the 8 FM-embedding projections against the handcrafted-feature
projection for one cohort/condition/target: a ranked bar chart of separation
quality (2-component logistic-regression balanced accuracy), plus PCA and UMAP
2-D scatter grids colored by class label. Reuses the already-computed
dim_reduction_summary.csv / embedding_*.npy / labels.npy from each source's
existing baseline (or balanced) run -- no re-computation, purely a comparison
view over what's already on disk.

Usage:
    python run_dimred_comparison.py --target epilepsy --variant baseline \
        --cohort all --condition EO --out-dir <dir>
"""
import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLEAN = "/home/mat/scratch/results/clean_results/dim_reduction"
MODELS = ["signaljepa", "labram", "cbramod", "luna", "biot", "bendr", "reve", "eegpt"]
BLUE = "#2a78d6"    # categorical slot 1 -- control / embeddings
ORANGE = "#eb6834"  # categorical slot 2 -- case / handcrafted


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def load_source(root, is_handcrafted, cond):
    """Returns (summary_df, pca2, umap2, labels) or None if missing."""
    sfx = f"_{cond}" if is_handcrafted else ""
    try:
        summ = pd.read_csv(root / f"dim_reduction_summary{sfx}.csv")
        pca2 = np.load(root / f"embedding_PCA_c2{sfx}.npy")
        umap2 = np.load(root / f"embedding_UMAP_c2{sfx}.npy")
        labels = np.load(root / f"labels{sfx}.npy")
    except FileNotFoundError as e:
        print(f"  missing for {root}: {e}", flush=True)
        return None
    return summ, pca2, umap2, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="epilepsy", choices=["epilepsy", "asm_resistant"])
    ap.add_argument("--variant", default="baseline", choices=["baseline", "balanced"])
    ap.add_argument("--cohort", default="all")
    ap.add_argument("--condition", default="EO", choices=["EO", "EC"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base = Path(CLEAN) / args.target / args.variant
    sources = {}
    for model in MODELS:
        root = base / "embed_extraction" / "averaged_epoch" / args.condition / model / args.cohort
        d = load_source(root, is_handcrafted=False, cond=args.condition)
        if d:
            sources[model] = d
    hc_root = base / "handcrafted" / args.cohort
    d = load_source(hc_root, is_handcrafted=True, cond=args.condition)
    if d:
        sources["handcrafted"] = d

    if not sources:
        print("no sources found; nothing to compare", flush=True)
        return
    print(f"loaded {len(sources)}/9 sources: {list(sources)}", flush=True)

    # ---- 1) ranked bar chart: separation (2-comp balanced accuracy) ----
    rows = []
    for name, (summ, *_ ) in sources.items():
        m = summ[(summ.reducer == "PCA") & (summ.n_components == 2)]
        sep = float(m["separation_logreg_balanced_accuracy"].iloc[0]) if len(m) else np.nan
        rows.append({"source": name, "sep": sep, "is_hc": name == "handcrafted"})
    rank_df = pd.DataFrame(rows).dropna(subset=["sep"]).sort_values("sep", ascending=True)

    fig, ax = plt.subplots(figsize=(7.8, 0.55 * len(rank_df) + 1.2))
    colors = [ORANGE if hc else BLUE for hc in rank_df.is_hc]
    ax.barh(rank_df.source, rank_df.sep, color=colors, height=0.6)
    xmax = float(rank_df.sep.max())
    for y, (v, hc) in enumerate(zip(rank_df.sep, rank_df.is_hc)):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color="#999", linestyle="--", linewidth=1)
    ax.set_xlim(0, xmax * 1.18)  # headroom so value labels never crowd the legend
    ax.set_xlabel("Separation (2-component PCA, logreg balanced accuracy)")
    ax.set_title(f"Embeddings vs handcrafted — {args.target} / {args.variant} / "
                 f"{args.cohort} / {args.condition}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="FM embeddings"),
                       Patch(color=ORANGE, label="Handcrafted features"),
                       plt.Line2D([0], [0], color="#999", linestyle="--", label="chance (0.5)")],
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    bar_png = _fig_b64(fig)

    # ---- 2) scatter grids (PCA-2D and UMAP-2D), colored by class label ----
    def scatter_grid(idx, title):
        names = list(sources.keys())
        n = len(names)
        ncols = 3
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for i, name in enumerate(names):
            ax = axes[i]
            _, pca2, umap2, labels = sources[name]
            pts = pca2 if idx == "pca" else umap2
            for cls, color, lbl in [(0, BLUE, "control"), (1, ORANGE, args.target)]:
                m = labels == cls
                ax.scatter(pts[m, 0], pts[m, 1], s=10, c=color, alpha=0.6,
                          edgecolors="none", label=lbl)
            sep = sources[name][0]
            sep_row = sep[(sep.reducer == ("PCA" if idx == "pca" else "UMAP")) & (sep.n_components == 2)]
            sep_v = float(sep_row["separation_logreg_balanced_accuracy"].iloc[0]) if len(sep_row) else float("nan")
            ax.set_title(f"{name}\nsep={sep_v:.3f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        for j in range(len(names), len(axes)):
            axes[j].axis("off")
        axes[0].legend(loc="upper right", frameon=False, fontsize=8, markerscale=1.5)
        fig.suptitle(title, fontsize=13, y=1.01 if nrows == 1 else 1.0)
        fig.tight_layout()
        return _fig_b64(fig)

    pca_png = scatter_grid("pca", f"PCA (2 components) — colored by {args.target}")
    umap_png = scatter_grid("umap", f"UMAP (2 components) — colored by {args.target}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(out / f"comparison_ranking_{args.cohort}_{args.condition}.csv", index=False)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
body{{font-family:-apple-system,sans-serif;max-width:1200px;margin:auto;padding:24px;color:#0b0b0b}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:6px}}
img{{max-width:100%;border:1px solid #e5e5e2;border-radius:4px;margin:8px 0}}
.meta{{color:#52514e;font-size:13px}}
</style></head><body>
<h1>Dim-reduction comparison: FM embeddings vs handcrafted features</h1>
<p class="meta">target={args.target} &middot; variant={args.variant} &middot; cohort={args.cohort} &middot;
condition={args.condition} &middot; level=subject (averaged_epoch embeddings vs handcrafted subject features)</p>
<h2>Separation ranking (2-component PCA, logistic-regression balanced accuracy)</h2>
<img src="data:image/png;base64,{bar_png}">
<h2>PCA projections, colored by {args.target}</h2>
<img src="data:image/png;base64,{pca_png}">
<h2>UMAP projections, colored by {args.target}</h2>
<img src="data:image/png;base64,{umap_png}">
</body></html>"""
    report_path = out / f"comparison_{args.cohort}_{args.condition}.html"
    report_path.write_text(html)
    print(f"--> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
