#!/usr/bin/env python3
"""
extracted_embeddig_vis.py  — Visualize REVE extracted embedding evaluation results.

Usage:
    python eeg_adhd_epilepsy/dl/reve/vis/extracted_embeddig_vis.py \
        --model reve --size base --pooling pool --representation subject_flat \
        --out data/results/eval/reve_base_pool_subject_flat.png
"""
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSV_PATH = "/home/mat/projects/EEG_psychostimulant/data/results/eval/embeddings_evaluation_epilepsy.csv"

CONDITION_ORDER = [
    "EC_baseline", "EO_baseline",
    "HV_EC", "HV_EO",
    "PHOTO_EC", "PHOTO_EO",
    "PostHV_EC", "PostHV_EO",
    "RAW_baseline",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            default=CSV_PATH)
    p.add_argument("--model",          default="reve")
    p.add_argument("--size",           default=None)
    p.add_argument("--pooling",        default=None)
    p.add_argument("--representation", default="subject_flat")
    p.add_argument("--last_layer",     default=None,
                   help="Filter by last_layer value, e.g. 'yes' for CBraMod last layer")
    p.add_argument("--out",            default=None,
                   help="Output PNG path. Defaults to next to the CSV.")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)

    # Build filter incrementally
    mask = df["model"].str.lower() == args.model.lower()
    mask &= df["representation"].str.lower() == args.representation.lower()
    if args.size is not None:
        mask &= df["size"].astype(str).str.lower() == args.size.lower()
    if args.pooling is not None:
        mask &= df["pooling"].astype(str).str.lower() == args.pooling.lower()
    if args.last_layer is not None:
        mask &= df["last_layer"].astype(str).str.lower() == args.last_layer.lower()

    sub = df[mask].drop_duplicates(subset=["condition"]).copy()

    if sub.empty:
        print(f"No rows matched filter: model={args.model} size={args.size} "
              f"pooling={args.pooling} representation={args.representation}")
        return

    # Keep conditions in preset order, append extras
    present = [c for c in CONDITION_ORDER if c in sub["condition"].values]
    extras  = [c for c in sub["condition"].unique() if c not in CONDITION_ORDER]
    present += sorted(extras)
    sub = sub.set_index("condition").loc[present].reset_index()

    x = np.arange(len(sub))
    width = 0.35

    # ── Light theme ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_acc = ax.bar(
        x - width / 2,
        sub["accuracy"],
        width,
        label="Accuracy",
        color="#87ceeb",   # light blue
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    bars_bacc = ax.bar(
        x + width / 2,
        sub["balanced accuracy"],
        width,
        label="Balanced Accuracy",
        color="#4b0082",   # dark purple
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    # ── Value labels ────────────────────────────────────────────────────
    def label_bars(rects, color):
        for rect in rects:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.009,
                f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=color,
            )

    label_bars(bars_acc,  "black")
    label_bars(bars_bacc, "black")

    # Chance line
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.2,
               alpha=0.6, label="Chance (0.5)", zorder=2)

    # ── Styling ─────────────────────────────────────────────────────────
    repr_label = args.representation.replace("_", " ").title()
    model_label = args.model.upper()
    parts = [model_label]
    if args.size:    parts.append(args.size.capitalize())
    if args.pooling: parts.append(args.pooling.capitalize())
    if args.last_layer and args.last_layer.lower() == "yes": parts.append("Last Layer")
    parts.append(repr_label)
    ax.set_title(
        "Performance of Extracted Embeddings — " + " ".join(parts),
        fontsize=14, color="black", pad=14, fontweight="bold"
    )
    ax.set_ylabel("Score", color="black", fontsize=12)
    ax.set_xlabel("Condition", color="black", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["condition"], rotation=40, ha="right",
                       fontsize=10, color="black")
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors="black")

    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Legend — top right
    ax.legend(
        fontsize=10,
        framealpha=0.9,
        labelcolor="black",
        facecolor="white",
        edgecolor="#aaaaaa",
        loc="upper right",
    )

    plt.tight_layout()

    # ── Output path ─────────────────────────────────────────────────────
    if args.out:
        out = args.out
    else:
        tag = f"{args.model}_{args.size}_{args.pooling}_{args.representation}"
        out = os.path.join(os.path.dirname(args.csv), f"eval_{tag}.png")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
