#!/usr/bin/env python3
"""
EO Classification Performance: CBraMod vs REVE (with/without fine-tuning)
Reads real fold-level data from CSVs and computes mean ± std per metric.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── helpers ──────────────────────────────────────────────────────────────────

def load_fold_rows(path, condition="EO_baseline"):
    """Return only per-fold (non-summary) rows for the requested condition."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalise column names: both formats use Val_AUC / Val_Acc / Val_BAcc
    # Identify the Fold column (sometimes 'Fold', values like '1','Fold 1','Average',…)
    fold_col = "Fold"

    # Keep only rows matching the condition
    cond_col = "Condition" if "Condition" in df.columns else df.columns[0]
    df = df[df[cond_col].astype(str).str.strip() == condition].copy()

    # Drop summary rows (Average / Std / StdDev)
    mask = df[fold_col].astype(str).str.match(r"^\d|^Fold\s+\d")
    df = df[mask]

    return df[["Val_AUC", "Val_Acc", "Val_BAcc"]].astype(float)


# ── load data ─────────────────────────────────────────────────────────────────

files = {
    "CBraMod\nNo FT":    "/home/mat/projects/EEG_psychostimulant/data/results/eval/cbramod_without_finetuning.csv",
    "CBraMod\nFine-tuned": "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/cbramod/lp_ft/finetune_results_cbramod_lp_ft.csv",
    "REVE\nNo FT":       "/home/mat/projects/EEG_psychostimulant/data/results/eval/reve_base_without_finetuning.csv",
    "REVE\nFine-tuned":  "/home/mat/projects/EEG_psychostimulant/data/results/finetuned_param/reve/base/unbalanced/finetune_results_reve_base_unbalanced.csv",
}

means = {}
stds  = {}
for label, path in files.items():
    df = load_fold_rows(path, condition="EO_baseline")
    means[label] = [df["Val_AUC"].mean(), df["Val_BAcc"].mean(), df["Val_Acc"].mean()]
    stds[label]  = [df["Val_AUC"].std(),  df["Val_BAcc"].std(),  df["Val_Acc"].std()]
    print(f"{label.replace(chr(10),' ')}: AUC={means[label][0]:.3f}±{stds[label][0]:.3f} "
          f"BAcc={means[label][1]:.3f}±{stds[label][1]:.3f} "
          f"Acc={means[label][2]:.3f}±{stds[label][2]:.3f}")

# ── plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         20,
    "axes.titlesize":    28,
    "axes.labelsize":    24,
    "xtick.labelsize":   21,
    "ytick.labelsize":   21,
    "legend.fontsize":   19,
    "figure.titlesize":  30,
})

metrics = ["AUC", "Balanced\nAccuracy", "Accuracy"]
labels  = list(means.keys())

# dark red, dark green, dark blue, dark purple
COLORS = {
    "CBraMod\nNo FT":    "#8B0000",   # dark red
    "CBraMod\nFine-tuned": "#1B5E20", # dark green
    "REVE\nNo FT":       "#0D2B6E",   # dark blue
    "REVE\nFine-tuned":  "#4A0072",   # dark purple
}

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)

x      = np.arange(len(metrics))
width  = 0.18
offsets = np.linspace(-1.5 * width, 1.5 * width, len(labels))

for label, offset in zip(labels, offsets):
    vals = means[label]
    errs = stds[label]
    bars = ax.bar(
        x + offset,
        vals,
        width,
        label=label,
        color=COLORS[label],
        yerr=errs,
        capsize=6,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2),
    )
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.018,   # above the SD cap
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
            color="black",
        )

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, label="Chance (0.5)")

ax.set_title("EO Classification Performance: CBraMod vs REVE", pad=18, weight="bold")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.45, 0.88)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_base = "/home/mat/projects/EEG_psychostimulant/data/results/EO_CBraMod_REVE_finetuning_comparison"
for ext in ("png", "pdf", "svg"):
    plt.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_base}.{ext}")

plt.close()
