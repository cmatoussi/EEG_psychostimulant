"""Report holes in the clean finetuning results tree.

For each target x strategy x model x condition, checks that both the result JSON
and the per-fold predictions pickle exist, and prints what's missing -- so a sweep
with holes can't pass for complete.

Usage:
    python check_ft_completeness.py
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path("/home/mat/scratch/results/clean_results/finetuning")
MODELS = ["biot", "cbramod", "luna", "reve", "moirai", "neurolm"]
CONDS = ["EO", "EC"]
STRATS = ["ft_only", "lp_ft"]
TARGETS = ["epilepsy", "asm_resistant"]


def main():
    missing = []
    for target in TARGETS:
        base = ROOT / target / "baseline" / "epoch"
        if not base.exists():
            print(f"[{target}] no epoch results dir at {base}")
            continue
        for strat in STRATS:
            for model in MODELS:
                for cond in CONDS:
                    want = {
                        "result json": base / strat / "all" / f"results_{model}_{cond}.json",
                        "preds pkl": base / strat / "_preds" / f"{model}_{cond}.pkl",
                    }
                    for kind, p in want.items():
                        if not p.exists():
                            missing.append((target, strat, model, cond, kind))
    if not missing:
        print("complete: every target x strategy x model x condition has results + preds")
        return
    print(f"{len(missing)} missing entries:")
    for target, strat, model, cond, kind in missing:
        print(f"  {target}/{strat}/{model}/{cond}: no {kind}")


if __name__ == "__main__":
    main()
