"""Coarse-to-fine 1-D hyperparameter zoom, used as the inner (nested) grid search by
probe_nested_optuna and run_handcrafted_cohorts. The old standalone head search that
lived here was retired: its best-of-search score was optimistic (selected on the same
folds it reported), and the nested probe answers tuned-vs-fixed honestly."""
from __future__ import annotations
import numpy as np


def refine_1d(score_fn, lo, hi, log_space, n_rounds=3, points=(7, 5, 5), is_int=False):
    """Coarse-to-fine 1-D search. Returns (best_value, best_score, history)."""
    history = []
    best_v, best_s = None, -np.inf
    cur_lo, cur_hi = lo, hi
    for rnd in range(n_rounds):
        n = points[min(rnd, len(points) - 1)]
        if log_space:
            grid = np.geomspace(cur_lo, cur_hi, n)
        else:
            grid = np.linspace(cur_lo, cur_hi, n)
        if is_int:
            grid = sorted(set(int(round(v)) for v in grid))
        else:
            grid = sorted(set(round(float(v), 6) for v in grid))
        for v in grid:
            s = score_fn(v)
            history.append({"round": rnd + 1, "value": v, "score": s})
            if s > best_s:
                best_s, best_v = s, v
        # narrow window around the best point found so far
        if log_space:
            half_span_log = (np.log10(hi) - np.log10(lo)) / (2 * (rnd + 2))
            cur_lo = max(lo, 10 ** (np.log10(best_v) - half_span_log))
            cur_hi = min(hi, 10 ** (np.log10(best_v) + half_span_log))
        else:
            half_span = (hi - lo) / (2 * (rnd + 2))
            cur_lo = max(lo, best_v - half_span)
            cur_hi = min(hi, best_v + half_span)
            if is_int and cur_hi - cur_lo < 1:
                cur_lo, cur_hi = max(lo, best_v - 1), min(hi, best_v + 1)
    return best_v, best_s, history
