"""Matched-operating-point comparison (reviewer requirement, 2026-10-21).

Comparing calibrators at the same nominal alpha is not a fair efficiency comparison: a method
can buy low-support coverage simply by deferring more (returning the full set {BBB-, BBB+}).
These helpers put every method on a common x-axis instead.

    operating_curve(build_T, S_test, y_test, alphas, ...)
        sweeps alpha and records (deferral rate, minority coverage, low-support minority
        coverage, set size, singleton error on the minority, screening cost).

    match_at(curve, value, by="full_rate")
        linearly interpolates a method's curve at a given deferral rate (or set size, or
        coverage), i.e. "what does this method achieve when it defers as often as Mondrian
        does at alpha = 0.1?".

    compare_at_matched(curves, reference, by, at_alpha)
        one row per method: its metrics interpolated to the reference method's operating
        point.  `achievable` is False when the requested point is outside the method's
        sweep range, which is itself informative (a degenerate method may never reach a low
        deferral rate at any alpha).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("cov_minority", "cov_min_bin0", "cov_majority", "coverage", "set_size", "full_rate",
           "singleton_rate", "singleton_error_minority", "cost")


def point_metrics(sets, y_test, bin_test, minority=0, c_fp=1.0, c_fn=2.0, c_defer=0.3):
    hit = sets[np.arange(len(y_test)), y_test]
    size = sets.sum(1)
    single = size == 1
    pred = sets.argmax(1)
    m = y_test == minority
    low = (bin_test == 0) & m
    fp = single & (pred != minority) & m          # advanced a true BBB- compound
    fn = single & (pred == minority) & (~m)       # discarded a true BBB+ compound
    return {"coverage": float(hit.mean()), "cov_minority": float(hit[m].mean()) if m.any() else np.nan,
            "cov_majority": float(hit[~m].mean()) if (~m).any() else np.nan,
            "cov_min_bin0": float(hit[low].mean()) if low.any() else np.nan,
            "set_size": float(size.mean()), "full_rate": float((size == sets.shape[1]).mean()),
            "singleton_rate": float(single.mean()),
            "singleton_error_minority": float((~hit[single & m]).mean()) if (single & m).any() else np.nan,
            "cost": float((c_fp * fp + c_fn * fn + c_defer * (~single)).mean())}


def operating_curve(build_sets, alphas, y_test, bin_test, minority=0):
    """build_sets(alpha) -> boolean (n_test, K) prediction-set matrix."""
    rows = []
    for a in alphas:
        rows.append({"alpha": float(a), **point_metrics(build_sets(a), y_test, bin_test, minority)})
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def match_at(curve: pd.DataFrame, value: float, by: str = "full_rate"):
    """Interpolate every metric of `curve` at the alpha where `by` equals `value`.
    Returns (dict of metrics, achievable).  `by` must be monotone in alpha up to noise; we
    sort by `by` and interpolate, which is robust to small non-monotonicity."""
    c = curve.dropna(subset=[by]).sort_values(by)
    x = c[by].values
    if len(c) < 2 or value < x.min() - 1e-12 or value > x.max() + 1e-12:
        return {m: np.nan for m in METRICS if m in c}, False
    out = {}
    for m in METRICS:
        if m in c:
            out[m] = float(np.interp(value, x, c[m].values))
    out["alpha_at_match"] = float(np.interp(value, x, c["alpha"].values))
    return out, True


def compare_at_matched(curves: dict, reference: str, at_alpha: float = 0.10, by: str = "full_rate"):
    """curves: {method_name: operating_curve DataFrame}.  Every method is evaluated at the
    operating point the reference method reaches at `at_alpha`."""
    ref = curves[reference]
    i = int(np.argmin(np.abs(ref.alpha.values - at_alpha)))
    target = float(ref.iloc[i][by])
    rows = []
    for name, c in curves.items():
        met, ok = match_at(c, target, by)
        rows.append({"method": name, "matched_on": by, "target": target, "achievable": ok,
                     "alpha_nominal": at_alpha, **met})
    return pd.DataFrame(rows)
