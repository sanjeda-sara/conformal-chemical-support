"""Conditional-coverage diagnostics beyond per-bin coverage (request items E40, E41).

worst_slab_coverage   Cauchois, Gupta & Duchi (2021) / Romano et al. (2020): minimum coverage
                      over slabs {a <= v'x <= b} of probability mass >= delta, for random
                      directions v in a fixed feature space (here standardized RDKit descriptors).
size_stratified       coverage within each set-size group (Angelopoulos et al. 2021).
tursunbadalov_metrics MCG, CII, acceptance rate and accuracy-on-accepted as defined in
                      Tursunbadalov & Tursunbadalov (arXiv 2607.06605):
                        MCG = (1 - alpha) - cov_min,  CII = |cov_maj - cov_min|,
                        acceptance = fraction of singleton sets.
"""
from __future__ import annotations

import numpy as np


def worst_slab_coverage(F, hit, delta=0.2, n_dirs=200, seed=0, mask=None):
    """F: (n, d) features (standardize before), hit: (n,) bool covered.  Optionally restrict
    to rows in `mask` (e.g. minority test molecules).  Returns the minimum coverage over
    slabs that contain at least ceil(delta * n) points, searched over n_dirs random
    directions and contiguous windows of the sorted projections."""
    if mask is not None:
        F, hit = F[mask], hit[mask]
    n = len(hit)
    if n < 10:
        return np.nan
    L = max(5, int(np.ceil(delta * n)))
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_dirs, F.shape[1])); V /= np.linalg.norm(V, axis=1, keepdims=True)
    worst = 1.0
    h = hit.astype(float)
    for v in V:
        o = np.argsort(F @ v)
        c = np.concatenate([[0.0], np.cumsum(h[o])])
        for length in (L, int(1.5 * L), 2 * L):
            if length > n:
                continue
            win = (c[length:] - c[:-length]) / length
            worst = min(worst, float(win.min()))
    return worst


def size_stratified(sets, y):
    size = sets.sum(1); hit = sets[np.arange(len(y)), y]
    out = {}
    for s in range(sets.shape[1] + 1):
        m = size == s
        out[f"cov_size{s}"] = hit[m].mean() if m.sum() >= 10 else np.nan
        out[f"n_size{s}"] = int(m.sum())
    vals = [v for k, v in out.items() if k.startswith("cov_size") and not np.isnan(v)]
    out["ssc_min"] = min(vals) if vals else np.nan
    return out


def tursunbadalov_metrics(sets, y, alpha, minority=0):
    size = sets.sum(1); hit = sets[np.arange(len(y)), y]
    m = y == minority
    cov_min = hit[m].mean() if m.any() else np.nan
    cov_maj = hit[~m].mean() if (~m).any() else np.nan
    acc = size == 1
    return {"mcg": (1 - alpha) - cov_min, "cii": abs(cov_maj - cov_min),
            "acceptance_rate": acc.mean(),
            "acc_on_accepted": hit[acc].mean() if acc.any() else np.nan,
            "minority_acceptance_rate": acc[m].mean() if m.any() else np.nan,
            "minority_acc_on_accepted": hit[acc & m].mean() if (acc & m).any() else np.nan}
