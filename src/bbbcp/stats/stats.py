"""Statistics: binomial intervals, paired comparisons, multiplicity correction."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    a = 1 - conf
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(1 - (1 - conf) / 2, len(x) - 1)
    return (m, m - t * se, m + t * se)


def paired_compare(a, b, n_boot=10000, seed=0):
    """Paired difference a - b over matched splits: mean, bootstrap 95% CI,
    Wilcoxon signed-rank p, and Cohen's d_z."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); d = a[ok] - b[ok]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    try:
        p = stats.wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = np.nan
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    return {"n_pairs": int(len(d)), "mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "wilcoxon_p": float(p), "cohens_dz": float(dz)}


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i]); adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap_mean(values, clusters, n_boot=2000, seed=0):
    """Scaffold-cluster bootstrap CI of a per-molecule mean (resample whole scaffolds)."""
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values); cnts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return (float(values.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))
