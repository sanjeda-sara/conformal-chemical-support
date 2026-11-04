"""Split-conformal calibrators.  Every calibrator returns a threshold matrix
T (n_test, K): label y is in the set of test point i iff S_test[i, y] <= T[i, y].

What each calibrator controls (under exchangeability of calibration and test):
  marginal        P(Y in C(X)) >= 1 - alpha                         [Vovk et al. 2005]
  mondrian        P(Y in C(X) | Y = c) >= 1 - alpha for every class  [label-conditional]
  mondrian_pac    P_cal( coverage_c >= 1 - alpha ) >= 1 - delta      [training-conditional,
                  Vovk 2012; same idea as SSBC (Zwart 2026)]
  mondrian_x      coverage >= 1 - alpha within every (class, support-bin) cell; the bin
                  is a function of x only (similarity to the TRAINING set, which is
                  independent of cal/test), so it is a valid Mondrian taxonomy.
  sabm            support-aware adaptive-partition Mondrian (proposed): class x support-bin
                  cells, adjacent bins merged until each group has n_req calibration points
                  (n_req fixed in advance by the Beta law).  The partition depends only on
                  category COUNTS of the bag {cal + test point}, a symmetric function of the
                  n+1 exchangeable points, so class- and group-conditional coverage hold
                  exactly.  See docs/THEORY_NOTES.md.
  weighted        weighted CP for covariate shift (Tibshirani et al. 2019) with density
                  ratio from a cal-vs-test probabilistic classifier (CoDrug-style
                  approximation, NOT a reproduction of CoDrug or KMM-CP).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.stats import beta as beta_dist


# ----------------------------------------------------------------------------- quantiles
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-th smallest calibration score with k = ceil((n+1)(1-alpha)).
    Returns +inf when k > n (the set must then contain every label).  The legacy code
    clipped the level at 1.0 and returned the max score instead, which is anti-
    conservative for small n (e.g. n_c = 10 at alpha = 0.05)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return float("inf") if k > n or n == 0 else float(s[k - 1])


@lru_cache(maxsize=4096)
def pac_rank(n: int, alpha: float, delta: float) -> int | None:
    """Smallest rank k such that P(Beta(k, n+1-k) >= 1-alpha) >= 1-delta.
    Coverage of the k-th order statistic threshold, conditional on the calibration
    set, is Beta(k, n+1-k) distributed (continuous scores).  None -> infinite threshold."""
    if n < 1:
        return None
    ks = np.arange(1, n + 1)
    ok = beta_dist.sf(1 - alpha, ks, n + 1 - ks) >= 1 - delta
    return int(ks[ok][0]) if ok.any() else None


@lru_cache(maxsize=256)
def n_required(alpha: float, eps: float | None = None, delta: float = 0.10, n_max: int = 3000) -> int:
    """Pre-specified support rule (data independent): smallest n0 such that for EVERY
    n >= n0 the standard split-conformal threshold (k = ceil((n+1)(1-alpha))) satisfies
    P(coverage < 1 - alpha - eps) <= delta under the Beta(k, n+1-k) law.  (The plain
    'first n' is not monotone because of the ceiling.)  Used by SABM.
    Frozen 2026-10-09 (before confirmatory seeds 100-119): eps defaults to alpha (relative
    tolerance: realized miscoverage at most 2*alpha with prob >= 1 - delta), giving
    n_req = 22 / 11 / 5 at alpha = 0.05 / 0.10 / 0.20.  The original absolute rule
    (eps = 0.05; 22 / 60 / 105) is kept as the pre-registered sensitivity method sabm_abs05."""
    eps = alpha if eps is None else eps
    ns = np.arange(1, n_max + 1)
    ks = np.ceil((ns + 1) * (1 - alpha) - 1e-12).astype(int)
    valid = ks <= ns
    cdf = np.ones(len(ns))
    cdf[valid] = beta_dist.cdf(1 - alpha - eps, ks[valid], ns[valid] + 1 - ks[valid])
    ok = np.concatenate([[False], valid & (cdf <= delta)])
    bad = np.where(~ok[1:])[0]
    return int(bad.max() + 2) if len(bad) else 1


def beta_coverage_params(n: int, alpha: float):
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return (k, n + 1 - k) if k <= n else None


# ----------------------------------------------------------------------------- calibrators
def marginal(S_cal_true, y_cal, S_test, alpha, **_):
    q = conformal_quantile(S_cal_true, alpha)
    return np.full(S_test.shape, q)


def mondrian(S_cal_true, y_cal, S_test, alpha, **_):
    K = S_test.shape[1]
    T = np.empty(S_test.shape)
    for c in range(K):
        T[:, c] = conformal_quantile(S_cal_true[y_cal == c], alpha)
    return T


def mondrian_pac(S_cal_true, y_cal, S_test, alpha, delta=0.10, **_):
    K = S_test.shape[1]
    T = np.empty(S_test.shape)
    for c in range(K):
        s = np.sort(S_cal_true[y_cal == c])
        k = pac_rank(len(s), alpha, delta)
        T[:, c] = np.inf if k is None else s[k - 1]
    return T


def mondrian_x(S_cal_true, y_cal, S_test, alpha, bin_cal=None, bin_test=None, **_):
    K = S_test.shape[1]
    T = np.empty(S_test.shape)
    for c in range(K):
        for b in np.unique(bin_test):
            m = (y_cal == c) & (bin_cal == b)
            T[bin_test == b, c] = conformal_quantile(S_cal_true[m], alpha)
    return T


def _partition_bins(counts: np.ndarray, n_req: int) -> np.ndarray:
    """Deterministic adjacent-bin merging.  Walk bins from low to high similarity,
    accumulating until the group count reaches n_req; a deficient last group is merged
    into the previous one.  Returns group id per bin.  Depends ONLY on the counts."""
    gid = np.zeros(len(counts), int); g = 0; acc = 0
    for b, cnt in enumerate(counts):
        gid[b] = g; acc += cnt
        if acc >= n_req and b < len(counts) - 1:
            g += 1; acc = 0
    if acc < n_req and g > 0 and gid[-1] == g:
        gid[gid == g] = g - 1
    return gid


def sabm(S_cal_true, y_cal, S_test, alpha, bin_cal=None, bin_test=None, n_req=None, n_bins=None,
         **_):
    """Support-aware adaptive-partition Mondrian (proposed).

    Taxonomy: (class, chemical-support bin).  For every class c the support bins are merged
    into groups by `_partition_bins`, using the category counts of the BAG {calibration
    points} + {test point with candidate label c}.  Because the partition is a function of
    the unordered bag only, every point of a group would be assigned the same group if it
    were the test point, so Mondrian validity holds conditional on the realized partition:
        P(Y in C(X) | Y = c, group(X) = g) >= 1 - alpha   for every group g,
    and hence per-class coverage >= 1 - alpha.  It never falls back to a pooled (marginal)
    threshold, which would break the class-conditional guarantee.  n_req is fixed in
    advance by the Beta-law rule `n_required(alpha)`."""
    K = S_test.shape[1]
    n_req = n_required(alpha) if n_req is None else n_req
    nb = int(max(bin_cal.max(), bin_test.max()) + 1) if n_bins is None else n_bins
    T = np.empty(S_test.shape)
    level = np.empty(S_test.shape, dtype=np.int8)  # number of bins pooled in the group used
    for c in range(K):
        mc = y_cal == c
        base = np.bincount(bin_cal[mc], minlength=nb)
        for b in range(nb):
            rows = bin_test == b
            if not rows.any():
                continue
            cnt = base.copy(); cnt[b] += 1           # the test point with candidate label c
            gid = _partition_bins(cnt, n_req)
            members = np.isin(bin_cal, np.where(gid == gid[b])[0]) & mc
            T[rows, c] = conformal_quantile(S_cal_true[members], alpha)
            level[rows, c] = int((gid == gid[b]).sum())
    sabm.last_level = level
    return T


def weighted_quantile(scores, w_cal, w_test, alpha):
    """Weighted conformal quantile with the test point's weight placed at +inf."""
    order = np.argsort(scores)
    s, w = scores[order], w_cal[order]
    tot = w.sum() + w_test
    cdf = np.cumsum(w) / tot
    idx = np.searchsorted(cdf, 1 - alpha - 1e-12)
    return float("inf") if idx >= len(s) else float(s[idx])


def weighted(S_cal_true, y_cal, S_test, alpha, w_cal=None, w_test=None, by_class=False, **_):
    K = S_test.shape[1]
    T = np.empty(S_test.shape)
    for i in range(S_test.shape[0]):
        for c in range(K):
            m = (y_cal == c) if by_class else np.ones(len(y_cal), bool)
            T[i, c] = weighted_quantile(S_cal_true[m], w_cal[m], w_test[i], alpha)
    return T


def density_ratio(X_cal, X_test, seed=0, clip=(0.05, 20.0)):
    """w(x) = p_test(x)/p_cal(x) from a logistic-regression domain classifier fitted on
    UNLABELED cal and test features (transductive; uses no labels)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    Z = np.vstack([X_cal, X_test]).astype(np.float32)
    Z = TruncatedSVD(n_components=min(64, Z.shape[1] - 1), random_state=seed).fit_transform(Z)
    d = np.r_[np.zeros(len(X_cal)), np.ones(len(X_test))]
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z, d)
    p = clf.predict_proba(Z)[:, 1]
    r = (p / (1 - p)) * (len(X_cal) / len(X_test))
    r = np.clip(r, *clip)
    return r[: len(X_cal)], r[len(X_cal):]


CALIBRATORS = {"marginal": marginal, "mondrian": mondrian, "mondrian_pac": mondrian_pac,
               "mondrian_x": mondrian_x, "sabm": sabm,
               "sabm_abs05": lambda *a, **k: sabm(*a, n_req=n_required(a[3], 0.05), **k),
               "weighted": weighted,
               "weighted_mondrian": lambda *a, **k: weighted(*a, by_class=True, **k)}


def predict_sets(S_test: np.ndarray, T: np.ndarray) -> np.ndarray:
    return S_test <= T
