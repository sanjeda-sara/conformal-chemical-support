"""State-of-the-art conformal baselines (RESEARCH_DESIGN.md section 5.2; item list A1-A10).

Every function follows the calibrator signature used in calibrators.py:
    f(S_cal_true, y_cal, S_test, alpha, **ctx) -> T  (n_test, K)
and label y is in the set of test point i iff S_test[i, y] <= T[i, y].  A threshold of -inf
excludes a label (used by RC3P's rank test), +inf includes it.

ctx keys provided by evaluate.py (all computed WITHOUT test labels):
    bin_cal, bin_test      support tertile (edges from VAL)
    sim_cal, sim_test      max Tanimoto to TRAIN (continuous)
    P_cal, P_test          model probabilities (n, K)
    prior                  class prevalence estimated on VAL labels (K,)
    Tc_tan                 Tanimoto matrix TEST x CAL (ECFP4)
    Z_cal, Z_test          low-dimensional ECFP features (SVD fitted on CAL+TEST, unlabeled)
    tpsa_bin_cal/test      textbook TPSA bins
    kmm_*, kde_*, bbse_*   precomputed weights (see weights_for_file)
    score_kind, rng

Faithfulness notes (be explicit in the paper about these):
  * mondrian_ssbc     = Beta-law rank inversion with (alpha, delta); we implement it as the
                        Vovk (2012) training-conditional rank, which is how we read SSBC
                        (Zwart 2026).  [CHECK against the SSBC paper before claiming identity]
  * clustered_cells   = clustered CP (Ding et al. 2023) applied to class x support cells
                        (with only 2 labels, class-level clustering is degenerate).
  * condqr_sim        = split conditional calibration with a quantile regression on
                        [1, s, s^2] (s = support), the unaugmented approximation of
                        Gibbs, Cherian & Candes (2025); level inflated by d/(n+1).
  * kandinsky_qr      = the same QR machinery on OVERLAPPING group indicators
                        (support bins and TPSA bins) per class, our instantiation of
                        Kandinsky CP (Bairaktari et al. 2025) with indicator groups.
  * rc3p              = Shi et al. (2024) Eq. (3)/(7) with empirical top-k error.
  * interp_q*         = INTERP-Q (Ding, Fermanian & Salmon 2026), tau in {0.5, 0.9}.
  * pas               = prevalence-adjusted softmax score with standard CP (same paper).
  * fuzzy_raw         = raw fuzzy classwise CP (same paper, App. A), Gaussian kernel on the
                        prevalence map, without the reconformalization step.
  * macro_cov(_pas)   = label-weighted CP with the max_k w(k)/N_k correction
                        (Bhattacharyya, Ding & Barber 2026, Thm 1 / Cor 1).
  * lcp_tanimoto      = localized CP with a Tanimoto kernel (Guan 2023 style weighting,
                        test weight 1, no exactness correction) within each class.
  * rlcp_sim          = randomly-localized CP (Hore & Barber) on the support feature,
                        Gaussian kernel, within each class: exact class-conditional validity.
  * kmm_*, skmm_*     = our kernel-mean-matching weights (RBF on ECFP-SVD features, B = 30)
                        and a selective variant in the spirit of KMM-CP (Laghuvarapu 2026);
                        NOT the official code.  kde_weighted is a CoDrug-style KDE ratio.
  * labelshift_bbse   = weighted CP with BBSE label-shift weights (Lipton et al. 2018;
                        Podkopaev & Ramdas 2021).
"""
from __future__ import annotations

import math

import numpy as np

from bbbcp.conformal.calibrators import (conformal_quantile, mondrian_pac, n_required,
                                         weighted_quantile)


# ----------------------------------------------------------------------------- helpers
def _wq_no_test(scores, w, level):
    """min{t : sum_i w_i 1{s_i <= t} >= level} with normalized weights; +inf if never."""
    if len(scores) == 0:
        return float("inf")
    o = np.argsort(scores)
    s, ww = np.asarray(scores)[o], np.asarray(w, float)[o]
    cdf = np.cumsum(ww) / ww.sum()
    i = np.searchsorted(cdf, level - 1e-12)
    return float("inf") if i >= len(s) else float(s[i])


def _upper_score(S_cal_true, S_test):
    return float(max(np.max(S_cal_true), np.max(S_test[np.isfinite(S_test)])))


def _ranks(P):
    """1-based rank of each label by decreasing probability, ties by label index."""
    order = np.lexsort((np.broadcast_to(np.arange(P.shape[1]), P.shape), -P), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(P.shape[1])[None, :].repeat(len(P), 0), axis=1)
    return rank + 1


# ----------------------------------------------------------------------------- A1 SSBC
def mondrian_ssbc(S_cal_true, y_cal, S_test, alpha, delta=0.10, **ctx):
    return mondrian_pac(S_cal_true, y_cal, S_test, alpha, delta=delta)


# ----------------------------------------------------------------------------- A2 clustered
def clustered_cells(S_cal_true, y_cal, S_test, alpha, bin_cal=None, bin_test=None, rng=None,
                    gamma=0.3, n_clusters=2, min_cluster_pts=5, **ctx):
    """Clustered CP over units u = (class, support bin).  A fraction gamma of each unit's
    calibration points embeds the unit by score quantiles; k-means groups units; the rest
    calibrates one threshold per cluster.  Units too small to embed go to a null cluster
    calibrated marginally (as in Ding et al. 2023)."""
    from sklearn.cluster import KMeans
    rng = rng or np.random.default_rng(0)
    K = S_test.shape[1]
    nb = int(max(bin_cal.max(), bin_test.max()) + 1)
    unit_cal = y_cal * nb + bin_cal
    n_units = K * nb
    is_clust = np.zeros(len(y_cal), bool)
    for u in range(n_units):
        idx = np.where(unit_cal == u)[0]
        if len(idx):
            take = rng.choice(idx, int(round(gamma * len(idx))), replace=False)
            is_clust[take] = True
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1 - alpha]
    emb, ok_units = [], []
    for u in range(n_units):
        s = S_cal_true[(unit_cal == u) & is_clust]
        if len(s) >= min_cluster_pts:
            emb.append(np.quantile(s, levels)); ok_units.append(u)
    assign = np.full(n_units, -1)
    if len(ok_units) >= n_clusters:
        lab = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(np.array(emb))
        assign[np.array(ok_units)] = lab
    calib = ~is_clust
    q_null = conformal_quantile(S_cal_true[calib], alpha)
    q_cl = {g: conformal_quantile(S_cal_true[calib & np.isin(unit_cal, np.where(assign == g)[0])], alpha)
            for g in set(assign.tolist()) if g >= 0}
    T = np.empty(S_test.shape)
    for c in range(K):
        for b in range(nb):
            rows = bin_test == b
            g = assign[c * nb + b]
            T[rows, c] = q_cl[g] if g >= 0 else q_null
    return T


# ----------------------------------------------------------------------------- A3/A5 quantile regression
def _qr_threshold(Phi_cal, s_cal, Phi_test, alpha):
    from sklearn.linear_model import QuantileRegressor
    n, d = Phi_cal.shape
    if n == 0:
        return np.full(len(Phi_test), np.inf)
    tau = 1 - alpha + d / (n + 1)
    if tau >= 1:
        return np.full(len(Phi_test), np.inf)
    qr = QuantileRegressor(quantile=tau, alpha=0.0, fit_intercept=False, solver="highs")
    qr.fit(Phi_cal, s_cal)
    return Phi_test @ qr.coef_


def condqr_sim(S_cal_true, y_cal, S_test, alpha, sim_cal=None, sim_test=None, **ctx):
    feat = lambda s: np.column_stack([np.ones_like(s), s, s ** 2])
    T = np.empty(S_test.shape)
    for c in range(S_test.shape[1]):
        m = y_cal == c
        T[:, c] = _qr_threshold(feat(sim_cal[m]), S_cal_true[m], feat(sim_test), alpha)
    return T


def kandinsky_qr(S_cal_true, y_cal, S_test, alpha, bin_cal=None, bin_test=None,
                 tpsa_bin_cal=None, tpsa_bin_test=None, **ctx):
    nb = int(max(bin_cal.max(), bin_test.max()) + 1)

    def feat(b, t):
        A = np.eye(nb)[b]                      # support bins (sum to 1: plays the intercept)
        B = np.eye(3)[t][:, 1:]                # TPSA bins, one column dropped (identifiability)
        return np.hstack([A, B])
    T = np.empty(S_test.shape)
    for c in range(S_test.shape[1]):
        m = y_cal == c
        T[:, c] = _qr_threshold(feat(bin_cal[m], tpsa_bin_cal[m]), S_cal_true[m],
                                feat(bin_test, tpsa_bin_test), alpha)
    return T


# ----------------------------------------------------------------------------- A4 localized
def lcp_tanimoto(S_cal_true, y_cal, S_test, alpha, Tc_tan=None, h=0.5, **ctx):
    """Kernel w = exp(-(1 - Tanimoto)/h); test weight K(x, x) = 1.  h = 0.5 fixed a priori:
    with h = 0.2 the test point's own weight dominates (typical neighbour Tanimoto 0.3-0.5
    gives weights 0.03-0.08) and every threshold becomes +inf (full sets)."""
    T = np.empty(S_test.shape)
    for c in range(S_test.shape[1]):
        m = y_cal == c
        s = S_cal_true[m]
        W = np.exp(-(1.0 - Tc_tan[:, m]) / h)
        for i in range(S_test.shape[0]):
            T[i, c] = weighted_quantile(s, W[i], 1.0, alpha)
    return T


def lcp_knn(S_cal_true, y_cal, S_test, alpha, Tc_tan=None, k_frac=0.3, k_min=20, **ctx):
    """k-nearest-neighbour localized CP within class: the threshold for test x uses the k most
    Tanimoto-similar class-c calibration molecules (k = max(k_min, k_frac * n_c)).  Not exactly
    valid (neighbourhoods are not symmetric); reported as a heuristic localized baseline."""
    T = np.empty(S_test.shape)
    for c in range(S_test.shape[1]):
        m = np.where(y_cal == c)[0]
        k = int(min(len(m), max(k_min, k_frac * len(m))))
        nn = np.argsort(-Tc_tan[:, m], axis=1)[:, :k]
        for i in range(S_test.shape[0]):
            T[i, c] = conformal_quantile(S_cal_true[m[nn[i]]], alpha)
    return T


def rlcp_sim(S_cal_true, y_cal, S_test, alpha, sim_cal=None, sim_test=None, h=0.1, rng=None, **ctx):
    rng = rng or np.random.default_rng(0)
    xt = sim_test + h * rng.normal(size=len(sim_test))
    T = np.empty(S_test.shape)
    for c in range(S_test.shape[1]):
        m = y_cal == c
        s, sc = S_cal_true[m], sim_cal[m]
        for i in range(S_test.shape[0]):
            w = np.exp(-0.5 * ((sc - xt[i]) / h) ** 2)
            wt = math.exp(-0.5 * ((sim_test[i] - xt[i]) / h) ** 2)
            T[i, c] = weighted_quantile(s, w, wt, alpha)
    return T


# ----------------------------------------------------------------------------- A6 RC3P
def rc3p(S_cal_true, y_cal, S_test, alpha, P_cal=None, P_test=None, **ctx):
    K = S_test.shape[1]
    r_cal = _ranks(P_cal)[np.arange(len(y_cal)), y_cal]
    r_test = _ranks(P_test)
    T = np.empty(S_test.shape)
    for c in range(K):
        m = y_cal == c
        k_hat, eps = K, 0.0
        for k in range(1, K + 1):
            e = float((r_cal[m] > k).mean()) if m.any() else 1.0
            if e < alpha:
                k_hat, eps = k, e
                break
        q = conformal_quantile(S_cal_true[m], alpha - eps)
        T[:, c] = np.where(r_test[:, c] <= k_hat, q, -np.inf)
    return T


# ----------------------------------------------------------------------------- A7 INTERP-Q / fuzzy / PAS
def interp_q(S_cal_true, y_cal, S_test, alpha, tau=0.5, **ctx):
    q = conformal_quantile(S_cal_true, alpha)
    up = _upper_score(S_cal_true, S_test)
    T = np.empty(S_test.shape)
    for c in range(S_test.shape[1]):
        qc = conformal_quantile(S_cal_true[y_cal == c], alpha)
        qc = up if not np.isfinite(qc) else qc
        T[:, c] = tau * qc + (1 - tau) * q
    return T


def fuzzy_raw(S_cal_true, y_cal, S_test, alpha, prior=None, sigma=0.25, **ctx):
    K = S_test.shape[1]
    prior = np.full(K, 1 / K) if prior is None else np.asarray(prior)
    pi = prior / prior.max()
    T = np.empty(S_test.shape)
    for c in range(K):
        w = np.exp(-((pi[y_cal] - pi[c]) ** 2) / (2 * sigma ** 2))
        T[:, c] = weighted_quantile(S_cal_true, w, 1.0, alpha)
    return T


def _pas_scores(P, prior):
    return -P / np.asarray(prior)[None, :]


def pas(S_cal_true, y_cal, S_test, alpha, P_cal=None, P_test=None, prior=None, **ctx):
    """Score-free method: builds its own prevalence-adjusted softmax scores."""
    Sc = _pas_scores(P_cal, prior)[np.arange(len(y_cal)), y_cal]
    St = _pas_scores(P_test, prior)
    q = conformal_quantile(Sc, alpha)
    pas.S_override = St
    return np.full(St.shape, q)


# ----------------------------------------------------------------------------- A8 macro coverage
def _macro_threshold(s_cal, y_cal, alpha, K):
    N = np.bincount(y_cal, minlength=K).astype(float)
    wk = np.full(K, 1.0 / K)
    pos = N > 0
    delta = float(np.max(wk[pos] / N[pos]))
    a2 = alpha - delta
    if a2 <= 0:
        return float("inf")
    w = wk[y_cal] / N[y_cal]
    mass_missing = wk[~pos].sum()          # empty groups put their mass at +inf
    o = np.argsort(s_cal)
    cdf = np.cumsum(w[o])
    i = np.searchsorted(cdf, 1 - a2 - 1e-12)
    return float("inf") if (i >= len(o) or cdf[-1] + mass_missing < 1 - a2 - 1e-12) else float(s_cal[o][i])


def macro_cov(S_cal_true, y_cal, S_test, alpha, **ctx):
    return np.full(S_test.shape, _macro_threshold(S_cal_true, y_cal, alpha, S_test.shape[1]))


def macro_cov_pas(S_cal_true, y_cal, S_test, alpha, P_cal=None, P_test=None, prior=None, **ctx):
    Sc = _pas_scores(P_cal, prior)[np.arange(len(y_cal)), y_cal]
    St = _pas_scores(P_test, prior)
    macro_cov_pas.S_override = St
    return np.full(St.shape, _macro_threshold(Sc, y_cal, alpha, St.shape[1]))


# ----------------------------------------------------------------------------- A9 shift weights
def kmm_weights(Zc, Zt, B=30.0, lam=10.0, sigma=None, max_iter=500):
    """Kernel mean matching (Huang et al. 2007): min || 1/n sum w_i phi(x_i) - 1/m sum phi(z_j) ||^2
    s.t. 0 <= w <= B, mean(w) ~ 1 (quadratic penalty lam).  RBF kernel, median heuristic."""
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist
    n, m = len(Zc), len(Zt)
    if sigma is None:
        sub = np.vstack([Zc, Zt])[:: max(1, (n + m) // 500)]
        d = cdist(sub, sub); sigma = np.median(d[d > 0])
    g = 1.0 / (2 * sigma ** 2)
    Kcc = np.exp(-g * cdist(Zc, Zc, "sqeuclidean"))
    kap = np.exp(-g * cdist(Zc, Zt, "sqeuclidean")).mean(1)

    def f(w):
        Kw = Kcc @ w
        val = 0.5 * w @ Kw / n ** 2 - w @ kap / n + lam * (w.mean() - 1) ** 2
        grad = Kw / n ** 2 - kap / n + 2 * lam * (w.mean() - 1) / n
        return val, grad
    res = minimize(f, np.ones(n), jac=True, method="L-BFGS-B", bounds=[(0, B)] * n,
                   options={"maxiter": max_iter})
    return res.x, sigma


def skmm_select(Zc, Zt, w, sigma, tau=0.5, thr=0.2):
    """Selective step in the spirit of KMM-CP: keep test points whose weighted-calibration
    kernel density is at least `thr` x their test density; keep at least a fraction tau."""
    from scipy.spatial.distance import cdist
    g = 1.0 / (2 * sigma ** 2)
    dens_cal = (np.exp(-g * cdist(Zt, Zc, "sqeuclidean")) * w[None, :]).mean(1)
    dens_tst = np.exp(-g * cdist(Zt, Zt, "sqeuclidean")).mean(1)
    r = dens_cal / np.maximum(dens_tst, 1e-12)
    keep = r >= thr
    if keep.mean() < tau:
        keep = r >= np.quantile(r, 1 - tau)
    return keep


def kde_weights(Zc, Zt, clip=(0.05, 20.0)):
    from scipy.stats import gaussian_kde
    kc, kt = gaussian_kde(Zc.T), gaussian_kde(Zt.T)
    r = lambda Z: np.clip(kt(Z.T) / np.maximum(kc(Z.T), 1e-300), *clip)
    return r(Zc), r(Zt)


def bbse_weights(P_cal, y_cal, P_test, K=2, clip=(0.0, 20.0)):
    """Black-box shift estimation: C[j, k] = P_cal(yhat = j, y = k); mu[j] = P_test(yhat = j);
    w = C^{-1} mu solved by least squares; w(k) = q_test(k) / p_cal(k)."""
    yh_c, yh_t = P_cal.argmax(1), P_test.argmax(1)
    C = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            C[j, k] = np.mean((yh_c == j) & (y_cal == k))
    mu = np.bincount(yh_t, minlength=K) / len(yh_t)
    w, *_ = np.linalg.lstsq(C, mu, rcond=None)
    return np.clip(w, *clip)


def _weighted_by(S_cal_true, y_cal, S_test, alpha, w_cal, w_test, by_class):
    K = S_test.shape[1]
    T = np.empty(S_test.shape)
    for c in range(K):
        m = (y_cal == c) if by_class else np.ones(len(y_cal), bool)
        for i in range(S_test.shape[0]):
            T[i, c] = weighted_quantile(S_cal_true[m], w_cal[m], w_test[i], alpha)
    return T


def kmm_weighted(S_cal_true, y_cal, S_test, alpha, kmm_cal=None, kmm_test=None, **ctx):
    return _weighted_by(S_cal_true, y_cal, S_test, alpha, kmm_cal, kmm_test, False)


def kmm_mondrian(S_cal_true, y_cal, S_test, alpha, kmm_cal=None, kmm_test=None, **ctx):
    return _weighted_by(S_cal_true, y_cal, S_test, alpha, kmm_cal, kmm_test, True)


def skmm_mondrian(S_cal_true, y_cal, S_test, alpha, skmm_cal=None, skmm_test=None, skmm_keep=None, **ctx):
    T = _weighted_by(S_cal_true, y_cal, S_test, alpha, skmm_cal, skmm_test, True)
    skmm_mondrian.keep = skmm_keep
    return T


def kde_weighted(S_cal_true, y_cal, S_test, alpha, kde_cal=None, kde_test=None, **ctx):
    return _weighted_by(S_cal_true, y_cal, S_test, alpha, kde_cal, kde_test, False)


# ----------------------------------------------------------------------------- A10 label shift
def labelshift_bbse(S_cal_true, y_cal, S_test, alpha, bbse_w=None, **ctx):
    K = S_test.shape[1]
    wc = bbse_w[y_cal]
    T = np.empty(S_test.shape)
    for c in range(K):
        T[:, c] = weighted_quantile(S_cal_true, wc, float(bbse_w[c]), alpha)
    return T


SOTA = {
    "mondrian_ssbc": mondrian_ssbc,
    "clustered_cells": clustered_cells,
    "condqr_sim": condqr_sim,
    "kandinsky_qr": kandinsky_qr,
    "lcp_tanimoto": lcp_tanimoto,
    "lcp_knn": lcp_knn,
    "rlcp_sim": rlcp_sim,
    "rc3p": rc3p,
    "interp_q50": lambda *a, **k: interp_q(*a, tau=0.5, **k),
    "interp_q90": lambda *a, **k: interp_q(*a, tau=0.9, **k),
    "fuzzy_raw": fuzzy_raw,
    "macro_cov": macro_cov,
    "kmm_weighted": kmm_weighted,
    "kmm_mondrian": kmm_mondrian,
    "skmm_mondrian": skmm_mondrian,
    "kde_weighted": kde_weighted,
    "labelshift_bbse": labelshift_bbse,
}
SCORE_FREE = {"pas": pas, "macro_cov_pas": macro_cov_pas}
