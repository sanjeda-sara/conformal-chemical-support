"""Ablation calibrators (RESEARCH_DESIGN.md section 7; request items E30-E33, A13-A15).

Each wraps an existing calibrator and swaps the stratum assignment (bin_cal / bin_test) for
an alternative taxonomy that is a fixed function of x (so Mondrian validity is preserved),
or changes one design decision of SABM to show what it buys.

Strata keys provided in ctx by evaluate.py (all label-free):
    bins_b2_*, bins_b5_*        support bins with 2 / 5 quantile bins (edges from VAL)
    bins_caledges_*             support tertiles with edges from CAL x-values (A32)
    bins_top5_*                 tertiles of mean top-5 Tanimoto to TRAIN (A31)
    bins_maccs_*                tertiles of max MACCS-key Tanimoto to TRAIN (A31)
    bins_kmeans3_*              k-means (k=3) chemotype of ECFP-SVD fitted on TRAIN only (A33)
    bins_entropy_*              tertiles of predictive entropy (edges from VAL) (A33)
    bins_predlabel_*            predicted label argmax p (A33)
    tpsa_bin_*                  textbook TPSA bins < 60, 60-90, > 90 (A33)
    bins_random_*               uniformly random tertile, independent of x (A13 negative control)
"""
from __future__ import annotations

import numpy as np

from bbbcp.conformal.calibrators import (_partition_bins, conformal_quantile, mondrian_x,
                                         n_required, sabm)


def with_strata(fn, key):
    def wrapped(S_cal_true, y_cal, S_test, alpha, **ctx):
        ctx = dict(ctx)
        ctx["bin_cal"], ctx["bin_test"] = ctx[f"{key}_cal"], ctx[f"{key}_test"]
        return fn(S_cal_true, y_cal, S_test, alpha, **ctx)
    wrapped.__name__ = f"{fn.__name__}[{key}]"
    return wrapped


def sabm_calcount(S_cal_true, y_cal, S_test, alpha, bin_cal=None, bin_test=None, n_req=None, **_):
    """A14: SABM with the partition computed from CALIBRATION counts only (test point not
    added).  Breaks the exact bag symmetry by one count; measures whether it matters."""
    K = S_test.shape[1]
    n_req = n_required(alpha) if n_req is None else n_req
    nb = int(max(bin_cal.max(), bin_test.max()) + 1)
    T = np.empty(S_test.shape)
    for c in range(K):
        mc = y_cal == c
        gid = _partition_bins(np.bincount(bin_cal[mc], minlength=nb), n_req)
        for b in range(nb):
            rows = bin_test == b
            members = np.isin(bin_cal, np.where(gid == gid[b])[0]) & mc
            T[rows, c] = conformal_quantile(S_cal_true[members], alpha)
    return T


def sabm_pooled(S_cal_true, y_cal, S_test, alpha, bin_cal=None, bin_test=None, n_req=None, **_):
    """A15: the 'obvious heuristic'.  Cells with fewer than n_req calibration points of their
    class fall back to a threshold pooled over BOTH classes in that support bin.  Remark 1
    shows this can break class-conditional validity."""
    K = S_test.shape[1]
    n_req = n_required(alpha) if n_req is None else n_req
    nb = int(max(bin_cal.max(), bin_test.max()) + 1)
    T = np.empty(S_test.shape)
    for c in range(K):
        for b in range(nb):
            rows = bin_test == b
            m = (y_cal == c) & (bin_cal == b)
            if m.sum() + 1 >= n_req:
                T[rows, c] = conformal_quantile(S_cal_true[m], alpha)
            else:
                T[rows, c] = conformal_quantile(S_cal_true[bin_cal == b], alpha)
    return T


STRATA_KEYS = ["bins_b2", "bins_b5", "bins_caledges", "bins_top5", "bins_maccs", "bins_kmeans3",
               "bins_entropy", "bins_predlabel", "tpsa_bin", "bins_random"]

ABLATIONS = {
    "mondrian_x_b2": with_strata(mondrian_x, "bins_b2"),
    "mondrian_x_b5": with_strata(mondrian_x, "bins_b5"),
    "sabm_b2": with_strata(sabm, "bins_b2"),
    "sabm_b5": with_strata(sabm, "bins_b5"),
    "mondrian_x_caledges": with_strata(mondrian_x, "bins_caledges"),
    "mondrian_x_top5": with_strata(mondrian_x, "bins_top5"),
    "mondrian_x_maccs": with_strata(mondrian_x, "bins_maccs"),
    "sabm_top5": with_strata(sabm, "bins_top5"),
    "mondrian_kmeans3": with_strata(mondrian_x, "bins_kmeans3"),
    "mondrian_entropy": with_strata(mondrian_x, "bins_entropy"),
    "mondrian_predlabel": with_strata(mondrian_x, "bins_predlabel"),
    "mondrian_tpsa": with_strata(mondrian_x, "tpsa_bin"),
    "sabm_tpsa": with_strata(sabm, "tpsa_bin"),
    "mondrian_x_random": with_strata(mondrian_x, "bins_random"),
    "sabm_calcount": sabm_calcount,
    "sabm_pooled": sabm_pooled,
}
