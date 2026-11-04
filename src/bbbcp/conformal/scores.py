"""Nonconformity scores for K-class prediction sets.

All functions return an (n, K) matrix S where S[i, y] is the score of candidate
label y for example i (higher = less conforming).  Calibration uses the true-label
column; set construction uses every column.  Using ONE uniform U_i per example,
shared by all candidate labels, is the canonical randomized APS (Romano, Sesia &
Candes 2020).  The legacy MoML code drew a fresh U for every candidate label,
which is not the canonical construction (see docs/DATA_AUDIT.md, A6).

Ties: labels are ranked by probability, ties broken by label index (lower index
ranked first), deterministically.
"""
from __future__ import annotations

import numpy as np


def _rank_matrix(probs: np.ndarray):
    """order[i] = labels sorted by decreasing prob (stable on label index);
    rank[i, y] = 0-based position of label y."""
    order = np.lexsort((np.broadcast_to(np.arange(probs.shape[1]), probs.shape), -probs), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(probs.shape[1])[None, :].repeat(len(probs), 0), axis=1)
    return order, rank


def lac(probs: np.ndarray) -> np.ndarray:
    """Least-ambiguous-set / softmax score: 1 - p_y (Sadinle, Lei & Wasserman 2019)."""
    return 1.0 - probs


def aps(probs: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
    """APS score: mass of labels ranked strictly above y, plus U * p_y.
    u=None gives the non-randomized version (U = 1)."""
    n, k = probs.shape
    order, rank = _rank_matrix(probs)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum_above = np.concatenate([np.zeros((n, 1)), np.cumsum(sorted_p, axis=1)[:, :-1]], axis=1)
    above = np.take_along_axis(cum_above, rank, axis=1)  # mass strictly above label y
    uu = np.ones(n) if u is None else np.asarray(u, float)
    return above + uu[:, None] * probs


def raps(probs: np.ndarray, u: np.ndarray | None = None, lam: float = 0.1, k_reg: int = 1):
    """RAPS (Angelopoulos et al. 2021): APS + lam * max(0, rank_1based - k_reg).
    lam / k_reg must be fixed in advance or tuned on validation data, never on cal/test."""
    _, rank = _rank_matrix(probs)
    return aps(probs, u) + lam * np.maximum(0, (rank + 1) - k_reg)


def margin(probs: np.ndarray) -> np.ndarray:
    """Top-2 margin score: max_{y' != y} p_y' - p_y.  For K = 2 this is 1 - 2 p_y, a strictly
    increasing transform of LAC that is the same for every label, so every rank-based
    calibrator (marginal, Mondrian, SABM, ...) returns IDENTICAL sets to LAC
    (tests/test_extended_methods.py::test_margin_equals_lac_binary)."""
    out = np.empty_like(probs)
    for y in range(probs.shape[1]):
        other = np.delete(probs, y, axis=1).max(1)
        out[:, y] = other - probs[:, y]
    return out


def score_matrix(probs: np.ndarray, kind: str, u=None, **kw) -> np.ndarray:
    if kind == "margin":
        return margin(probs)
    if kind == "lac":
        return lac(probs)
    if kind == "aps":
        return aps(probs, u)
    if kind == "aps_det":
        return aps(probs, None)
    if kind == "raps":
        return raps(probs, u, **kw)
    raise ValueError(kind)
