"""Non-conformal and decision baselines (request items A12, A14, B15-B17).

All functions use VAL (for fitting calibrators of probabilities) and CAL (for Venn-Abers and
conformal selection) and evaluate on TEST.  No test label is used except in the metrics.

  temperature_scale / platt / isotonic   probability recalibration fitted on VAL (B17)
  venn_abers                             inductive Venn-Abers predictor on CAL (A12; Vovk & Petej 2014)
  selective_metrics                      accept-if-confident (selective classification) metrics
  ad_threshold                           applicability-domain rule: accept iff max Tanimoto >= tau (B15)
  error_detection                        AUROC / AURC of an uncertainty score for detecting errors (B16)
  conformal_selection                    BH on conformal p-values, FDR control (A14; Jin & Candes 2023)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def temperature_scale(p_val, y_val, p_apply):
    from scipy.optimize import minimize_scalar
    z = _logit(p_val[:, 1])

    def nll(logT):
        q = 1 / (1 + np.exp(-z / np.exp(logT)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))
    T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    q = 1 / (1 + np.exp(-_logit(p_apply[:, 1]) / T))
    return np.column_stack([1 - q, q]), T


def platt(p_val, y_val, p_apply):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e4).fit(_logit(p_val[:, 1])[:, None], y_val)
    q = lr.predict_proba(_logit(p_apply[:, 1])[:, None])[:, 1]
    return np.column_stack([1 - q, q])


def isotonic(p_val, y_val, p_apply):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val[:, 1], y_val)
    q = ir.predict(p_apply[:, 1])
    return np.column_stack([1 - q, q])


def venn_abers(p_cal, y_cal, p_apply):
    """Inductive Venn-Abers: for each test score s, fit isotonic regression on CAL plus (s, 0)
    and plus (s, 1); p0, p1 are the fitted values at s; merged p = p1 / (1 - p0 + p1)."""
    from sklearn.isotonic import IsotonicRegression
    s_cal = p_cal[:, 1]
    out = np.empty((len(p_apply), 3))
    for i, s in enumerate(p_apply[:, 1]):
        xs = np.append(s_cal, s)
        ps = []
        for lab in (0, 1):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(xs, np.append(y_cal, lab))
            ps.append(float(ir.predict([s])[0]))
        p0, p1 = ps
        out[i] = (p0, p1, p1 / (1 - p0 + p1) if (1 - p0 + p1) > 0 else 0.5)
    return out


def selective_metrics(probs, y, accept, minority=0):
    pred = probs.argmax(1)
    acc = accept.astype(bool)
    m = y == minority
    return {"acceptance": acc.mean(),
            "acc_on_accepted": (pred[acc] == y[acc]).mean() if acc.any() else np.nan,
            "minority_acceptance": acc[m].mean() if m.any() else np.nan,
            "minority_acc_on_accepted": (pred[acc & m] == y[acc & m]).mean() if (acc & m).any() else np.nan,
            "risk_on_accepted": 1 - ((pred[acc] == y[acc]).mean() if acc.any() else np.nan)}


def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); a = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = (conf > lo) & (conf <= hi)
        if mm.any():
            e += mm.mean() * abs(a[mm].mean() - conf[mm].mean())
    return float(e)


def classwise_ece(probs, y, minority=0, n_bins=10):
    m = y == minority
    return ece(probs[m], y[m], n_bins) if m.sum() > 5 else np.nan


def error_detection(unc, probs, y):
    """AUROC of an uncertainty score for detecting misclassifications and the area under the
    risk-coverage curve (lower is better)."""
    err = (probs.argmax(1) != y).astype(int)
    au = roc_auc_score(err, unc) if 0 < err.sum() < len(err) else np.nan
    o = np.argsort(unc)
    risk = np.cumsum(err[o]) / np.arange(1, len(o) + 1)
    return {"err_auroc": au, "aurc": float(risk.mean())}


def conformal_pvalues(v_cal_null, v_test):
    """One-sided conformal p-values, larger v = more evidence AGAINST the null.
    p_j = (1 + #{i null: v_i >= v_j}) / (n0 + 1)."""
    v0 = np.sort(v_cal_null)
    ge = len(v0) - np.searchsorted(v0, v_test, side="left")
    return (1 + ge) / (len(v0) + 1)


def benjamini_hochberg(p, q):
    m = len(p); o = np.argsort(p)
    thr = q * np.arange(1, m + 1) / m
    ok = p[o] <= thr
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    sel = np.zeros(m, bool); sel[o[:k]] = True
    return sel


def conformal_selection(p_cal, y_cal, p_test, y_test, target=1, q=0.1, strata=None):
    """Select test molecules predicted to be `target` (e.g. BBB+) with FDR <= q.
    Nulls are calibration molecules whose label is NOT target (Jin & Candes 2023)."""
    v_cal = p_cal[:, target]; v_test = p_test[:, target]
    pv = conformal_pvalues(v_cal[y_cal != target], v_test)
    sel = benjamini_hochberg(pv, q)
    out = {"n_selected": int(sel.sum()),
           "fdr": float((y_test[sel] != target).mean()) if sel.any() else 0.0,
           "power": float(sel[y_test == target].mean()) if (y_test == target).any() else np.nan}
    if strata is not None:
        for b in np.unique(strata):
            mb = strata == b
            s = sel & mb
            out[f"fdr_bin{b}"] = float((y_test[s] != target).mean()) if s.any() else np.nan
            out[f"n_selected_bin{b}"] = int(s.sum())
            out[f"power_bin{b}"] = float(s[(y_test == target)].sum() / max(1, ((y_test == target) & mb).sum()))
    return out
