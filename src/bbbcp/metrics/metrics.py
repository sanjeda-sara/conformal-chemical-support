"""Classification and prediction-set metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)


def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def classification_metrics(probs, y, minority: int = 0):
    p1 = probs[:, 1]; pred = probs.argmax(1)
    out = {"n": len(y), "n_minority": int((y == minority).sum())}
    if len(np.unique(y)) == 2:
        out["auroc"] = roc_auc_score(y, p1)
        # AUPRC with the MINORITY class as positive (the class screening cares about)
        pm = probs[:, minority]
        out["auprc_minority"] = average_precision_score((y == minority).astype(int), pm)
        out["auprc_pos"] = average_precision_score(y, p1)
    out["balanced_acc"] = balanced_accuracy_score(y, pred)
    out["sensitivity"] = float((pred[y == 1] == 1).mean()) if (y == 1).any() else np.nan
    out["specificity"] = float((pred[y == 0] == 0).mean()) if (y == 0).any() else np.nan
    out["f1_macro"] = f1_score(y, pred, average="macro")
    out["brier"] = brier_score_loss(y, p1)
    out["nll"] = log_loss(y, np.clip(probs, 1e-12, 1), labels=[0, 1])
    out["ece"] = ece(probs, y)
    return out


def set_metrics(sets: np.ndarray, y: np.ndarray, minority: int = 0):
    n, K = sets.shape
    size = sets.sum(1)
    hit = sets[np.arange(n), y]
    out = {"coverage": hit.mean(), "set_size": size.mean(),
           "singleton_rate": (size == 1).mean(), "empty_rate": (size == 0).mean(),
           "full_rate": (size == K).mean()}
    cls = []
    for c in range(K):
        m = y == c
        out[f"cov_c{c}"] = hit[m].mean() if m.any() else np.nan
        out[f"n_c{c}"] = int(m.sum())
        cls.append(out[f"cov_c{c}"])
    out["cov_minority"] = out[f"cov_c{minority}"]
    out["cov_majority"] = out[f"cov_c{1 - minority}"]
    out["worst_class_cov"] = np.nanmin(cls)
    out["macro_cov"] = np.nanmean(cls)
    single = size == 1
    out["singleton_error"] = (~hit[single]).mean() if single.any() else np.nan
    # among singletons predicting minority / majority
    pred = sets.argmax(1)
    for c, name in ((minority, "min"), (1 - minority, "maj")):
        m = single & (pred == c)
        out[f"singleton_{name}_precision"] = hit[m].mean() if m.any() else np.nan
    return out


def screening_cost(sets, y, c_fp=1.0, c_fn=1.0, c_defer=0.3, positive=1):
    """Normalized per-compound cost of a BBB screening triage.
    singleton {positive} -> advance; wrong if y != positive (cost c_fp)
    singleton {negative} -> discard;  wrong if y == positive (cost c_fn)
    full set or EMPTY set -> defer to assay (cost c_defer).  Empty sets are deferred,
    not silently free as in the legacy MoML cost code."""
    size = sets.sum(1)
    pred = sets.argmax(1)
    single = size == 1
    fp = single & (pred == positive) & (y != positive)
    fn = single & (pred != positive) & (y == positive)
    defer = ~single
    return float((c_fp * fp + c_fn * fn + c_defer * defer).mean())
