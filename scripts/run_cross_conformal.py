"""Item A11: Mondrian cross-conformal prediction (Vovk 2015; Sun et al. 2017) vs split Mondrian.

Cross-conformal uses every non-test molecule both to train and to calibrate, which matters
when the minority calibration set is small.  For each split seed:
  pool  = TRAIN + CAL of the usual split (VAL and TEST untouched; TEST identical to Stage A)
  K = 5 folds of the pool (random, or by scaffold group for scaffold-type regimes)
  model_k trained on pool minus fold k (RF on ECFP4, same hyper-parameters as Stage A)
  p_c(x) = (1 + sum_k #{j in fold k, y_j = c : s_k(j) >= s_k(x, c)}) / (n_c + 1)
  C(x)   = {c : p_c(x) > alpha}       (Mondrian over classes)
  and a class x support variant using the same support tertiles as Stage B.
Cross-conformal validity is approximate (Vovk 2015 gives 1 - 2 alpha in the worst case);
it is reported as an efficiency baseline, not a guaranteed method.

usage: python scripts/run_cross_conformal.py --datasets bbbp b3db --regimes random scaffold \
          --seeds 100-119 [--model rf] [--folds 5]
Output: results/processed/cross_conformal.csv
"""
import argparse

import numpy as np
import pandas as pd

from bbbcp.chemistry.fingerprints import max_tanimoto
from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.metrics.metrics import set_metrics
from bbbcp.models.baselines import train_rf
from bbbcp.pipeline import REPO, prepare, split_for


def folds_for(idx, scaffolds, K, rng, by_group):
    if not by_group:
        perm = rng.permutation(len(idx))
        return [idx[perm[k::K]] for k in range(K)]
    groups = {}
    for i in idx:
        groups.setdefault(scaffolds[i], []).append(i)
    out = [[] for _ in range(K)]
    for g in sorted(groups.values(), key=len, reverse=True):
        out[int(np.argmin([len(o) for o in out]))].extend(g)
    return [np.array(o, int) for o in out]


def cc_pvalues(fold_scores, fold_labels, test_scores, K_lab=2):
    """fold_scores[k]: (n_k,) true-label LAC scores of fold k under model k;
    test_scores[k]: (n_test, K_lab) LAC scores of the test set under model k."""
    n_test = test_scores[0].shape[0]
    P = np.zeros((n_test, K_lab))
    n_c = np.zeros(K_lab)
    for s_f, y_f, S_t in zip(fold_scores, fold_labels, test_scores):
        for c in range(K_lab):
            sc = np.sort(s_f[y_f == c]); n_c[c] += len(sc)
            P[:, c] += len(sc) - np.searchsorted(sc, S_t[:, c], side="left")
    return (P + 1) / (n_c[None, :] + 1)


def cc_pvalues_strata(fold_scores, fold_labels, fold_bins, test_scores, test_bins, nb=3, K_lab=2):
    n_test = test_scores[0].shape[0]
    P = np.zeros((n_test, K_lab)); N = np.zeros((nb, K_lab))
    for s_f, y_f, b_f, S_t in zip(fold_scores, fold_labels, fold_bins, test_scores):
        for c in range(K_lab):
            for b in range(nb):
                sc = np.sort(s_f[(y_f == c) & (b_f == b)]); N[b, c] += len(sc)
                rows = test_bins == b
                P[rows, c] += len(sc) - np.searchsorted(sc, S_t[rows, c], side="left")
    return (P + 1) / (N[test_bins] + 1)


def run(dataset, regime, seed, K=5, alphas=(0.05, 0.1, 0.2)):
    d = prepare(dataset); X, y, df = d["X"], d["df"].y.values, d["df"]
    minority = int(d["audit"]["minority_label"])
    sp = split_for(dataset, regime, seed)
    rng = np.random.default_rng(seed)
    pool = np.sort(np.concatenate([sp["train"], sp["cal"]]))
    folds = folds_for(pool, df.scaffold.values, K, rng, regime != "random")
    fps = d["fps"]
    tr_fps = [fps[i] for i in sp["train"]]
    sim_val = max_tanimoto([fps[i] for i in sp["val"]], tr_fps)
    sim_test = max_tanimoto([fps[i] for i in sp["test"]], tr_fps)
    # support of a fold-k point is measured against model k's training set (pool minus fold k),
    # with the same VAL-derived tertile edges as Stage B
    test_bins, edges = sim_bins(sim_val, sim_test)
    fold_scores, fold_labels, fold_bins, test_scores = [], [], [], []
    for k in range(K):
        tr = np.setdiff1d(pool, folds[k])
        m, _ = train_rf(X[tr], y[tr], "ce", seed * 10 + k)
        pf = m.predict_proba(X[folds[k]]); pt = m.predict_proba(X[sp["test"]])
        fold_scores.append(1 - pf[np.arange(len(folds[k])), y[folds[k]]])
        fold_labels.append(y[folds[k]])
        tr_fps_k = [fps[i] for i in tr]
        fold_bins.append(np.digitize(max_tanimoto([fps[i] for i in folds[k]], tr_fps_k), edges))
        test_scores.append(1 - pt)
    y_test = y[sp["test"]]
    P_cc = cc_pvalues(fold_scores, fold_labels, test_scores)
    P_ccx = cc_pvalues_strata(fold_scores, fold_labels, fold_bins, test_scores, test_bins)
    # split Mondrian reference: train on TRAIN only, calibrate on CAL (identical to Stage A RF)
    m, _ = train_rf(X[sp["train"]], y[sp["train"]], "ce", seed)
    pc, pt = m.predict_proba(X[sp["cal"]]), m.predict_proba(X[sp["test"]])
    s_cal = 1 - pc[np.arange(len(sp["cal"])), y[sp["cal"]]]
    sim_cal = max_tanimoto([fps[i] for i in sp["cal"]], tr_fps)
    cal_bins = np.digitize(sim_cal, edges)
    rows = []
    for a in alphas:
        variants = {"cross_mondrian": P_cc > a, "cross_mondrian_x": P_ccx > a,
                    "split_mondrian": C.predict_sets(1 - pt, C.mondrian(s_cal, y[sp["cal"]], 1 - pt, a)),
                    "split_sabm": C.predict_sets(1 - pt, C.sabm(s_cal, y[sp["cal"]], 1 - pt, a,
                                                              bin_cal=cal_bins, bin_test=test_bins))}
        for name, sets in variants.items():
            r = {"dataset": dataset, "regime": regime, "seed": seed, "alpha": a, "method": name,
                 **set_metrics(sets, y_test, minority)}
            hit = sets[np.arange(len(y_test)), y_test]
            for b in range(3):
                mm = (test_bins == b) & (y_test == minority)
                r[f"cov_min_bin{b}"] = hit[mm].mean() if mm.any() else np.nan
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bbbp", "b3db"])
    ap.add_argument("--regimes", nargs="+", default=["random", "scaffold"])
    ap.add_argument("--seeds", default="100-119"); ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rows = []
    for ds in a.datasets:
        for rg in a.regimes:
            for s in range(lo, hi + 1):
                rows += run(ds, rg, s, a.folds)
                print(ds, rg, s, flush=True)
    out = pd.DataFrame(rows)
    dest = REPO / "results/processed/cross_conformal.csv"
    out.to_csv(dest, index=False); print("wrote", dest, out.shape)
    print(out[np.isclose(out.alpha, 0.1)].groupby(["dataset", "regime", "method"])[
        ["cov_minority", "cov_min_bin0", "set_size"]].mean().round(3))


if __name__ == "__main__":
    main()
