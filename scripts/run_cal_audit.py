"""D1: can the hidden low-support failure be detected BEFORE deployment, from calibration data
alone (no test labels)?

For each split, the leave-one-out (LOO) coverage of class-conditional LAC on CAL is computed:
a BBB- calibration molecule i counts as covered if its score is <= the Mondrian threshold
computed from the OTHER BBB- calibration molecules at level alpha (rank ceil(n(1-alpha)) among
n-1 others, i.e. exactly the threshold molecule i would face as a test point).  Averaging over
the low-support CAL molecules estimates low-support BBB- coverage.  This uses CAL labels only,
which are available to any practitioner who calibrates.  It is a standard LOO estimate applied
as an audit, not a new method.

Compared against the TEST low-support coverage of the full-CAL Mondrian predictor on the same
split.  Reported: agreement across splits (Spearman, mean absolute error), cell-level agreement
(12 setting x backbone cells), and a pre-stated flag rule: flag a split when its CAL audit is
below 0.85, success = TEST low-support coverage below 0.85.
Confirmatory seeds 100-119, exchangeable settings, LAC, alpha 0.10, CE.  EXPLORATORY.

usage: python scripts/run_cal_audit.py
Outputs: tables/cal_audit_per_split.csv, tables/cal_audit_cells.csv, tables/cal_audit_flags.csv
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]
ALPHA, FLAG = 0.10, 0.85


def loo_covered(scores, alpha):
    """For each score, is it <= the conformal threshold computed from the others?"""
    s = np.asarray(scores, float); n = len(s)
    if n < 2:
        return np.ones(n, bool)
    k = math.ceil(n * (1 - alpha) - 1e-12)          # rank among the n-1 others (n-1 cal + 1 test)
    if k > n - 1:
        return np.ones(n, bool)                        # infinite threshold
    order = np.argsort(s, kind="stable"); srt = s[order]
    pos = np.empty(n, int); pos[order] = np.arange(n)
    thr = np.where(pos < k, srt[k], srt[k - 1])        # k-th smallest after removing self
    return s <= thr


def main():
    rows = []
    for st in SETTINGS:
        for md in MODELS:
            for seed in range(100, 120):
                z = np.load(REPO / "results/raw" / st / f"{md}_ce" / f"seed{seed}.npz", allow_pickle=True)
                mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
                bc, _ = sim_bins(ref, z["sim_cal"]); bt, _ = sim_bins(ref, z["sim_test"])
                sc = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
                mc = yc == mino
                cov_c = loo_covered(sc[mc], ALPHA)
                T = C.mondrian(sc, yc, St, ALPHA)
                hit = C.predict_sets(St, T)[np.arange(len(yt)), yt]
                mt = yt == mino
                r = {"setting": st, "model": md, "seed": seed}
                for b, nm in ((0, "low"), (2, "high")):
                    r[f"cal_{nm}"] = cov_c[bc[mc] == b].mean() if (bc[mc] == b).any() else np.nan
                    r[f"n_cal_{nm}"] = int((bc[mc] == b).sum())
                    r[f"test_{nm}"] = hit[mt & (bt == b)].mean() if (mt & (bt == b)).any() else np.nan
                    r[f"n_test_{nm}"] = int((mt & (bt == b)).sum())
                r["cal_all"] = cov_c.mean(); r["test_all"] = hit[mt].mean()
                rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(REPO / "tables/cal_audit_per_split.csv", index=False)
    cells = []
    for (st, md), g in D.groupby(["setting", "model"], sort=False):
        c_mu, c_lo, c_hi = mean_ci(g.cal_low.values); t_mu, t_lo, t_hi = mean_ci(g.test_low.values)
        cells.append({"setting": st, "model": md, "cal_low_mean": c_mu, "cal_low_hi": c_hi, "test_low_mean": t_mu,
                      "test_low_hi": t_hi, "cal_flags_failure": c_hi < 0.9, "test_shows_failure": t_hi < 0.9,
                      "mean_n_cal_low": g.n_cal_low.mean(), "cal_gap_high_minus_low": (g.cal_high - g.cal_low).mean(),
                      "test_gap_high_minus_low": (g.test_high - g.test_low).mean()})
    Cc = pd.DataFrame(cells); Cc.to_csv(REPO / "tables/cal_audit_cells.csv", index=False)
    ok = D.dropna(subset=["cal_low", "test_low"])
    rho = spearmanr(ok.cal_low, ok.test_low)
    flag = ok.cal_low < FLAG; fail = ok.test_low < FLAG
    tp, fp = int((flag & fail).sum()), int((flag & ~fail).sum()); fn, tn = int((~flag & fail).sum()), int((~flag & ~fail).sum())
    F = pd.DataFrame([{"n_splits": len(ok), "spearman_split": rho.statistic, "spearman_p": rho.pvalue,
                       "mae_split": float((ok.cal_low - ok.test_low).abs().mean()),
                       "spearman_cell": spearmanr(Cc.cal_low_mean, Cc.test_low_mean).statistic,
                       "flag_threshold": FLAG, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                       "ppv": tp / max(tp + fp, 1), "base_rate_test_fail": float(fail.mean()),
                       "cell_agreement": int((Cc.cal_flags_failure == Cc.test_shows_failure).sum()),
                       "split_gap_sign_cal_pos": float(((ok.cal_high - ok.cal_low) > 0).mean()),
                       "split_gap_sign_test_pos": float(((ok.test_high - ok.test_low) > 0).mean()),
                       "split_gap_sign_agree": float((np.sign(ok.cal_high - ok.cal_low) == np.sign(ok.test_high - ok.test_low)).mean()),
                       "n_cells": len(Cc)}])
    F.to_csv(REPO / "tables/cal_audit_flags.csv", index=False)
    pd.set_option("display.width", 220)
    print(Cc.round(3).to_string(index=False)); print(); print(F.round(3).T.to_string())


if __name__ == "__main__":
    main()
