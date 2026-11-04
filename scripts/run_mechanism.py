"""D8: a testable mechanism for the masked failure (EXPLORATORY).

Hypothesis (stated before running on the confirmatory seeds): off the training distribution,
classifiers regress toward the MAJORITY class.  If that is the cause, three predictions follow,
each checkable on the cached predictions without any refitting:

  P1  Asymmetry.  Under class-conditional calibration, low-support MINORITY compounds are
      under-covered while low-support MAJORITY compounds are not (they are covered at or above
      target).  A generic 'models are worse off-domain' account predicts both classes suffer.
  P2  Prior regression.  Among true minority compounds, the predicted minority probability falls
      as support falls; among true majority compounds, the predicted majority probability does
      not fall comparably.
  P3  Calibration-in-the-large by stratum.  In the low-support tertile, the mean predicted
      minority probability is BELOW the observed minority fraction (minority under-predicted);
      in the high-support tertile the gap is smaller or reversed.

Per setting x backbone, seeds 100-119, LAC, alpha 0.10, CE.  Means over splits with t-intervals.

usage: python scripts/run_mechanism.py [--runs bbbp/random ...] [--models rf gin chemprop] [--tag mechanism]
Outputs: tables/<tag>_per_split.csv, tables/<tag>_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.10


def split_rows(z):
    mino = int(z["minority"]); maj = 1 - mino
    yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    bt, _ = sim_bins(ref, z["sim_test"])
    sc = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
    hit = C.predict_sets(St, C.mondrian(sc, yc, St, ALPHA))[np.arange(len(yt)), yt]
    p_min = z["p_test"][:, mino].astype(float)
    r = {}
    for b, nm in ((0, "low"), (1, "mid"), (2, "high")):
        sb = bt == b
        for cls, lab in ((mino, "min"), (maj, "maj")):
            k = sb & (yt == cls)
            r[f"cov_{lab}_{nm}"] = hit[k].mean() if k.any() else np.nan
            r[f"p_true_{lab}_{nm}"] = (p_min[k] if cls == mino else 1 - p_min[k]).mean() if k.any() else np.nan
        r[f"mean_pmin_{nm}"] = p_min[sb].mean() if sb.any() else np.nan
        r[f"obs_min_frac_{nm}"] = (yt[sb] == mino).mean() if sb.any() else np.nan
        r[f"calib_gap_{nm}"] = r[f"mean_pmin_{nm}"] - r[f"obs_min_frac_{nm}"]
    r["minority_prevalence_test"] = (yt == mino).mean()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"])
    ap.add_argument("--models", nargs="+", default=["rf", "gin", "chemprop"])
    ap.add_argument("--tag", default="mechanism")
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        for md in a.models:
            for seed in range(100, 120):
                p = REPO / "results/raw" / run / f"{md}_ce" / f"seed{seed}.npz"
                if p.exists():
                    rows.append({"run": run, "model": md, "seed": seed, **split_rows(np.load(p, allow_pickle=True))})
    D = pd.DataFrame(rows); D.to_csv(REPO / f"tables/{a.tag}_per_split.csv", index=False)
    out = []
    for (run, md), g in D.groupby(["run", "model"], sort=False):
        r = {"run": run, "model": md, "n_seeds": g.seed.nunique(), "prevalence": g.minority_prevalence_test.mean()}
        for c in [c for c in g.columns if c.startswith(("cov_", "p_true_", "calib_gap_", "mean_pmin_", "obs_min_frac_"))]:
            mu, lo, hi = mean_ci(g[c].values); r[c] = mu; r[c + "_lo"] = lo; r[c + "_hi"] = hi
        r["P1_min_low_below_090"] = bool(r["cov_min_low_hi"] < 0.9)
        r["P1_maj_low_at_or_above_090"] = bool(r["cov_maj_low_lo"] >= 0.88)
        r["P2_drop_p_true_min_high_to_low"] = r["p_true_min_high"] - r["p_true_min_low"]
        r["P2_drop_p_true_maj_high_to_low"] = r["p_true_maj_high"] - r["p_true_maj_low"]
        r["P3_low_gap_negative"] = bool(r["calib_gap_low_hi"] < 0)
        out.append(r)
    S = pd.DataFrame(out); S.to_csv(REPO / f"tables/{a.tag}_summary.csv", index=False)
    pd.set_option("display.width", 230)
    cols = ["run", "model", "prevalence", "cov_min_low", "cov_maj_low", "cov_min_high", "cov_maj_high",
            "P2_drop_p_true_min_high_to_low", "P2_drop_p_true_maj_high_to_low", "calib_gap_low", "calib_gap_high",
            "P1_min_low_below_090", "P1_maj_low_at_or_above_090", "P3_low_gap_negative"]
    print(S[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
