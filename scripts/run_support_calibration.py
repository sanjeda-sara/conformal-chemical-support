"""D7: WHY does one class-conditional threshold fail at low support?  (Corrected version of
'does support explain failure beyond model confidence'.)

Note on design.  Under class-conditional LAC, whether a BBB- molecule is missed is a
deterministic function of its predicted probability p(BBB-) and the CAL threshold, so 'failure
~ confidence + support' among BBB- molecules is degenerate (confidence explains it exactly).
The informative question is about the probability scale itself: at the SAME predicted p(BBB-),
are low-support compounds more often truly BBB-?  If so, the model is under-confident about
BBB- exactly where support is low, and any single per-class threshold must under-cover there.

For every TEST compound (both classes), per setting x backbone, seeds 100-119:
  logistic  1[Y = BBB-] ~ logit(p_BBB-) + support       (support per +0.1 Tanimoto)
A negative support coefficient means that, holding the score fixed, low support implies a
higher chance of being BBB-.  Interval: 400 resamples of whole splits.  Also a model with the
three support tertiles as indicators, and a binned reliability table.  EXPLORATORY.

usage: python scripts/run_support_calibration.py
Outputs: tables/support_calibration_coefs.csv, tables/support_calibration_bins.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bbbcp.evaluate import sim_bins

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]


def collect():
    rows = []
    for st in SETTINGS:
        for md in MODELS:
            for seed in range(100, 120):
                z = np.load(REPO / "results/raw" / st / f"{md}_ce" / f"seed{seed}.npz", allow_pickle=True)
                mino = int(z["minority"]); yt = z["y_test"].astype(int)
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
                p = np.clip(z["p_test"][:, mino].astype(float), 1e-6, 1 - 1e-6)
                rows.append(pd.DataFrame({"setting": st, "model": md, "seed": seed, "y_min": (yt == mino).astype(int),
                                          "logit_p": np.log(p / (1 - p)), "p": p, "support": z["sim_test"],
                                          "tert": sim_bins(ref, z["sim_test"])[0]}))
    return pd.concat(rows, ignore_index=True)


def fit(g):
    X = np.column_stack([g.logit_p.values, g.support.values * 10])
    return LogisticRegression(C=1e6, max_iter=3000).fit(X, g.y_min.values).coef_[0]


def main():
    D = collect()
    rng = np.random.default_rng(20261031)
    out = []
    for (st, md), g in D.groupby(["setting", "model"], sort=False):
        b = fit(g); by = {s: h for s, h in g.groupby("seed")}; seeds = list(by)
        boots = np.array([fit(pd.concat([by[s] for s in rng.choice(seeds, len(seeds), replace=True)])) for _ in range(400)])
        lo, hi = np.percentile(boots[:, 1], [2.5, 97.5])
        out.append({"setting": st, "model": md, "coef_logit_p": b[0], "coef_support_per_0.1": b[1],
                    "support_ci_lo": lo, "support_ci_hi": hi, "support_ci_excludes_0": bool(lo > 0 or hi < 0),
                    "n_test_appearances": len(g)})
    Cf = pd.DataFrame(out); Cf.to_csv(REPO / "tables/support_calibration_coefs.csv", index=False)
    # reliability: within bins of predicted p(BBB-), observed BBB- rate by support tertile (pooled, descriptive)
    D["pbin"] = pd.cut(D.p, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    R = D.groupby(["setting", "pbin", "tert"], observed=True).agg(n=("y_min", "size"), obs_rate=("y_min", "mean"),
                                                                 mean_p=("p", "mean")).reset_index()
    R.to_csv(REPO / "tables/support_calibration_bins.csv", index=False)
    pd.set_option("display.width", 200)
    print(Cf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
