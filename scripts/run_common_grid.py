"""A4: fixed 2-bin vs fixed 3-bin vs SABM on ONE common evaluation grid.  No refitting.

Why.  A class x support partition with b bins guarantees coverage in ITS OWN bins.  Comparing
'low-support coverage' across b is meaningless if 'low support' means the bottom half for
b = 2 and the bottom third for b = 3.  Here every method is scored on the SAME strata, fixed
from VAL similarities (independent of cal and test):
    common tertiles  t0 (lowest support), t1, t2      -> the paper's low-support subgroup
    common sextiles  s0..s5                           -> worst-stratum minority coverage
Calibration partitions compared (all LAC, the frozen n_req rule for SABM):
    mondrian (class only), fixed_b2, fixed_b3 (= mondrian_x), fixed_b5, sabm_b3 (= sabm), sabm_b5
Calibration size: the realized one ('all') and minority subsamples 80 / 40 / 20 (10 draws each,
majority calibration kept whole), which is where adaptive merging can differ from fixed.
Confirmatory seeds 100-119, exchangeable settings, CE, alpha 0.10 (0.05 and 0.20 secondary).
EXPLORATORY: not part of the 36-member primary family; reported descriptively with paired
seed-level intervals.

usage: python scripts/run_common_grid.py [--models rf gin chemprop] [--alphas 0.1] [--draws 10]
Outputs: tables/common_grid_per_run.csv.gz, tables/common_grid_summary.csv,
         tables/common_grid_paired.csv
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.conformal.calibrators import n_required
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import mean_ci, paired_compare

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]   # external = none
METHODS = {"mondrian": ("mondrian", 3), "fixed_b2": ("mondrian_x", 2), "fixed_b3": ("mondrian_x", 3),
           "fixed_b5": ("mondrian_x", 5), "fixed_b8": ("mondrian_x", 8), "sabm_b2": ("sabm", 2),
           "sabm_b3": ("sabm", 3), "sabm_b5": ("sabm", 5), "sabm_b8": ("sabm", 8)}
NBINS = (2, 3, 5, 8)


def run_file(args):
    path, setting, model, seed, alphas, n_min_list, draws = args
    z = np.load(path, allow_pickle=True)
    mino = int(z["minority"]); yc_all = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    s_all = 1.0 - z["p_cal"][np.arange(len(yc_all)), yc_all]; St = 1.0 - z["p_test"]
    t3, _ = sim_bins(ref, z["sim_test"], 3); t6, _ = sim_bins(ref, z["sim_test"], 6)
    bins_cal = {b: sim_bins(ref, z["sim_cal"], b)[0] for b in NBINS}
    bins_test = {b: sim_bins(ref, z["sim_test"], b)[0] for b in NBINS}
    mi = np.where(yc_all == mino)[0]; ma = np.where(yc_all != mino)[0]
    rng = np.random.default_rng(seed)
    m = yt == mino; rows = []
    for n_min in n_min_list:
        if n_min != "all" and n_min > len(mi):
            continue
        for rep in range(1 if n_min == "all" else draws):
            sub = np.arange(len(yc_all)) if n_min == "all" else np.r_[ma, rng.choice(mi, n_min, replace=False)]
            for a in alphas:
                for name, (base, nb) in METHODS.items():
                    T = C.CALIBRATORS[base](s_all[sub], yc_all[sub], St, a,
                                            bin_cal=bins_cal[nb][sub], bin_test=bins_test[nb])
                    sets = C.predict_sets(St, T); hit = sets[np.arange(len(yt)), yt]; size = sets.sum(1)
                    r = {"setting": setting, "model": model, "seed": seed, "n_min_cal": str(n_min), "rep": rep,
                         "alpha": a, "method": name, "n_cal_minority": int((yc_all[sub] == mino).sum()),
                         "cov_minority": hit[m].mean(), "set_size": size.mean(),
                         "singleton_rate": (size == 1).mean(), "full_rate": (size == 2).mean(),
                         "singleton_rate_minority": (size[m] == 1).mean(),
                         "inf_threshold_frac": float(np.isinf(T).mean()),
                         # degenerate cell = (class, bin) cell holding test molecules whose threshold is
                         # infinite (too few calibration points); fraction over such cells
                         "degenerate_cell_frac": float(np.mean([np.isinf(T[bins_test[nb] == b, c][0])
                                                                for c in (0, 1) for b in np.unique(bins_test[nb])]))}
                    for k in range(3):
                        mk = m & (t3 == k)
                        r[f"cov_min_t{k}"] = hit[mk].mean() if mk.any() else np.nan
                        r[f"n_min_t{k}"] = int(mk.sum())
                        r[f"size_t{k}"] = size[t3 == k].mean() if (t3 == k).any() else np.nan
                    sx = [hit[m & (t6 == k)].mean() for k in range(6) if (m & (t6 == k)).sum() >= 5]
                    r["worst_sextile_cov_min"] = min(sx) if sx else np.nan
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["rf", "gin", "chemprop"])
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--n-min", nargs="*", default=["all", "160", "80", "40", "20", "10"])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    n_min_list = [x if x == "all" else int(x) for x in a.n_min]
    jobs = []
    for s, mdl, seed in itertools.product(SETTINGS, a.models, range(100, 120)):
        p = REPO / "results/raw" / s / f"{mdl}_ce" / f"seed{seed}.npz"
        if p.exists():
            jobs.append((p, s, mdl, seed, tuple(a.alphas), n_min_list, a.draws))
    print(f"{len(jobs)} files")
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rr in ex.map(run_file, jobs, chunksize=4) for r in rr]
    D = pd.DataFrame(rows)
    D.to_csv(REPO / "tables/common_grid_per_run.csv.gz", index=False)
    # average subsample draws within a seed first, so the seed stays the unit
    S = D.groupby(["setting", "model", "seed", "n_min_cal", "alpha", "method"]).mean(numeric_only=True).reset_index()
    met = ["cov_minority", "cov_min_t0", "cov_min_t1", "cov_min_t2", "worst_sextile_cov_min", "set_size",
           "singleton_rate", "full_rate", "inf_threshold_frac", "degenerate_cell_frac", "n_cal_minority", "n_min_t0",
           "singleton_rate_minority"]
    summ = []
    for k, g in S.groupby(["setting", "model", "n_min_cal", "alpha", "method"]):
        r = dict(zip(["setting", "model", "n_min_cal", "alpha", "method"], k)); r["n_seeds"] = g.seed.nunique()
        for c in met:
            mu, lo, hi = mean_ci(g[c].values); r[c] = mu; r[c + "_lo"] = lo; r[c + "_hi"] = hi
        summ.append(r)
    pd.DataFrame(summ).to_csv(REPO / "tables/common_grid_summary.csv", index=False)
    pairs = [("fixed_b2", "sabm_b8"), ("fixed_b2", "sabm_b3"), ("fixed_b2", "fixed_b3"), ("fixed_b3", "sabm_b3"), ("fixed_b5", "sabm_b5"),
             ("fixed_b2", "mondrian"), ("fixed_b3", "mondrian")]
    out = []
    for k, g in S.groupby(["setting", "model", "n_min_cal", "alpha"]):
        P = g.pivot(index="seed", columns="method")
        for x, y in pairs:
            for ep in ("cov_min_t0", "worst_sextile_cov_min", "set_size", "singleton_rate", "cov_minority"):
                pc = paired_compare(P[ep][x].values, P[ep][y].values)
                out.append({**dict(zip(["setting", "model", "n_min_cal", "alpha"], k)), "contrast": f"{x} - {y}",
                            "endpoint": ep, **pc})
    pd.DataFrame(out).to_csv(REPO / "tables/common_grid_paired.csv", index=False)
    pd.set_option("display.width", 220)
    v = pd.DataFrame(summ)
    v = v[(v.alpha == 0.1)]
    print(v.pivot_table(index=["setting", "model", "n_min_cal"], columns="method", values="cov_min_t0").round(3).to_string())
    print(v.pivot_table(index=["setting", "model", "n_min_cal"], columns="method", values="set_size").round(3).to_string())


if __name__ == "__main__":
    main()
