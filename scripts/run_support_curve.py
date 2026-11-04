"""D5: is the support gradient a tertile artefact, or continuous in maximal Tanimoto support?

For every BBB- test appearance (seeds 100-119), records whether the class-conditional LAC set at
alpha 0.10 covers it, and its max-Tanimoto support to TRAIN.  Per setting x backbone:
  * logistic slope of coverage on support (per +0.1 Tanimoto), with a 95% interval from
    resampling whole splits (400 draws);
  * coverage in fixed absolute support bins, pooled over splits, with Wilson intervals
    (descriptive: molecules recur across splits, so these intervals are optimistic).
Class x support calibration is reported alongside.  EXPLORATORY.

usage: python scripts/run_support_curve.py
Outputs: tables/support_curve_slopes.csv, tables/support_curve_bins.csv,
         figures/supp_support_curve.{png,pdf}
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import wilson

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]
EDGES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001]
ALPHA = 0.10


def collect():
    rows = []
    for st in SETTINGS:
        for md in MODELS:
            for seed in range(100, 120):
                z = np.load(REPO / "results/raw" / st / f"{md}_ce" / f"seed{seed}.npz", allow_pickle=True)
                mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
                bc, _ = sim_bins(ref, z["sim_cal"]); bt, _ = sim_bins(ref, z["sim_test"])
                sc = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
                m = yt == mino
                for meth in ("mondrian", "mondrian_x"):
                    T = C.CALIBRATORS[meth](sc, yc, St, ALPHA, bin_cal=bc, bin_test=bt)
                    hit = C.predict_sets(St, T)[np.arange(len(yt)), yt]
                    rows.append(pd.DataFrame({"setting": st, "model": md, "seed": seed, "method": meth,
                                              "support": z["sim_test"][m], "hit": hit[m].astype(int)}))
    return pd.concat(rows, ignore_index=True)


def slope(g):
    if g.hit.nunique() < 2:
        return np.nan
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(g[["support"]].values * 10, g.hit.values)
    return float(lr.coef_[0][0])                         # log-odds per +0.1 Tanimoto


def main():
    D = collect()
    rng = np.random.default_rng(20261031)
    out = []
    for (st, md, meth), g in D.groupby(["setting", "model", "method"], sort=False):
        b0 = slope(g); seeds = g.seed.unique(); by = {s: h for s, h in g.groupby("seed")}
        boots = [slope(pd.concat([by[s] for s in rng.choice(seeds, len(seeds), replace=True)])) for _ in range(400)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out.append({"setting": st, "model": md, "method": meth, "slope_per_0.1": b0, "ci_lo": lo, "ci_hi": hi,
                    "ci_excludes_0": bool(lo > 0 or hi < 0), "n_appearances": len(g)})
    Sl = pd.DataFrame(out); Sl.to_csv(REPO / "tables/support_curve_slopes.csv", index=False)
    D["bin"] = pd.cut(D.support, EDGES, right=False, labels=[f"{a:.1f}-{b:.1f}" for a, b in zip(EDGES[:-1], EDGES[1:])])
    B = D.groupby(["setting", "model", "method", "bin"], observed=True).hit.agg(["sum", "count"]).reset_index()
    B["coverage"] = B["sum"] / B["count"]
    ci = [wilson(int(k), int(n)) for k, n in zip(B["sum"], B["count"])]
    B["lo"] = [c[0] for c in ci]; B["hi"] = [c[1] for c in ci]
    B.to_csv(REPO / "tables/support_curve_bins.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bbbcp.utils.figio import save
    cols = {"rf": "#2a78d6", "gin": "#eb6834", "chemprop": "#1baf7a"}
    mids = {f"{a:.1f}-{b:.1f}": (a + min(b, 1.0)) / 2 for a, b in zip(EDGES[:-1], EDGES[1:])}
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    for ax, st in zip(axes, SETTINGS):
        for md in MODELS:
            for meth, ls in (("mondrian", "-"), ("mondrian_x", "--")):
                q = B[(B.setting == st) & (B.model == md) & (B.method == meth) & (B["count"] >= 20)]
                ax.plot([mids[b] for b in q.bin], q.coverage, ls, marker="o", ms=3, color=cols[md], lw=1.2,
                        label=f"{md}, {'class-only' if meth == 'mondrian' else 'class x support'}" if st == SETTINGS[0] else None)
        ax.axhline(0.9, ls=":", c="#444", lw=0.7); ax.set_title(st, fontsize=8, loc="left"); ax.tick_params(labelsize=7)
        ax.set_xlabel("max Tanimoto to TRAIN", fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("BBB$-$ coverage (pooled)", fontsize=7)
    fig.legend(fontsize=6.5, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); save(fig, REPO / "figures/supp_support_curve"); plt.close(fig)
    pd.set_option("display.width", 200)
    print(Sl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
