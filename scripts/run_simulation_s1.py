"""S1: synthetic simulation of the hidden stratum failure (RESEARCH_DESIGN.md, experiment S1).

World: support u ~ Unif(0, 1) plays the role of max-Tanimoto similarity to training.
Label Y ~ Bernoulli(1 - pi0) (class 0 = minority).  A fixed, imperfect classifier outputs
    logit = (2Y - 1) * m(u) + b + sigma * N(0, 1),   m(u) = g0 + g1 * (u - 0.5),
with b = log((1 - pi0) / pi0) (majority-favouring prior), so prediction quality for both
classes falls as support u falls when g1 > 0.  g1 = 0 is the negative control (no structure).
Strata = tertiles of u with edges estimated on an independent VAL sample (as in the paper).

Panel A: minority coverage in each support tertile vs heterogeneity g1, for marginal,
Mondrian, class x support Mondrian and SABM; the stratum floor 1 - alpha / w (Prop. P1).
Panel B: calibration budget.  Per-cell minority calibration size n in {5..160}; the
across-replicate 5th percentile of (test-estimated) cell coverage vs the Beta-law prediction.

Output: tables/s1_simulation_panelA.csv, tables/s1_simulation_panelB.csv,
        figures/s1_simulation.{png,pdf}
usage:  python scripts/run_simulation_s1.py [--reps 300]
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from bbbcp.conformal import calibrators as C
from bbbcp.utils.figio import save

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.10


def draw(rng, n, pi0, g0, g1, sigma=1.0, u=None, y=None):
    u = rng.uniform(size=n) if u is None else u
    y = (rng.uniform(size=n) > pi0).astype(int) if y is None else y
    b = math.log((1 - pi0) / pi0)
    logit = (2 * y - 1) * (g0 + g1 * (u - 0.5)) + b + sigma * rng.normal(size=n)
    p1 = 1 / (1 + np.exp(-logit))
    P = np.column_stack([1 - p1, p1])
    return u, y, 1 - P  # LAC score matrix S[:, c] = 1 - p_c


def tert(u_val, u):
    edges = np.quantile(u_val, [1 / 3, 2 / 3])
    return np.digitize(u, edges)


def panel_a(reps, rng, pi0=0.25, g0=1.5, n_cal=600, n_test=20000):
    rows = []
    for g1 in (0.0, 1.0, 2.0, 3.0, 4.0):
        for r in range(reps):
            uv, _, _ = draw(rng, 600, pi0, g0, g1)
            uc, yc, Sc = draw(rng, n_cal, pi0, g0, g1)
            ut, yt, St = draw(rng, n_test, pi0, g0, g1)
            bc, bt = tert(uv, uc), tert(uv, ut)
            s_true = Sc[np.arange(n_cal), yc]
            for m in ("marginal", "mondrian", "mondrian_x", "sabm"):
                T = C.CALIBRATORS[m](s_true, yc, St, ALPHA, bin_cal=bc, bin_test=bt)
                hit = (St <= T)[np.arange(n_test), yt]
                size = (St <= T).sum(1).mean()
                for b in range(3):
                    mm = (bt == b) & (yt == 0)
                    rows.append(dict(g1=g1, rep=r, method=m, stratum=b, cov_min=hit[mm].mean(),
                                     n_cal_min_stratum=int(((bc == b) & (yc == 0)).sum()),
                                     set_size=size, cov_min_class=hit[yt == 0].mean()))
    return pd.DataFrame(rows)


def panel_b(reps, rng, pi0=0.25, g0=1.5, g1=3.0, n_test=20000):
    rows = []
    for n in (5, 10, 20, 40, 80, 160):
        k = math.ceil((n + 1) * (1 - ALPHA))
        for r in range(reps):
            # exactly n minority calibration points in the low-support tertile (u < 1/3)
            u = rng.uniform(0, 1 / 3, size=n)
            _, _, Sc = draw(rng, n, pi0, g0, g1, u=u, y=np.zeros(n, int))
            q = C.conformal_quantile(Sc[:, 0], ALPHA)
            ut = rng.uniform(0, 1 / 3, size=n_test)
            _, _, St = draw(rng, n_test, pi0, g0, g1, u=ut, y=np.zeros(n_test, int))
            rows.append(dict(n=n, rep=r, cov=(St[:, 0] <= q).mean(), k=k))
    df = pd.DataFrame(rows)
    agg = df.groupby("n").agg(mean_cov=("cov", "mean"), p05_cov=("cov", lambda x: np.quantile(x, 0.05)),
                              frac_below_080=("cov", lambda x: (x < 1 - 2 * ALPHA).mean())).reset_index()
    agg["beta_mean"] = [(math.ceil((n + 1) * (1 - ALPHA))) / (n + 1) if math.ceil((n + 1) * (1 - ALPHA)) <= n else 1.0 for n in agg.n]
    agg["beta_p05"] = [beta_dist.ppf(0.05, kk, n + 1 - kk) if kk <= n else 1.0
                       for n, kk in ((n, math.ceil((n + 1) * (1 - ALPHA))) for n in agg.n)]
    agg["beta_P_below_080"] = [beta_dist.cdf(1 - 2 * ALPHA, kk, n + 1 - kk) if kk <= n else 0.0
                               for n, kk in ((n, math.ceil((n + 1) * (1 - ALPHA))) for n in agg.n)]
    return agg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=300); ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    A = panel_a(a.reps, rng)
    B = panel_b(a.reps, rng)
    aggA = A.groupby(["g1", "method", "stratum"]).agg(
        cov_min=("cov_min", "mean"), cov_min_p10=("cov_min", lambda x: np.quantile(x, 0.10)),
        cov_min_class=("cov_min_class", "mean"), set_size=("set_size", "mean"),
        n_cal_min_stratum=("n_cal_min_stratum", "mean")).reset_index()
    aggA["n_reps"] = a.reps
    (REPO / "tables").mkdir(exist_ok=True)
    aggA.to_csv(REPO / "tables/s1_simulation_panelA.csv", index=False)
    B.to_csv(REPO / "tables/s1_simulation_panelB.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    sty = {"marginal": ("#999999", ":"), "mondrian": ("#2a7a5f", "-"), "mondrian_x": ("#3a7bd5", "--"), "sabm": ("#c0392b", "-")}
    lab = {"marginal": "marginal", "mondrian": "Mondrian (class)", "mondrian_x": "class x support", "sabm": "SABM"}
    for m, (col, ls) in sty.items():
        d = aggA[(aggA.method == m) & (aggA.stratum == 0)]
        ax[0].plot(d.g1, d.cov_min, ls, color=col, marker="o", ms=3, label=lab[m])
        d2 = aggA[(aggA.method == m) & (aggA.g1 == 3.0)]
        ax[1].plot(d2.stratum, d2.cov_min, ls, color=col, marker="o", ms=3, label=lab[m])
    for k in (0, 1):
        ax[k].axhline(1 - ALPHA, color="k", lw=0.8, ls="--")
        ax[k].axhline(1 - 3 * ALPHA, color="k", lw=0.6, ls="-.")
    ax[0].text(0.05, 1 - 3 * ALPHA + 0.01, "floor 1 - 3α (Prop. 1)", fontsize=7)
    ax[0].set_xlabel("support heterogeneity g1 (0 = negative control)"); ax[0].set_ylabel("minority coverage, low-support tertile")
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(["low", "mid", "high"]); ax[1].set_xlabel("support tertile (g1 = 3)"); ax[1].set_ylabel("minority coverage")
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    ax[2].plot(B.n, B.p05_cov, "o-", color="#c0392b", ms=3, label="simulated 5th pct")
    ax[2].plot(B.n, B.beta_p05, "k--", lw=1, label="Beta law 5th pct")
    ax[2].axhline(1 - ALPHA, color="k", lw=0.8, ls="--")
    ax[2].set_xscale("log"); ax[2].set_xlabel("minority calibration points in the cell"); ax[2].set_ylabel("cell coverage")
    ax[2].legend(fontsize=6.5, frameon=False, loc="lower right")
    for k, t in enumerate("ABC"):
        ax[k].set_title(t, loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, REPO / "figures/s1_simulation")
    print(aggA[aggA.stratum == 0].pivot(index="g1", columns="method", values="cov_min").round(3))
    print(aggA[aggA.g1 == 3.0].pivot(index="stratum", columns="method", values="cov_min").round(3))
    print(B.round(3))


if __name__ == "__main__":
    main()
