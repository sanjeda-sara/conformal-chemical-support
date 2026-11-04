"""Supplementary figures drawn from the analysis tables (no recomputation; every plotted value is a table entry).

  supp_feasibility        degenerate-cell fraction vs minority calibration molecules per stratum (design rule)
  supp_mechanism          true-class probability by support tertile, BBB- vs BBB+ (12 cells + other tasks)
  supp_reliability        observed BBB- rate vs mean predicted BBB- probability, by support tertile
  supp_descriptors        TPSA, HBD, MW, cLogP of shared failures vs other eligible BBB- molecules, 5 settings
  supp_cal_audit          CAL leave-one-out vs TEST low-support coverage, per split
  supp_generality         low- and high-support minority coverage on other tasks and the prior protocol
usage: python scripts/make_supp_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from bbbcp.utils.figio import save

REPO = Path(__file__).resolve().parents[1]
T = REPO / "tables"; F = REPO / "figures"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
SET = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
SLAB = {"bbbp/random": "BBBP random", "bbbp/scaffold": "BBBP scaffold", "b3db/random": "B3DB random",
        "b3db/scaffold": "B3DB scaffold", "bbbp→b3db/scaffold": "BBBP→B3DB"}
MOD = [("rf", "RF", "o"), ("gin", "GIN", "s"), ("chemprop", "Chemprop", "^")]
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#8a8a85", "#1baf7a"


def feasibility():
    G = pd.read_csv(T / "common_grid_summary.csv"); G["n_min_cal"] = G.n_min_cal.astype(str)
    G = G[(G.alpha == 0.1) & (G.n_seeds == 20) & (G.n_min_cal != "all")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    cols = {2: BLUE, 3: GREEN, 5: ORANGE, 8: "#b54a9a"}
    for nb in (2, 3, 5, 8):
        g = G[G.method == f"fixed_b{nb}"].groupby("n_min_cal").degenerate_cell_frac.mean()
        x = np.array([int(n) / nb for n in g.index]); o = np.argsort(x)
        ax.plot(x[o], g.values[o], "o-", color=cols[nb], ms=3.5, lw=1.1, label=f"{nb} strata")
    ax.axvline(9, ls=":", color="#444", lw=0.8); ax.axvline(11, ls="--", color="#444", lw=0.8)
    ax.text(8.7, 0.53, "finite\nthreshold\n(n≥9) ", fontsize=6.5, va="top", ha="right"); ax.text(11.5, 0.42, "$n_{req}=11$", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel("BBB− calibration molecules per stratum (subsampled)")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([2, 5, 10, 20, 40, 80]); ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("fraction of cells with\ninfinite threshold"); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, F / "supp_feasibility"); plt.close(fig)


def mechanism():
    M = pd.read_csv(T / "mechanism_summary.csv"); MG = pd.read_csv(T / "mechanism_generality_summary.csv")
    fig, axes = plt.subplots(1, 5, figsize=(6.5, 2.2), sharey=True)
    for ax, st in zip(axes[:4], SET):
        for md, lab, mk in MOD:
            r = M[(M.run == st) & (M.model == md)].iloc[0]
            for cls, col in (("min", ORANGE), ("maj", BLUE)):
                ax.plot(range(3), [r[f"p_true_{cls}_{t}"] for t in ("low", "mid", "high")], mk + "-", color=col, ms=2.8, lw=0.9)
        ax.set_title(SLAB[st], loc="left"); ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"])
    ax = axes[4]
    for run, lab, mk in (("bace/random", "BACE", "o"), ("tox21_sr_are/random", "Tox21", "s"), ("clintox/random", "ClinTox", "^")):
        r = MG[MG.run == run].iloc[0]
        for cls, col in (("min", ORANGE), ("maj", BLUE)):
            ax.plot(range(3), [r[f"p_true_{cls}_{t}"] for t in ("low", "mid", "high")], mk + "-", color=col, ms=2.8, lw=0.9)
    ax.set_title("other tasks (RF)", loc="left"); ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"])
    axes[0].set_ylabel("mean predicted probability\nof the true class")
    h = [plt.Line2D([], [], color=ORANGE, lw=1.2), plt.Line2D([], [], color=BLUE, lw=1.2)] + \
        [plt.Line2D([], [], ls="", marker=m, color="#444", ms=3.5) for _, _, m in MOD]
    fig.legend(h, ["minority (BBB−)", "majority (BBB+)", "RF / BACE", "GIN / Tox21", "Chemprop / ClinTox"], ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.01, "support tertile", ha="center", fontsize=7)
    fig.tight_layout(w_pad=0.5); save(fig, F / "supp_mechanism"); plt.close(fig)


def reliability():
    B = pd.read_csv(T / "support_calibration_bins.csv")
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), sharey=True, sharex=True)
    for ax, st in zip(axes, SET):
        for t, col, lab in ((0, ORANGE, "low support"), (1, GREY, "mid"), (2, BLUE, "high support")):
            g = B[(B.setting == st) & (B.tert == t) & (B.n >= 30)]
            ax.plot(g.mean_p, g.obs_rate, "o-", color=col, ms=2.8, lw=1, label=lab)
        ax.plot([0, 1], [0, 1], ":", color="#444", lw=0.7); ax.set_title(SLAB[st], loc="left")
        ax.set_xlabel("mean predicted P(BBB−)")
    axes[0].set_ylabel("observed BBB− fraction"); axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.5); save(fig, F / "supp_reliability"); plt.close(fig)


def descriptors():
    W = pd.read_csv(T / "persistent_failure_molecules_ce.csv", low_memory=False); W = W[W.eligible.astype(bool)]
    sts = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold", "bbbp→b3db/scaffold"]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.3))
    for ax, (col, lab, lim) in zip(axes, (("TPSA", "TPSA (Å²)", (0, 250)), ("HBD", "H-bond donors", (0, 8)),
                                          ("MW", "molecular weight", (0, 800)), ("cLogP", "cLogP", (-4, 8)))):
        for i, st in enumerate(sts):
            g = W[W.setting == st]; m = g.majority_all3_mondrian.astype(bool)
            for k, (sel, c) in enumerate(((~m, GREY), (m, ORANGE))):
                v = g.loc[sel, col].dropna().values
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[i + (k - 0.5) * 0.36], widths=0.3, showfliers=False, patch_artist=True,
                                medianprops={"color": "k", "lw": 0.8}, whiskerprops={"lw": 0.5}, capprops={"lw": 0.5})
                bp["boxes"][0].set(facecolor=c, alpha=0.6, lw=0.5)
        ax.set_xticks(range(len(sts))); ax.set_xticklabels(["BBBP\nrand.", "BBBP\nscaf.", "B3DB\nrand.", "B3DB\nscaf.", "BBBP→\nB3DB"], fontsize=6)
        ax.set_ylim(*lim); ax.set_title(lab, loc="left")
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6) for c in (GREY, ORANGE)]
    fig.legend(h, ["other eligible BBB−", "missed by all three backbones in most tests (majority rule)"], ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.6); save(fig, F / "supp_descriptors"); plt.close(fig)


def cal_audit():
    D = pd.read_csv(T / "cal_audit_per_split.csv").dropna(subset=["cal_low", "test_low"])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for st, c in zip(SET, (BLUE, "#5aa3e8", GREEN, ORANGE)):
        g = D[D.setting == st]; ax.plot(g.cal_low, g.test_low, "o", ms=2.5, color=c, alpha=0.75, mew=0, label=SLAB[st])
    ax.plot([0.5, 1], [0.5, 1], ":", color="#444", lw=0.7)
    ax.axhline(0.85, color="#999", lw=0.5); ax.axvline(0.85, color="#999", lw=0.5)
    ax.set_xlabel("CAL leave-one-out low-support coverage"); ax.set_ylabel("TEST low-support coverage")
    ax.set_xlim(0.5, 1.01); ax.set_ylim(0.5, 1.01); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); save(fig, F / "supp_cal_audit"); plt.close(fig)


def generality():
    rows = []
    for fam in ("generality", "tursun_confirm"):
        S = pd.read_csv(T / f"{fam}_summary.csv")
        for _, r in S.iterrows():
            nm = ("BBBP 50/25/25\n" + r.model.upper()) if fam == "tursun_confirm" else \
                r.run.replace("tox21_sr_are", "Tox21").replace("bace", "BACE").replace("clintox", "ClinTox").replace("/", "\n")
            rows.append((nm, r["mondrian.low"], r["mondrian.high"], r["mondrian_x.low"], r["mondrian_x.high"]))
    fig, ax = plt.subplots(figsize=(6.5, 2.3)); x = np.arange(len(rows))
    for j, (lab, col, dx) in enumerate((("class-cond., low", BLUE, -0.3), ("class-cond., high", "#9cc3ee", -0.1),
                                        ("class × support, low", ORANGE, 0.1), ("class × support, high", "#f4b496", 0.3))):
        ax.bar(x + dx, [r[1 + j] for r in rows], width=0.2, color=col, label=lab)
    ax.axhline(0.9, ls="--", lw=0.7, color="#444"); ax.set_ylim(0.7, 1.02)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5); ax.set_ylabel("minority coverage")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(); save(fig, F / "supp_generality"); plt.close(fig)


if __name__ == "__main__":
    feasibility(); mechanism(); reliability(); descriptors(); cal_audit(); generality()
    print("wrote figures/supp_{feasibility,mechanism,reliability,descriptors,cal_audit,generality}")
