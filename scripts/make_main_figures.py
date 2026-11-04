"""Main-text Figures 2-4 drawn at their printed size (6.5 in wide), so no text is smaller than 7 pt.

Reads only analysis tables (no recomputation):
  Fig. 2  tables/confirmatory_summary.csv            BBB- coverage by support tertile, three calibrations
  Fig. 3  tables/persistent_failure_concordance_ce.csv, tables/concordance_conditional.csv,
          tables/persistent_failure_molecules_ce.csv  shared failures vs two nulls; TPSA and support
  Fig. 4  tables/screening_utility_paired.csv, tables/calibrated_rejection_paired.csv
                                                     BBB- accepted as BBB+, class x support minus comparator
usage: python scripts/make_main_figures.py
Writes figures/paper_fig{2,3,4}_*.{pdf,png} (same file names the paper includes).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bbbcp.utils.figio import save

REPO = Path(__file__).resolve().parents[1]
T = REPO / "tables"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
SET = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
SLAB = {"bbbp/random": "BBBP random", "bbbp/scaffold": "BBBP scaffold", "b3db/random": "B3DB random",
        "b3db/scaffold": "B3DB scaffold", "bbbp→b3db/scaffold": "BBBP→B3DB"}
MOD = [("rf", "RF", "o"), ("gin", "GIN", "s"), ("chemprop", "Chemprop", "^")]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"


def fig2():
    S = pd.read_csv(T / "confirmatory_summary.csv", low_memory=False)
    S = S[(S.score == "lac") & (S.alpha == 0.1) & (S.loss == "ce") & S.external.isna() & (S.n_seeds == 20)]
    meths = [("mondrian", "class-conditional", BLUE, -0.18), ("mondrian_x_random", "class × random strata", GREY, 0.0),
             ("mondrian_x", "class × Tanimoto support", ORANGE, 0.18)]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.35), sharey=True)
    for ax, st in zip(axes, SET):
        for meth, lab, col, dx in meths:
            vals = []
            for j, (md, mlab, mk) in enumerate(MOD):
                r = S[(S.setting == st) & (S.model_name == md) & (S.method == meth)].iloc[0]
                v = [r[f"cov_min_bin{b}"] for b in range(3)]; vals.append(v)
                ax.plot(np.arange(3) + dx + (j - 1) * 0.045, v, mk, ms=2.6, color=col, alpha=0.9, mew=0)
            ax.plot(np.arange(3) + dx, np.mean(vals, axis=0), "-", color=col, lw=1.3, label=lab)
        ax.axhline(0.9, ls="--", lw=0.6, color="#444")
        ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"]); ax.set_xlim(-0.45, 2.45)
        ax.set_title(SLAB[st], loc="left"); ax.set_ylim(0.76, 1.0)
    axes[0].set_ylabel("BBB− coverage")
    fig.text(0.53, -0.005, "support tertile (maximal Tanimoto similarity to TRAIN, edges from VAL)", ha="center", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    mh = [plt.Line2D([], [], ls="", marker=mk, color="#444", ms=3.5) for _, _, mk in MOD]
    fig.legend(h + mh, l + [m[1] for m in MOD], ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.08),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, REPO / "figures/paper_fig2_hidden_failure_confirmatory"); plt.close(fig)


def fig3():
    C = pd.read_csv(T / "persistent_failure_concordance_ce.csv"); C = C[C.miss_definition == "mondrian"].set_index("setting")
    CC = pd.read_csv(T / "concordance_conditional.csv").set_index("setting")
    W = pd.read_csv(T / "persistent_failure_molecules_ce.csv", low_memory=False)
    W = W[W.eligible.astype(bool)]
    sts = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold", "bbbp→b3db/scaffold"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]; x = np.arange(len(sts))
    ax.bar(x, [C.loc[s, "n_cross_arch"] for s in sts], color=ORANGE, width=0.6, label="observed")
    ax.plot(x - 0.12, [C.loc[s, "null_mean"] for s in sts], "_", color="k", ms=11, mew=2, label="null: independent")
    ax.plot(x + 0.12, [CC.loc[s, "null_cov_mean"] for s in sts], "_", color=BLUE, ms=11, mew=2, label="null: within support, TPSA strata")
    ax.set_xticks(x); ax.set_xticklabels(["BBBP\nrand.", "BBBP\nscaf.", "B3DB\nrand.", "B3DB\nscaf.", "BBBP→\nB3DB"], fontsize=6.5)
    ax.set_ylabel("BBB− molecules missed by all\nthree backbones at every test")
    ax.set_title("A  shared persistent failures", loc="left"); ax.set_ylim(0, 20); ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    for j, (col, lab, lim) in enumerate((("TPSA", "TPSA (Å²)", (0, 200)), ("mean_sim_train", "support (max Tanimoto)", (0.1, 0.9)))):
        ax = axes[j + 1]
        for i, st in enumerate(["b3db/random", "b3db/scaffold"]):
            g = W[W.setting == st]; f = g.cross_arch_mondrian.astype(bool); m = g.majority_all3_mondrian.astype(bool)
            for k, (sel, c, nm) in enumerate(((~m, GREY, "other BBB−"), (m & ~f, BLUE, "majority-rule"), (f, ORANGE, "strict"))):
                v = g.loc[sel, col].dropna().values
                pos = i * 1.0 + (k - 1) * 0.28
                bp = ax.boxplot(v, positions=[pos], widths=0.22, showfliers=False, patch_artist=True,
                                medianprops={"color": "k", "lw": 0.9}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
                bp["boxes"][0].set(facecolor=c, alpha=0.55, lw=0.6)
                if k == 2:
                    ax.plot(pos + np.random.default_rng(1).uniform(-0.06, 0.06, len(v)), v, "o", ms=2, color=c, mew=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["B3DB\nrandom", "B3DB\nscaffold"]); ax.set_ylim(*lim); ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(lab); ax.set_title(("B  polarity", "C  chemical support")[j], loc="left")
    hb = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.55) for c in (GREY, BLUE, ORANGE)]
    fig.legend(hb, ["other eligible BBB−", "missed by all three in most tests (majority rule)", "missed by all three in every test (strict)"],
               frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.62, 1.07), fontsize=6.5, handlelength=1.2)
    fig.tight_layout(w_pad=0.8)
    save(fig, REPO / "figures/paper_fig3_persistent_failures_ce"); plt.close(fig)


def fig4():
    U = pd.read_csv(T / "screening_utility_paired.csv"); U = U[U.alpha == 0.1]
    R = pd.read_csv(T / "calibrated_rejection_paired.csv")
    comps = [("fixed_b3 - mondrian", U, "class-conditional alone"), ("fixed_b3 - mond_ad", R, "class-cond. + AD filter"),
             ("fixed_b3 - mond_conf", R, "class-cond. + confidence filter"), ("fixed_b3 - ad_reject", U, "AD rejection"),
             ("fixed_b3 - conf_reject", U, "confidence rejection")]
    sts = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold", "bbbp→b3db/scaffold"]
    cols = ["#2a78d6", "#5aa3e8", "#1baf7a", "#eb6834", "#b54a9a"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True)
    for ax, (ep, title) in zip(axes, (("minority_false_accept_rate", "A  BBB− accepted as BBB+"),
                                      ("accepted_error", "B  error among accepted calls"))):
        for i, (con, D, lab) in enumerate(comps):
            for j, st in enumerate(sts):
                g = D[(D.contrast == con) & (D.endpoint == ep) & (D.setting == st)].mean_diff.values
                y = len(comps) - 1 - i + (j - 2) * 0.13
                ax.plot([g.min(), g.max()], [y, y], "-", color=cols[j], lw=0.9)
                ax.plot(g.mean(), y, "o" if j < 4 else "D", color=cols[j], ms=3, mew=0,
                        label=SLAB[st] if (i == 0 and ax is axes[0]) else None)
        ax.axvline(0, color="#444", lw=0.6)
        ax.set_title(title, loc="left"); ax.set_xlabel("class × support minus comparator")
    axes[0].set_yticks(range(len(comps))); axes[0].set_yticklabels([c[2] for c in comps][::-1])
    fig.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.6, 1.07), handlelength=0.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, REPO / "figures/paper_fig4_screening_utility"); plt.close(fig)


if __name__ == "__main__":
    fig2(); fig3(); fig4()
    print("wrote figures/paper_fig2_hidden_failure_confirmatory, paper_fig3_persistent_failures_ce, paper_fig4_screening_utility")
