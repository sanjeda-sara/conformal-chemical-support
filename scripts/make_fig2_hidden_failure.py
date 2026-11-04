"""Figure 2 (confirmatory): BBB- coverage by support tertile, class-only Mondrian vs class x support,
three backbones, four exchangeable settings.  LAC, alpha 0.10, CE, seeds 100-119, mean and 95% t-CI.
Source: tables/confirmatory_summary.csv (external = none).  Every plotted mean is a key verified by
scripts/verify_paper_numbers.py (t.<setting>.<model>.<method>.cov_min_bin{0,2}) except the middle
tertile, which is plotted for context and recomputed by the same verifier logic.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bbbcp.utils.figio import save

REPO = Path(__file__).resolve().parents[1]
S = pd.read_csv(REPO / "tables/confirmatory_summary.csv", low_memory=False)
S = S[(S.score == "lac") & (S.alpha == 0.1) & (S.loss == "ce") & S.external.isna() & (S.n_seeds == 20)]
SET = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MOD = [("rf", "RF"), ("gin", "GIN"), ("chemprop", "Chemprop")]
COL = {"mondrian": "#2a78d6", "mondrian_x": "#eb6834"}
LAB = {"mondrian": "class-only Mondrian", "mondrian_x": "class × support"}
fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
for ax, st in zip(axes, SET):
    for j, (md, mlab) in enumerate(MOD):
        for k, meth in enumerate(("mondrian", "mondrian_x")):
            r = S[(S.setting == st) & (S.model_name == md) & (S.method == meth)].iloc[0]
            x = j * 4 + np.arange(3) * 1.15 + (k - 0.5) * 0.42
            mu = [r[f"cov_min_bin{b}"] for b in range(3)]
            lo = [r[f"cov_min_bin{b}_lo"] for b in range(3)]; hi = [r[f"cov_min_bin{b}_hi"] for b in range(3)]
            ax.errorbar(x, mu, yerr=[np.subtract(mu, lo), np.subtract(hi, mu)], fmt="o", ms=3.5, color=COL[meth],
                        capsize=1.5, lw=1, label=LAB[meth] if (j == 0 and st == SET[0]) else None)
    ax.axhline(0.9, ls="--", lw=0.7, color="#444")
    ax.set_xticks([j * 4 + 1.15 for j in range(3)]); ax.set_xticklabels([m[1] for m in MOD], fontsize=7)
    for j in range(3):
        for b, t in enumerate(("low", "mid", "high")):
            ax.text(j * 4 + b * 1.15, 0.665, t, ha="center", fontsize=5.5, color="#6b6b66")
    ax.set_title(st, fontsize=8, loc="left"); ax.set_ylim(0.65, 1.0); ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("BBB− coverage (mean, 95% CI, 20 splits)", fontsize=7)
fig.legend(fontsize=7, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, REPO / "figures/paper_fig2_hidden_failure_confirmatory"); plt.close(fig)
print("wrote figures/paper_fig2_hidden_failure_confirmatory.(png,pdf)")
