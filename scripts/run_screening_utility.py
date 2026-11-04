"""A3: screening decision utility (Figure 4).  No refitting, no test-label tuning.

Decision rule for a conformal method: a singleton set is an ACCEPTED call of that label; a
full or empty set is a DEFERRAL (send to assay).  Compared at alpha = 0.10 (0.05, 0.20 also):
    marginal, mondrian (class only), fixed_b3 (class x support tertile = mondrian_x), fixed_b2
and two non-conformal rejection baselines MATCHED to fixed_b3's deferral rate on the same split:
    ad_reject    applicability domain: defer the d lowest max-Tanimoto-to-train compounds,
                 accept the argmax label for the rest
    conf_reject  selective classification: defer the d least confident (max prob) compounds
d is fixed_b3's test deferral fraction.  It is computed from set sizes and test features only
(both label-free), so matching uses no test label.  Test labels enter only the metrics.

Metrics (per split; the seed is the unit):
    deferral_rate               fraction deferred
    accepted_error              error rate among accepted calls
    minority_false_accept_rate  BBB- compounds accepted as BBB+, divided by all BBB- compounds
                                (the costly CNS-screening error: a non-permeant compound advanced)
    minority_false_per_1000     same, per 1000 screened compounds
    low_support_accepted_error  accepted error within the low-support tertile (VAL edges)
    minority_deferral_rate      fraction of BBB- compounds deferred

usage: python scripts/run_screening_utility.py
Outputs: tables/screening_utility_per_run.csv, tables/screening_utility_summary.csv,
         tables/screening_utility_paired.csv, figures/paper_fig4_screening_utility.{png,pdf}
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import mean_ci, paired_compare

REPO = Path(__file__).resolve().parents[1]
SETTINGS = {"bbbp/random": "bbbp/random", "bbbp/scaffold": "bbbp/scaffold", "b3db/random": "b3db/random",
            "b3db/scaffold": "b3db/scaffold", "bbbp→b3db/scaffold": "bbbp__to__b3db/scaffold"}
MODELS = ["rf", "gin", "chemprop"]
ALPHAS = (0.05, 0.10, 0.20)


def decision_metrics(accept, label, y, mino, low):
    m = y == mino
    wrong = accept & (label != y)
    fa_min = accept & m & (label != mino)
    return {"deferral_rate": 1 - accept.mean(),
            "accepted_error": wrong.sum() / accept.sum() if accept.any() else np.nan,
            "minority_false_accept_rate": fa_min.sum() / m.sum() if m.any() else np.nan,
            "minority_false_per_1000": 1000 * fa_min.mean(),
            "minority_deferral_rate": 1 - accept[m].mean() if m.any() else np.nan,
            "low_support_accepted_error": (wrong & low).sum() / (accept & low).sum() if (accept & low).any() else np.nan,
            "accepted_n": int(accept.sum())}


def set_shape(S):
    k = S.sum(1)
    return {"singleton_rate": float((k == 1).mean()), "full_rate": float((k == 2).mean()), "empty_rate": float((k == 0).mean())}


def match_reject(score, d):
    """Defer the round(d * n) LOWEST-score compounds (stable order); score is label-free."""
    n = len(score); k = int(round(d * n))
    acc = np.ones(n, bool)
    if k > 0:
        acc[np.argsort(score, kind="stable")[:k]] = False
    return acc


def run_file(path, setting, model, seed):
    z = np.load(path, allow_pickle=True)
    mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    s_cal = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
    bins = {b: (sim_bins(ref, z["sim_cal"], b)[0], sim_bins(ref, z["sim_test"], b)[0]) for b in (2, 3)}
    low = bins[3][1] == 0
    argmax = z["p_test"].argmax(1)
    rows = []
    for a in ALPHAS:
        sets = {}
        for name, base, nb in (("marginal", "marginal", 3), ("mondrian", "mondrian", 3),
                               ("fixed_b3", "mondrian_x", 3), ("fixed_b2", "mondrian_x", 2), ("sabm", "sabm", 3)):
            T = C.CALIBRATORS[base](s_cal, yc, St, a, bin_cal=bins[nb][0], bin_test=bins[nb][1])
            sets[name] = C.predict_sets(St, T)
        for name, S in sets.items():
            acc = S.sum(1) == 1
            rows.append({"setting": setting, "model": model, "seed": seed, "alpha": a, "method": name,
                         **decision_metrics(acc, S.argmax(1), yt, mino, low), **set_shape(S)})
        d = 1 - (sets["fixed_b3"].sum(1) == 1).mean()
        for name, score in (("ad_reject", z["sim_test"]), ("conf_reject", z["p_test"].max(1))):
            acc = match_reject(score, d)
            rows.append({"setting": setting, "model": model, "seed": seed, "alpha": a, "method": name,
                         "matched_to": "fixed_b3", **decision_metrics(acc, argmax, yt, mino, low)})
        # Fully held-out variant: the deferral rate and both rejection thresholds are set on VAL
        # (fixed_b3 sets for VAL molecules use the CAL thresholds; no TEST quantity is used).
        if len(z["sim_val"]):
            Sv = 1.0 - z["p_val"]; bv = sim_bins(ref, z["sim_val"], 3)[0]
            Tv = C.mondrian_x(s_cal, yc, Sv, a, bin_cal=bins[3][0], bin_test=bv)
            d_val = 1 - (C.predict_sets(Sv, Tv).sum(1) == 1).mean()
            for name, sv, st in (("ad_reject_val", z["sim_val"], z["sim_test"]),
                                 ("conf_reject_val", z["p_val"].max(1), z["p_test"].max(1))):
                tau = np.quantile(sv, d_val)
                acc = st > tau
                rows.append({"setting": setting, "model": model, "seed": seed, "alpha": a, "method": name,
                             "matched_to": "fixed_b3 deferral on VAL", **decision_metrics(acc, argmax, yt, mino, low)})
    return rows


def figure(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bbbcp.utils.figio import save
    cols = {"marginal": "#9a9a94", "mondrian": "#2a78d6", "fixed_b3": "#eb6834", "ad_reject": "#1baf7a",
            "conf_reject": "#e87ba4"}
    labs = {"marginal": "marginal CP", "mondrian": "class-only Mondrian", "fixed_b3": "class × support",
            "ad_reject": "AD rejection (matched)", "conf_reject": "confidence rejection (matched)"}
    eps = [("minority_false_accept_rate", "BBB− advanced as BBB+\n(fraction of all BBB−)"),
           ("accepted_error", "Error among accepted calls"),
           ("low_support_accepted_error", "Accepted error, low-support tertile"),
           ("deferral_rate", "Deferral rate")]
    sets_ = list(SETTINGS)
    g = S[(S.alpha == 0.10)].groupby(["setting", "method"])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    yy = np.arange(len(sets_))[::-1]
    for ax, (ep, title) in zip(axes, eps):
        for j, m in enumerate(cols):
            off = (j - 2) * 0.15
            mu = [g.get_group((s, m))[ep].mean() if (s, m) in g.groups else np.nan for s in sets_]
            # backbone spread: min-max of the three backbone means
            bb = S[(S.alpha == 0.10) & (S.method == m)].groupby(["setting", "model"])[ep].mean().unstack()
            lo = [bb.loc[s].min() if s in bb.index else np.nan for s in sets_]
            hi = [bb.loc[s].max() if s in bb.index else np.nan for s in sets_]
            ax.hlines(yy + off, lo, hi, color=cols[m], lw=1.2)
            ax.plot(mu, yy + off, "o", color=cols[m], ms=4, label=labs[m])
        ax.set_title(title, fontsize=8, loc="left"); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(yy); axes[0].set_yticklabels(sets_, fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, out); plt.close(fig)


def main():
    rows = []
    for s, sub in SETTINGS.items():
        for mdl in MODELS:
            for seed in range(100, 120):
                p = REPO / "results/raw" / sub / f"{mdl}_ce" / f"seed{seed}.npz"
                if p.exists():
                    rows += run_file(p, s, mdl, seed)
    D = pd.DataFrame(rows)
    D.to_csv(REPO / "tables/screening_utility_per_run.csv", index=False)
    met = ["deferral_rate", "accepted_error", "minority_false_accept_rate", "minority_false_per_1000",
           "minority_deferral_rate", "low_support_accepted_error", "singleton_rate", "full_rate", "empty_rate"]
    summ = []
    for k, g in D.groupby(["setting", "model", "alpha", "method"]):
        r = dict(zip(["setting", "model", "alpha", "method"], k)); r["n_seeds"] = g.seed.nunique()
        for c in met:
            mu, lo, hi = mean_ci(g[c].values); r[c] = mu; r[c + "_lo"] = lo; r[c + "_hi"] = hi
        summ.append(r)
    pd.DataFrame(summ).to_csv(REPO / "tables/screening_utility_summary.csv", index=False)
    out = []
    for k, g in D.groupby(["setting", "model", "alpha"]):
        P = g.pivot(index="seed", columns="method")
        for x, y in (("fixed_b3", "ad_reject"), ("fixed_b3", "conf_reject"), ("fixed_b3", "mondrian"),
                     ("mondrian", "marginal"), ("fixed_b3", "ad_reject_val"), ("fixed_b3", "conf_reject_val"),
                     ("fixed_b2", "ad_reject"), ("fixed_b3", "sabm")):
            for ep in ("minority_false_accept_rate", "accepted_error", "low_support_accepted_error", "deferral_rate"):
                out.append({**dict(zip(["setting", "model", "alpha"], k)), "contrast": f"{x} - {y}", "endpoint": ep,
                            **paired_compare(P[ep][x].values, P[ep][y].values)})
    pd.DataFrame(out).to_csv(REPO / "tables/screening_utility_paired.csv", index=False)
    figure(D, REPO / "figures/paper_fig4_screening_utility")
    pd.set_option("display.width", 220)
    v = pd.DataFrame(summ); v = v[v.alpha == 0.10]
    for ep in ("deferral_rate", "minority_false_accept_rate", "accepted_error", "low_support_accepted_error"):
        print(f"\n== {ep} (alpha 0.10) ==")
        print(v.pivot_table(index=["setting", "model"], columns="method", values=ep).round(3).to_string())
    pr = pd.DataFrame(out)
    pr = pr[(pr.alpha == 0.10) & (pr.endpoint == "minority_false_accept_rate")]
    print("\n== paired, minority false-accept rate, alpha 0.10 ==")
    print(pr[["setting", "model", "contrast", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
