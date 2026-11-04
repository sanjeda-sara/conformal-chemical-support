"""A5: molecule- and scaffold-cluster bootstrap of the PRE-SPECIFIED 36-member primary family.

Objection answered.  The 20 confirmatory splits re-use the same molecules, so seed-level
Wilcoxon tests treat overlapping test sets as independent replicates.  Here the resampling unit
is the TEST MOLECULE (or its Murcko scaffold; acyclic molecules are their own cluster), drawn
once and applied to every split in which it appears.  Each split's calibration and thresholds
are held fixed (inference is conditional on the calibration draws).

Family (unchanged, see docs/STATISTICAL_ANALYSIS_PLAN.md amendment):
    endpoint   minority (BBB-) coverage in the low-support tertile (cov_min_bin0)
    score LAC, alpha 0.10, loss CE, seeds 100-119
    settings   bbbp/random, bbbp/scaffold, b3db/random, b3db/scaffold   (external = none)
    models     rf, gin, chemprop
    contrasts  mondrian_x - mondrian; sabm - mondrian; sabm - mondrian_x       -> 36 members
Estimate = mean over seeds of the per-seed difference, exactly as in the seed-level analysis.
Two-sided bootstrap p = min(1, 2 * min(P*(d <= 0), P*(d >= 0))), with (1 + count)/(1 + B).
Holm across all 36, separately for each resampling unit.  A contrast that is identically zero
in every split (sabm - mondrian_x) gets p = 1.

usage: python scripts/run_cluster_bootstrap.py [--B 20000]
Output: tables/primary_cluster_bootstrap.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import holm

REPO = Path(__file__).resolve().parents[1]
SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]
CONTRASTS = [("mondrian_x", "mondrian"), ("sabm", "mondrian"), ("sabm", "mondrian_x")]


def per_molecule_hits(setting, model, alpha=0.10):
    rows = []
    for seed in range(100, 120):
        p = REPO / "results/raw" / setting / f"{model}_ce" / f"seed{seed}.npz"
        z = np.load(p, allow_pickle=True)
        mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
        ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
        bc, _ = sim_bins(ref, z["sim_cal"]); bt, _ = sim_bins(ref, z["sim_test"])
        s_cal = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
        keep = (yt == mino) & (bt == 0)
        hits = {}
        for m in ("mondrian", "mondrian_x", "sabm"):
            T = C.CALIBRATORS[m](s_cal, yc, St, alpha, bin_cal=bc, bin_test=bt)
            hits[m] = C.predict_sets(St, T)[np.arange(len(yt)), yt][keep].astype(float)
        rows.append(pd.DataFrame({"seed": seed, "mol_id": z["test_ids"].astype(str)[keep],
                                  "scaffold": z["test_scaffold"].astype(str)[keep], **hits}))
    return pd.concat(rows, ignore_index=True)


def boot(H, unit, B, rng):
    """H: per (seed, molecule) rows.  Returns observed per-contrast estimate and bootstrap draws."""
    key = H["mol_id"] if unit == "molecule" else np.where(H.scaffold == "", "acyc:" + H.mol_id, H.scaffold)
    units, u_idx = np.unique(key, return_inverse=True)
    seeds, s_idx = np.unique(H.seed, return_inverse=True)
    U, S = len(units), len(seeds)
    N = np.zeros((U, S)); np.add.at(N, (u_idx, s_idx), 1)
    sums = {}
    for m in ("mondrian", "mondrian_x", "sabm"):
        A = np.zeros((U, S)); np.add.at(A, (u_idx, s_idx), H[m].values); sums[m] = A
    W = rng.multinomial(U, np.full(U, 1 / U), size=B).astype(np.float32)   # B x U
    den = W @ N
    out = {}
    for x, y in CONTRASTS:
        D = sums[x] - sums[y]
        obs = np.nanmean(np.where(N.sum(0) > 0, D.sum(0) / np.maximum(N.sum(0), 1), np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            per_seed = (W @ D) / den
        draws = np.nanmean(per_seed, axis=1)
        out[f"{x} - {y}"] = (obs, draws, U)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261029)
    rows = []
    for s in SETTINGS:
        for mdl in MODELS:
            H = per_molecule_hits(s, mdl)
            for unit in ("molecule", "scaffold"):
                for comp, (obs, d, U) in boot(H, unit, a.B, rng).items():
                    if np.allclose(d, 0) and abs(obs) < 1e-12:
                        p = 1.0
                    else:
                        p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (1 + len(d)), (1 + (d >= 0).sum()) / (1 + len(d))))
                    rows.append({"setting": s, "model": mdl, "comparison": comp, "unit": unit, "n_units": U,
                                 "n_rows": len(H), "mean_diff": obs, "ci_lo": np.nanpercentile(d, 2.5),
                                 "ci_hi": np.nanpercentile(d, 97.5), "boot_p": p, "B": a.B})
            print(f"done {s} {mdl}: {H.mol_id.nunique()} unique low-support BBB- molecules, {len(H)} rows", flush=True)
    R = pd.DataFrame(rows)
    for unit, g in R.groupby("unit"):
        R.loc[g.index, "holm_p"] = holm(g.boot_p.values)
        R.loc[g.index, "family_size"] = len(g)
    R["significant_holm_0.05"] = R.holm_p < 0.05
    R.to_csv(REPO / "tables/primary_cluster_bootstrap.csv", index=False)
    seed_level = pd.read_csv(REPO / "tables/confirmatory_paired_primary.csv")
    M = R.merge(seed_level[["setting", "model", "comparison", "mean_diff", "holm_p"]],
                on=["setting", "model", "comparison"], suffixes=("", "_seedlevel"))
    pd.set_option("display.width", 220)
    print(M[M.comparison == "mondrian_x - mondrian"][["setting", "model", "unit", "n_units", "mean_diff",
          "mean_diff_seedlevel", "ci_lo", "ci_hi", "boot_p", "holm_p", "holm_p_seedlevel"]].round(4).to_string(index=False))
    print(R.groupby(["unit", "comparison"])["significant_holm_0.05"].sum())


if __name__ == "__main__":
    main()
