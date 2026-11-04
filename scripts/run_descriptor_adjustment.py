"""Does the TPSA/HBD reliability disparity survive adjustment for chemical support?
(Reviewer objection H / ranked fix #7.)

The pilot shows that under Mondrian, \bbbm\ molecules with low TPSA and few H-bond donors are
the least covered.  Low-TPSA molecules are also, in these datasets, less similar to the
training set.  If the descriptor association is entirely explained by chemical support, it is
not an independent finding and must not be reported as one.

Three analyses, all on cached Stage A predictions (no refitting), all label-free in their
inputs except the coverage outcome:

  1. STRATIFIED.  Miss rate by TPSA bin WITHIN each Tanimoto-support tertile, with Wilson
     intervals and counts.  If the TPSA gradient vanishes inside support strata, support
     explains it.
  2. ADJUSTED LOGISTIC.  miss ~ TPSA + HBD + support (+ MW, cLogP), standardized, with a
     cluster bootstrap over split seeds.  Reports each coefficient with and without the
     support term so the attenuation is visible.
  3. CONDITIONAL PERMUTATION.  Permute TPSA within support tertiles and recompute the
     TPSA coefficient; the null distribution answers "is there TPSA signal beyond support?"
     without assuming the logistic model is correctly specified.

usage:
  python scripts/run_descriptor_adjustment.py
  python scripts/run_descriptor_adjustment.py --runs "results/raw/bbbp/*/rf_ce/seed1??.npz" --method mondrian
Outputs: tables/descriptor_adjustment_{stratified,coefficients,permutation}.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bbbcp.conformal import calibrators as C
from bbbcp.conformal.scores import score_matrix
from bbbcp.evaluate import sim_bins
from bbbcp.pipeline import REPO, features, prepare
from bbbcp.stats.stats import wilson

DESC = {"TPSA": 2, "HBD": 3, "MW": 0, "cLogP": 1}
TPSA_EDGES = (60.0, 90.0)


def collect(runs, method="mondrian", alpha=0.10, kind="lac", minority=0):
    rows = []
    for path in sorted(glob.glob(str(REPO / runs))):
        z = np.load(path, allow_pickle=False)
        if "minority" in z.files:
            minority = int(z["minority"])
        tag = str(path).split("/raw/")[1]
        src = tag.split("/")[0].split("__to__")
        dtest = src[-1]
        idx = z["idx_test"]
        desc = features(dtest)["desc"][idx]
        y_cal, y_test = z["y_cal"].astype(int), z["y_test"].astype(int)
        Sc = score_matrix(z["p_cal"], kind); St = score_matrix(z["p_test"], kind)
        s_cal = Sc[np.arange(len(y_cal)), y_cal]
        ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
        bc, _ = sim_bins(ref, z["sim_cal"]); bt, _ = sim_bins(ref, z["sim_test"])
        T = C.CALIBRATORS[method](s_cal, y_cal, St, alpha, bin_cal=bc, bin_test=bt)
        hit = C.predict_sets(St, T)[np.arange(len(y_test)), y_test]
        m = y_test == minority
        for i in np.where(m)[0]:
            rows.append({"setting": "/".join(tag.split("/")[:2]), "model": tag.split("/")[2],
                         "seed": int(str(path).split("seed")[-1].split(".")[0]),
                         "miss": int(~hit[i]), "support": float(z["sim_test"][i]),
                         "support_bin": int(bt[i]), "tpsa_bin": int(np.digitize(desc[i, 2], TPSA_EDGES)),
                         **{k: float(desc[i, j]) for k, j in DESC.items()}})
    return pd.DataFrame(rows)


def stratified(df):
    out = []
    lab = {0: "<60", 1: "60-90", 2: ">90"}
    for (st, sb, tb), g in df.groupby(["setting", "support_bin", "tpsa_bin"]):
        k, n = int((~g.miss.astype(bool)).sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"setting": st, "support_bin": sb, "tpsa_bin": lab[tb], "n": n,
                    "coverage": k / n if n else np.nan, "wilson_lo": lo, "wilson_hi": hi,
                    "miss_rate": g.miss.mean()})
    o = pd.DataFrame(out)
    piv = o.pivot_table(index=["setting", "support_bin"], columns="tpsa_bin", values="coverage")
    if "<60" in piv and ">90" in piv:
        piv["gradient(<60 minus >90)"] = piv["<60"] - piv[">90"]
    return o, piv


def coefficients(df, B=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for st, g in df.groupby("setting"):
        seeds = g.seed.unique()
        for name, cols in (("unadjusted", ["TPSA", "HBD"]),
                           ("adjusted for support", ["TPSA", "HBD", "support"]),
                           ("adjusted for support + size/lipo", ["TPSA", "HBD", "support", "MW", "cLogP"])):
            mu, sd = g[cols].mean(), g[cols].std().replace(0, 1)
            coefs = []
            for _ in range(B):
                h = pd.concat([g[g.seed == s] for s in rng.choice(seeds, len(seeds), replace=True)])
                if h.miss.nunique() < 2:
                    continue
                X = (h[cols] - mu) / sd
                coefs.append(LogisticRegression(C=1e4, max_iter=3000).fit(X, h.miss).coef_[0])
            if not coefs:
                continue
            coefs = np.array(coefs)
            for j, c in enumerate(cols):
                out.append({"setting": st, "model_spec": name, "term": c,
                            "coef": coefs[:, j].mean(), "ci_lo": np.quantile(coefs[:, j], 0.025),
                            "ci_hi": np.quantile(coefs[:, j], 0.975),
                            "excludes_zero": bool(np.quantile(coefs[:, j], 0.025) > 0 or np.quantile(coefs[:, j], 0.975) < 0),
                            "n_boot": len(coefs), "n_molecules": len(g)})
    return pd.DataFrame(out)


def conditional_permutation(df, n_perm=500, seed=0):
    """Permute TPSA WITHIN support tertiles: destroys TPSA signal, keeps the support
    relationship. A TPSA coefficient outside the null band is signal beyond support."""
    rng = np.random.default_rng(seed)
    out = []
    for st, g in df.groupby("setting"):
        cols = ["TPSA", "HBD", "support"]
        mu, sd = g[cols].mean(), g[cols].std().replace(0, 1)
        X = (g[cols] - mu) / sd
        obs = LogisticRegression(C=1e4, max_iter=3000).fit(X, g.miss).coef_[0][0]
        null = []
        for _ in range(n_perm):
            h = g.copy()
            for b, idx in h.groupby("support_bin").groups.items():
                vals = h.loc[idx, "TPSA"].values.copy(); rng.shuffle(vals)
                h.loc[idx, "TPSA"] = vals
            Xp = (h[cols] - mu) / sd
            null.append(LogisticRegression(C=1e4, max_iter=3000).fit(Xp, h.miss).coef_[0][0])
        null = np.array(null)
        out.append({"setting": st, "tpsa_coef_observed": obs,
                    "null_lo": np.quantile(null, 0.025), "null_hi": np.quantile(null, 0.975),
                    "p_two_sided": float((np.abs(null) >= abs(obs)).mean()), "n_perm": n_perm,
                    "signal_beyond_support": bool((np.abs(null) >= abs(obs)).mean() < 0.05)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/raw/{bbbp,b3db}/{random,scaffold}/rf_ce/seed1??.npz")  # confirmatory seeds only
    ap.add_argument("--method", default="mondrian"); ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--tag", default="", help="suffix for output tables")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from run_adaptive_vs_fixed import _expand
    df = pd.concat([collect(p, a.method, a.alpha) for p in _expand(a.runs)], ignore_index=True)
    print(f"{len(df)} minority test molecules over {df.seed.nunique()} seeds, {df.setting.nunique()} settings")
    strat, piv = stratified(df)
    strat.to_csv(REPO / f"tables/descriptor_adjustment_stratified{a.tag}.csv", index=False)
    co = coefficients(df); co.to_csv(REPO / f"tables/descriptor_adjustment_coefficients{a.tag}.csv", index=False)
    pe = conditional_permutation(df, a.perms); pe.to_csv(REPO / f"tables/descriptor_adjustment_permutation{a.tag}.csv", index=False)
    print("\n=== 1. Coverage by TPSA bin WITHIN each support tertile ===")
    print(piv.round(3).to_string())
    print("\n=== 2. Logistic coefficients for MISS (positive = more likely missed) ===")
    print(co[co.term.isin(["TPSA", "support"])].round(3).to_string(index=False))
    print("\n=== 3. TPSA signal beyond support (permutation within support tertiles) ===")
    print(pe.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
