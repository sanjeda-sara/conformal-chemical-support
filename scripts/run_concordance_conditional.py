"""Cross-architecture concordance against a covariate-stratified null (EXPLORATORY, added 1 Nov 2026).

The primary null permutes each backbone's persistence indicator within strata of the number of test
appearances k.  Backbones share molecules, labels and splits, so some agreement is expected from shared
difficulty.  This null permutes within strata of k x support tertile x TPSA tertile (tertiles computed
within each setting over eligible BBB- molecules), so agreement explained by those covariates is absorbed.
Also reports the appearance-count distribution of strict failures and the count when k >= 3 or k >= 4 is
required.  Reads tables/persistent_failure_molecules_ce.csv (verified by verify_persistent_failures.py).

usage: python scripts/run_concordance_conditional.py [--perms 10000]
Output: tables/concordance_conditional.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MODELS = ["rf", "gin", "chemprop"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=10000); a = ap.parse_args()
    W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
    W = W[W.eligible.astype(bool)].copy()
    rng = np.random.default_rng(20261101)
    rows = []
    for st, g in W.groupby("setting", sort=False):
        g = g.copy()
        k = g.n_appear.clip(upper=6).astype(int).values
        sup = pd.qcut(g.mean_sim_train.rank(method="first"), 3, labels=False).values
        tp = pd.qcut(g.TPSA.rank(method="first"), 3, labels=False).values
        P = np.column_stack([g[f"persist_mondrian__{m}"].astype(bool).values for m in MODELS])
        obs = int(P.all(1).sum())
        res = {}
        for name, strata in (("k", k), ("k_support_tpsa", k * 9 + sup * 3 + tp)):
            groups = [np.where(strata == s)[0] for s in np.unique(strata)]
            null = np.empty(a.perms)
            for i in range(a.perms):
                Q = P.copy()
                for j in range(3):
                    for idx in groups:
                        Q[idx, j] = Q[rng.permutation(idx), j]
                null[i] = Q.all(1).sum()
            res[name] = (null.mean(), np.quantile(null, 0.975), (1 + (null >= obs).sum()) / (1 + a.perms))
        strict = g[P.all(1)]
        rows.append({"setting": st, "n_eligible": len(g), "n_cross_arch": obs,
                     "null_k_mean": res["k"][0], "null_k_q975": res["k"][1], "p_k": res["k"][2],
                     "null_cov_mean": res["k_support_tpsa"][0], "null_cov_q975": res["k_support_tpsa"][1],
                     "p_cov": res["k_support_tpsa"][2],
                     "excess_ratio_cov": obs / res["k_support_tpsa"][0] if res["k_support_tpsa"][0] > 0 else np.nan,
                     "k_median": float(strict.n_appear.median()) if len(strict) else np.nan,
                     "k_min": float(strict.n_appear.min()) if len(strict) else np.nan,
                     "k_max": float(strict.n_appear.max()) if len(strict) else np.nan,
                     "n_k2": int((strict.n_appear == 2).sum()), "n_kge3": int((strict.n_appear >= 3).sum()),
                     "n_kge4": int((strict.n_appear >= 4).sum())})
    R = pd.DataFrame(rows); R.to_csv(REPO / "tables/concordance_conditional.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
