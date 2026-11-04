"""Items F42/F43: our numbers under Tursunbadalov & Tursunbadalov's protocol vs their reported values.

Their reported BBBP values (arXiv 2607.06605v1, alpha = 0.1, LAC, 10 random 50/25/25 splits):
  RF       marginal minority 64.8% (Table 5: 64.1%), Mondrian 90.3% (Table 5: 90.0%);
           majority 98.1% / 89.6%; set size 1.05 / 1.26; marginal acceptance 95%
  GCN      marginal minority 63.3%, Mondrian 90.9%
  ChemBERTa marginal minority 72.4%, Mondrian 90.1%

usage: python scripts/compare_tursunbadalov.py [results/processed/tursun_repro_v13_conformal.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.stats.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]
THEIRS = {("rf", "marginal"): dict(cov_minority=0.648, cov_majority=0.981, set_size=1.05, acceptance_rate=0.95),
          ("rf", "mondrian"): dict(cov_minority=0.903, cov_majority=0.896, set_size=1.26),
          ("gcn", "marginal"): dict(cov_minority=0.633), ("gcn", "mondrian"): dict(cov_minority=0.909),
          ("chemberta", "marginal"): dict(cov_minority=0.724), ("chemberta", "mondrian"): dict(cov_minority=0.901)}


def main():
    cands = sys.argv[1:] or [str(p) for p in sorted((REPO / "results/processed").glob("tursun_repro*_v13_conformal.csv*"))]
    df = pd.concat([pd.read_csv(c) for c in cands], ignore_index=True)
    d = df[np.isclose(df.alpha, 0.1) & df.score.isin(["lac", "aps"]) & df.method.isin(["marginal", "mondrian", "sabm", "mondrian_x"])]
    rows = []
    for (ds, mdl, score, meth), g in d.groupby(["dataset", "model_name", "score", "method"]):
        r = {"dataset": ds, "model": mdl, "score": score, "method": meth, "n_seeds": g.seed.nunique()}
        for m in ("cov_minority", "cov_majority", "set_size", "acceptance_rate", "mcg", "cii", "cov_min_bin0"):
            mu, lo, hi = mean_ci(g[m].values)
            r[m] = mu; r[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        th = THEIRS.get((mdl, meth), {}) if score == "lac" else {}
        for k, v in th.items():
            r[f"theirs_{k}"] = v
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "tables/tursunbadalov_comparison.csv", index=False)
    show = ["dataset", "model", "score", "method", "n_seeds", "cov_minority", "theirs_cov_minority", "cov_majority",
            "theirs_cov_majority", "set_size", "theirs_set_size", "acceptance_rate", "cov_min_bin0"]
    print(out[[c for c in show if c in out]].round(3).to_string())


if __name__ == "__main__":
    main()
