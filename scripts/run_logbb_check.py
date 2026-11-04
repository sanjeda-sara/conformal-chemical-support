"""D6: are persistently missed BBB- compounds borderline labels?  B3DB assigns BBB+ when
logBB >= -1.  If persistent failures sit near -1, part of the 'failure' is threshold noise.

Uses tables/persistent_failure_molecules_ce.csv (run scripts/run_persistent_failures.py first).
Only molecules with a numeric logBB can be used: own B3DB record (B3DB settings and transfer) or
the matching B3DB record of a BBBP molecule.  Descriptive; reports n for every group.

usage: python scripts/run_logbb_check.py
Output: tables/logbb_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
THRESH = -1.0


def main():
    W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
    W = W[W.eligible.astype(bool)].copy()
    W["logbb_any"] = W.logBB.where(W.logBB.notna(), W.logBB_in_other)
    rows = []
    for st, g in W.groupby("setting", sort=False):
        groups = {"cross_arch_strict": g[g.cross_arch_mondrian.astype(bool)],
                  "majority_all3": g[g.majority_all3_mondrian.astype(bool)],
                  "rest": g[~g.majority_all3_mondrian.astype(bool)]}
        for name, h in groups.items():
            v = h.logbb_any.dropna().values
            r = {"setting": st, "group": name, "n_molecules": len(h), "n_with_logBB": len(v),
                 "median_logBB": np.median(v) if len(v) else np.nan,
                 "frac_within_0.5_of_threshold": float(np.mean(v > THRESH - 0.5)) if len(v) else np.nan}
            if name == "majority_all3":
                rest = groups["rest"].logbb_any.dropna().values
                r["mw_p_vs_rest"] = mannwhitneyu(v, rest).pvalue if len(v) >= 5 and len(rest) >= 5 else np.nan
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(REPO / "tables/logbb_check.csv", index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
