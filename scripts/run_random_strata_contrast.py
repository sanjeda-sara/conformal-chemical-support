"""Same-cell-size control for the repair (EXPLORATORY, added 1 Nov 2026 after internal review).

Splitting each class into three calibration cells makes every threshold more conservative, whatever the
cells are.  Part of class x support's gain over class-conditional calibration could therefore come from
smaller cells rather than from targeting support.  The fair contrast holds the number and size of cells
fixed: class x Tanimoto-tertile cells (mondrian_x) against class x randomly assigned strata of equal size
(mondrian_x_random), both scored on the same lowest VAL Tanimoto tertile.

Predictions: random strata raise coverage in every tertile (uniform conservativeness); Tanimoto strata
move coverage from high- to low-support compounds.

Per setting x backbone (LAC, alpha 0.10, CE, seeds 100-119, external = none): paired difference in
low-support BBB- coverage, seed bootstrap CI, Wilcoxon signed-rank (zsplit) and Holm over the 12 cells.
Reads per-seed Stage B rows (results/processed/full_v13_conformal.csv); no refitting.

usage: python scripts/run_random_strata_contrast.py
Output: tables/random_strata_contrast.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.stats.stats import holm, paired_compare

REPO = Path(__file__).resolve().parents[1]

def stageb_csv(name="full_v13_conformal"):
    """Stage B per-seed rows; the public repository ships them gzipped (pandas reads either)."""
    for p in (REPO / f"results/processed/{name}.csv", REPO / f"results/processed/{name}.csv.gz"):
        if p.exists():
            return p
    raise FileNotFoundError(f"results/processed/{name}.csv(.gz) not found; run Stage B (scripts/run_grid.py)")

METHODS = ["mondrian", "mondrian_x", "mondrian_x_random"]


def main():
    f = pd.read_csv(stageb_csv(), low_memory=False,
                    usecols=["dataset", "external", "regime", "model_name", "loss", "score", "method", "alpha", "seed",
                             "cov_min_bin0", "cov_min_bin2", "set_size"])
    f = f[f.seed.between(100, 119) & f.external.isna() & f.regime.isin(["random", "scaffold"]) & (f.loss == "ce")
          & (f.score == "lac") & np.isclose(f.alpha, 0.1) & f.method.isin(METHODS)]
    rows = []
    for (ds, rg, md), g in f.groupby(["dataset", "regime", "model_name"], sort=True):
        w = {m: g[g.method == m].set_index("seed").sort_index() for m in METHODS}
        assert all(len(v) == 20 for v in w.values()), (ds, rg, md)
        pc = paired_compare(w["mondrian_x"].cov_min_bin0.values, w["mondrian_x_random"].cov_min_bin0.values)
        r = {"setting": f"{ds}/{rg}", "model": md, **pc}
        for m in METHODS:
            for c in ("cov_min_bin0", "cov_min_bin2", "set_size"):
                r[f"{m}.{c}"] = w[m][c].mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    R["holm_p"] = holm(R.wilcoxon_p.values); R["family_size"] = len(R)
    R.to_csv(REPO / "tables/random_strata_contrast.csv", index=False)
    pd.set_option("display.width", 200)
    print(R[["setting", "model", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p",
             "mondrian_x_random.cov_min_bin0", "mondrian_x_random.cov_min_bin2", "mondrian.cov_min_bin2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
