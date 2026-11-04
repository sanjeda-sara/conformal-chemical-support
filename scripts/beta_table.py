"""Analytic Beta-law table for class-conditional coverage given the calibration set
(Vovk 2012).  Writes tables/beta_law_table.csv."""
import math
import pandas as pd
from scipy.stats import beta
from bbbcp.conformal.calibrators import n_required, pac_rank
from bbbcp.pipeline import REPO

rows = []
for a in (0.05, 0.10, 0.20):
    for n in (10, 20, 40, 80, 160, 320, 640):
        k = math.ceil((n + 1) * (1 - a) - 1e-12)
        if k > n:
            rows.append({"alpha": a, "n_c": n, "k": None, "note": "threshold = +inf (full set for this class)"}); continue
        B = beta(k, n + 1 - k)
        kp = pac_rank(n, a, 0.10)
        rows.append({"alpha": a, "n_c": n, "k": k, "mean_cov": B.mean(), "sd_cov": B.std(),
                     "p05_cov": B.ppf(0.05), "P_cov_below_target_minus_0.05": B.cdf(1 - a - 0.05),
                     "P_cov_below_target": B.cdf(1 - a), "pac_rank_delta0.1": kp,
                     "n_required_eps0.05_delta0.1": n_required(a, 0.05),
                     "n_required_epsalpha_delta0.1": n_required(a)})
df = pd.DataFrame(rows); df.to_csv(REPO / "tables/beta_law_table.csv", index=False)
print(df.round(3).to_string())
