"""D3 + D4: do existing conditional / localized conformal methods already remove the hidden
low-support BBB- failure?  And are post-hoc recalibrations (temperature scaling, logit
adjustment) exact no-ops for class-conditional sets, as Proposition S3 states?

Reads tables/confirmatory_summary.csv only (Stage B already evaluated every method on the
same cached predictions).  No refitting.  EXPLORATORY; not part of the 36-member family.
Several methods are our own implementations of the published ideas (see the 'note' column);
they are labelled as such and no superiority claim is made over the authors' code.

usage: python scripts/run_conditional_baselines.py
Outputs: tables/conditional_baselines.csv, tables/posthoc_invariance_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# method -> (label, implementation note)
METHODS = {
    "marginal": ("marginal split CP", "standard"),
    "mondrian": ("class-conditional (Mondrian)", "standard"),
    "mondrian_x": ("class x support, fixed tertiles", "standard Mondrian taxonomy"),
    "mondrian_pac": ("Mondrian, training-conditional (PAC)", "Vovk 2012 rank rule"),
    "mondrian_ssbc": ("Mondrian, SSBC-style Beta rank", "our implementation"),
    "clustered_cells": ("clustered CP over class x support cells", "our implementation of Ding et al. 2023"),
    "kandinsky_qr": ("Kandinsky-style group QR", "our approximation, not the authors' algorithm"),
    "condqr_sim": ("conditional QR on support", "our approximation of Gibbs et al. 2025"),
    "interp_q50": ("INTERP-Q (q=0.5)", "our implementation of Ding et al. 2026"),
    "interp_q90": ("INTERP-Q (q=0.9)", "our implementation of Ding et al. 2026"),
    "rc3p": ("RC3P", "our implementation"),
    "lcp_tanimoto": ("localized CP, Tanimoto kernel", "our implementation of Guan 2023"),
    "lcp_knn": ("localized CP, kNN", "our implementation"),
    "rlcp_sim": ("randomly localized CP", "our implementation of Hore and Barber 2025"),
    "weighted_mondrian": ("weighted Mondrian (density ratio)", "our approximation"),
    "kmm_mondrian": ("KMM-weighted Mondrian", "our approximation, not released KMM-CP"),
    "macro_cov": ("macro-coverage CP", "our implementation"),
}
INVARIANCE = [("mondrian", "mondrian_ts"), ("mondrian", "mondrian_la"), ("mondrian_x", "mondrian_x_ts"),
              ("mondrian_x", "mondrian_x_la"), ("sabm", "sabm_ts"), ("sabm", "sabm_la")]


def main():
    S = pd.read_csv(REPO / "tables/confirmatory_summary.csv", low_memory=False)
    S = S[(S.n_seeds == 20) & (S.alpha == 0.1) & (S.loss == "ce") & S.external.isna()
          & S.regime.isin(["random", "scaffold"])]
    lac = S[S.score == "lac"]
    rows = []
    for m, (lab, note) in METHODS.items():
        g = lac[lac.method == m]
        if g.empty:
            print(f"WARNING: {m} not in summary at LAC; skipped"); continue
        if len(g) != 12:
            print(f"WARNING: {m} has {len(g)} cells, expected 12")
        rows.append({"method": m, "label": lab, "note": note, "n_cells": len(g),
                     "cov_minority_mean": g.cov_minority.mean(),
                     "low_mean": g.cov_min_bin0.mean(), "low_min": g.cov_min_bin0.min(), "low_max": g.cov_min_bin0.max(),
                     "high_mean": g.cov_min_bin2.mean(),
                     "n_cells_low_ci_below_090": int((g.cov_min_bin0_hi < 0.9).sum()),
                     "set_size_mean": g.set_size.mean(), "set_size_min": g.set_size.min(), "set_size_max": g.set_size.max()})
    R = pd.DataFrame(rows)
    R.to_csv(REPO / "tables/conditional_baselines.csv", index=False)
    inv = []
    for a, b in INVARIANCE:
        A = lac[lac.method == a].set_index(["setting", "model_name"]); B = lac[lac.method == b].set_index(["setting", "model_name"])
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        if j.empty:
            print(f"WARNING: no overlap for {a} vs {b}"); continue
        inv.append({"base": a, "posthoc": b, "n_cells": len(j),
                    "max_abs_diff_low": float((j.cov_min_bin0_a - j.cov_min_bin0_b).abs().max()),
                    "max_abs_diff_set_size": float((j.set_size_a - j.set_size_b).abs().max()),
                    "max_abs_diff_cov_minority": float((j.cov_minority_a - j.cov_minority_b).abs().max())})
    # contrast: marginal sets.  For binary LAC, temperature scaling maps both classes' scores through the
    # same monotone function, so marginal sets are invariant to it too; logit adjustment shifts the two
    # classes differently, so marginal sets are NOT invariant to it.
    A = lac[lac.method == "marginal"].set_index(["setting", "model_name"])
    for ph in ("marginal_ts", "marginal_la"):
      B = lac[lac.method == ph].set_index(["setting", "model_name"])
      j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
      if len(j):
        inv.append({"base": "marginal", "posthoc": ph, "n_cells": len(j),
                    "max_abs_diff_low": float((j.cov_min_bin0_a - j.cov_min_bin0_b).abs().max()),
                    "max_abs_diff_set_size": float((j.set_size_a - j.set_size_b).abs().max()),
                    "max_abs_diff_cov_minority": float((j.cov_minority_a - j.cov_minority_b).abs().max())})
    I = pd.DataFrame(inv)
    I.to_csv(REPO / "tables/posthoc_invariance_check.csv", index=False)
    pd.set_option("display.width", 220)
    print(R[["method", "n_cells", "cov_minority_mean", "low_mean", "low_min", "low_max", "high_mean",
             "n_cells_low_ci_below_090", "set_size_mean"]].round(3).to_string(index=False))
    print("\nPost-hoc invariance (expected 0 for class-conditional rows, nonzero allowed for marginal):")
    print(I.to_string(index=False))


if __name__ == "__main__":
    main()