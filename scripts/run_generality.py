"""D2 and R2: does class-conditional calibration mask support-dependent minority undercoverage
outside our BBB setup?
  D2  other imbalanced MoleculeNet tasks (BACE, ClinTox CT_TOX, Tox21 SR-ARE), RF, random and
      scaffold splits, seeds 100-119 (Stage A: configs/experiments/generality.yaml)
  R2  the protocol of Tursunbadalov & Tursunbadalov (BBBP, all parseable records, random
      50/25/25, RF and GCN), seeds 100-119 (Stage A: configs/experiments/tursun_confirm.yaml).
      That protocol has no VAL split, so support-tertile edges come from CAL similarities
      (flagged in the output; a deviation from our main protocol).

Per run x model: minority coverage (marginal, class-conditional, class x support), low/high
support-tertile minority coverage, set size, a random-strata control, and the paired
class x support - class-conditional low-support gain with a Wilcoxon test; Holm across all
cells in this invocation.  The minority label is read from each cache (y=1 for BACE/ClinTox/
Tox21, y=0 for BBBP).  EXPLORATORY: not part of the pre-specified 36-member family.

usage:
  python scripts/run_generality.py --runs bace/random bace/scaffold clintox/random clintox/scaffold \
         tox21_sr_are/random tox21_sr_are/scaffold --models rf --tag generality
  python scripts/run_generality.py --runs bbbp_raw/random5025 --models rf gcn --tag tursun_confirm
Outputs: tables/<tag>_summary.csv, tables/<tag>_paired.csv
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.stats.stats import holm, mean_ci, paired_compare

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.10


def split_metrics(path):
    z = np.load(path, allow_pickle=True)
    mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
    has_val = len(z["sim_val"]) > 0
    ref = z["sim_val"] if has_val else z["sim_cal"]
    bc, _ = sim_bins(ref, z["sim_cal"]); bt, _ = sim_bins(ref, z["sim_test"])
    rng = np.random.default_rng(zlib.crc32(str(path).encode()))
    rc = rng.integers(0, 3, len(yc)); rt = rng.integers(0, 3, len(yt))       # random-strata control
    sc = 1.0 - z["p_cal"][np.arange(len(yc)), yc]; St = 1.0 - z["p_test"]
    m = yt == mino
    r = {"edges_from": "VAL" if has_val else "CAL", "n_cal_min": int((yc == mino).sum()), "n_test_min": int(m.sum()),
         "n_test_min_low": int((m & (bt == 0)).sum()), "minority_label": mino}
    for meth in ("marginal", "mondrian", "mondrian_x"):
        T = C.CALIBRATORS[meth](sc, yc, St, ALPHA, bin_cal=bc, bin_test=bt)
        S = C.predict_sets(St, T); hit = S[np.arange(len(yt)), yt]
        r[f"{meth}.cls"] = hit[m].mean()
        r[f"{meth}.low"] = hit[m & (bt == 0)].mean() if (m & (bt == 0)).any() else np.nan
        r[f"{meth}.high"] = hit[m & (bt == 2)].mean() if (m & (bt == 2)).any() else np.nan
        r[f"{meth}.size"] = S.sum(1).mean()
        if meth == "mondrian":
            rr = [hit[m & (rt == k)].mean() for k in range(3) if (m & (rt == k)).any()]
            r["mondrian.rand_spread"] = max(rr) - min(rr) if rr else np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="dataset/regime, e.g. bace/scaffold")
    ap.add_argument("--models", nargs="+", default=["rf"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[100, 119])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows, missing = [], []
    for run in a.runs:
        for md in a.models:
            for seed in range(a.seeds[0], a.seeds[1] + 1):
                p = REPO / "results/raw" / run / f"{md}_ce" / f"seed{seed}.npz"
                if not p.exists():
                    missing.append(str(p.relative_to(REPO))); continue
                rows.append({"run": run, "model": md, "seed": seed, **split_metrics(p)})
    if missing:
        print(f"WARNING: {len(missing)} caches missing, e.g. {missing[:3]}")
    D = pd.DataFrame(rows)
    summ, pairs = [], []
    for (run, md), g in D.groupby(["run", "model"], sort=False):
        r = {"run": run, "model": md, "n_seeds": g.seed.nunique(), "edges_from": g.edges_from.iloc[0],
             "minority_label": int(g.minority_label.iloc[0]), "n_cal_min": g.n_cal_min.mean(),
             "n_test_min": g.n_test_min.mean(), "n_test_min_low": g.n_test_min_low.mean()}
        for c in [c for c in g.columns if "." in c]:
            mu, lo, hi = mean_ci(g[c].values); r[c] = mu; r[c + "_lo"] = lo; r[c + "_hi"] = hi
        summ.append(r)
        pc = paired_compare(g["mondrian_x.low"].values, g["mondrian.low"].values)
        pairs.append({"run": run, "model": md, "contrast": "class x support - class-conditional", "endpoint": "low-support minority coverage", **pc})
    S = pd.DataFrame(summ); P = pd.DataFrame(pairs)
    P["holm_p"] = holm(P.wilcoxon_p.fillna(1).values); P["family_size"] = len(P)
    S.to_csv(REPO / f"tables/{a.tag}_summary.csv", index=False); P.to_csv(REPO / f"tables/{a.tag}_paired.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["run", "model", "n_seeds", "edges_from", "minority_label", "n_cal_min", "n_test_min_low", "marginal.cls", "mondrian.cls",
            "mondrian.low", "mondrian.low_hi", "mondrian.high", "mondrian_x.low", "mondrian.rand_spread", "mondrian.size", "mondrian_x.size"]
    print(S[cols].round(3).to_string(index=False)); print()
    print(P[["run", "model", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
