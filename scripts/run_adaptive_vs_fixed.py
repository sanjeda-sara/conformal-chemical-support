"""THE decisive experiment: does count-adaptive merging (SABM) do anything that a FIXED
class x support Mondrian partition does not?  (Reviewer objection A / ranked fix #2.)

Background.  At alpha = 0.10 the frozen budget gives n_req = 11, and on BBBP/B3DB every
class x tertile cell already holds >= 11 calibration molecules.  SABM therefore never merges
and is bit-identical to fixed `mondrian_x`.  Presenting SABM as the headline method on that
evidence alone is indefensible.  The honest claim is narrower and testable:

    SABM is a ROBUSTNESS mechanism, not a coverage mechanism.  A fixed partition that is
    finer than the calibration data can support degenerates: sparse cells get a +inf
    threshold and emit FULL sets, so the predictor is valid but useless.  SABM merges those
    cells and keeps the predictor informative, at the same guarantee.

This script measures exactly that, across the regime where the two can differ:
    * bins        b in {2, 3, 5, 8}          (partition fineness)
    * alpha       in {0.05, 0.10, 0.20}      (n_req = 22 / 11 / 5)
    * n_min       minority calibration size, subsampled: all, 160, 80, 40, 20, 10
Everything is Stage-B only (cached predictions), so it is cheap.

Reported per configuration, for `mondrian_x` (fixed) and `sabm` (adaptive):
    merge_rate        fraction of (class, bin) lookups whose group pools >1 bin
    mean_bins_pooled  average number of bins in the group actually used
    degenerate_rate   fraction of (class, bin) cells with an infinite threshold
    full_rate         fraction of TEST molecules receiving the full set {0,1}  (= deferral)
    singleton_rate    fraction receiving an actionable singleton  (= acceptance)
    cov_min_bin0      low-support minority coverage
    cov_minority      class-level minority coverage
    + a MATCHED-DEFERRAL comparison (reviewer objection D): for each method we also report
      singleton_error (error rate among accepted singletons) overall, for the minority class,
      and for low-support molecules, so coverage gains cannot be bought with deferral.

usage:
  python scripts/run_adaptive_vs_fixed.py                       # all cached CE RF/GIN runs
  python scripts/run_adaptive_vs_fixed.py --runs "results/raw/bbbp/random/rf_ce/seed*.npz"
  python scripts/run_adaptive_vs_fixed.py --n-min 10 20 40 --draws 20
Outputs: tables/adaptive_vs_fixed.csv (per run) and tables/adaptive_vs_fixed_summary.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.conformal.calibrators import _partition_bins, conformal_quantile, n_required
from bbbcp.conformal.scores import score_matrix
from bbbcp.evaluate import sim_bins
from bbbcp.metrics.matched import compare_at_matched, operating_curve
from bbbcp.stats.stats import mean_ci

REPO = Path(__file__).resolve().parents[1]


def partition_diagnostics(y_cal, bin_cal, bin_test, alpha, nb, n_req=None):
    """How often does the adaptive rule actually merge, and how often would the FIXED
    partition degenerate?  Mirrors calibrators.sabm exactly (counts include the test point)."""
    n_req = n_required(alpha) if n_req is None else n_req
    k_min = int(np.ceil(1 / alpha)) - 1          # fewest cal points for a finite threshold
    merged, pooled, cells, degen_fixed, degen_adapt = 0, [], 0, 0, 0
    for c in (0, 1):
        mc = y_cal == c
        base = np.bincount(bin_cal[mc], minlength=nb)
        for b in range(nb):
            if not (bin_test == b).any():
                continue
            cells += 1
            cnt = base.copy(); cnt[b] += 1
            gid = _partition_bins(cnt, n_req)
            size = int((gid == gid[b]).sum())
            pooled.append(size); merged += size > 1
            degen_fixed += base[b] < k_min
            degen_adapt += base[np.where(gid == gid[b])[0]].sum() < k_min
    return {"merge_rate": merged / max(cells, 1), "mean_bins_pooled": float(np.mean(pooled)) if pooled else np.nan,
            "degenerate_rate_fixed": degen_fixed / max(cells, 1), "degenerate_rate_adaptive": degen_adapt / max(cells, 1),
            "n_cells": cells, "n_req": n_req}


def metrics(sets, y_test, bin_test, minority=0):
    n = len(y_test)
    hit = sets[np.arange(n), y_test]
    size = sets.sum(1); single = size == 1; full = size == 2
    m = y_test == minority; low = bin_test == 0
    out = {"cov_minority": hit[m].mean() if m.any() else np.nan,
           "cov_min_bin0": hit[m & low].mean() if (m & low).any() else np.nan,
           "cov_majority": hit[~m].mean() if (~m).any() else np.nan,
           "set_size": size.mean(), "singleton_rate": single.mean(), "full_rate": full.mean(),
           "empty_rate": (size == 0).mean()}
    # matched-deferral quantities: among ACCEPTED (singleton) predictions, how often wrong?
    out["singleton_error"] = (~hit[single]).mean() if single.any() else np.nan
    out["singleton_error_minority"] = (~hit[single & m]).mean() if (single & m).any() else np.nan
    out["singleton_error_low_support"] = (~hit[single & low]).mean() if (single & low).any() else np.nan
    out["acceptance_minority"] = single[m].mean() if m.any() else np.nan
    out["acceptance_low_support"] = single[low].mean() if low.any() else np.nan
    return out


MATCH_GRID = np.round(np.concatenate([np.arange(0.01, 0.31, 0.02), [0.35, 0.40]]), 3)


def run_file(path, n_bins=(2, 3, 5, 8), alphas=(0.05, 0.10, 0.20), n_min_list=("all", 160, 80, 40, 20, 10),
             draws=10, minority=0, kind="lac", seed=0, matched=False):
    z = np.load(path, allow_pickle=False)
    if "minority" in z.files:
        minority = int(z["minority"])
    y_cal, y_test = z["y_cal"].astype(int), z["y_test"].astype(int)
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    rng = np.random.default_rng(seed)
    Sc_all = score_matrix(z["p_cal"], kind); St = score_matrix(z["p_test"], kind)
    s_cal_all = Sc_all[np.arange(len(y_cal)), y_cal]
    min_idx = np.where(y_cal == minority)[0]; maj_idx = np.where(y_cal != minority)[0]
    tag = str(Path(path)).split("/raw/")[1]
    rows, matched_rows = [], []
    for nb in n_bins:
        bc_all, _ = sim_bins(ref, z["sim_cal"], nb); bt, _ = sim_bins(ref, z["sim_test"], nb)
        bt3, _ = sim_bins(ref, z["sim_test"], 3)          # low-support bin always from tertiles
        for n_min in n_min_list:
            reps = 1 if n_min == "all" else draws
            if n_min != "all" and n_min > len(min_idx):
                continue
            for rep in range(reps):
                if n_min == "all":
                    sub = np.arange(len(y_cal))
                else:
                    sub = np.concatenate([maj_idx, rng.choice(min_idx, n_min, replace=False)])
                s_cal, yc, bc = s_cal_all[sub], y_cal[sub], bc_all[sub]
                for a in alphas:
                    diag = partition_diagnostics(yc, bc, bt, a, nb)
                    for name, fn in (("mondrian_x", C.mondrian_x), ("sabm", C.sabm), ("mondrian", C.mondrian)):
                        T = fn(s_cal, yc, St, a, bin_cal=bc, bin_test=bt)
                        r = {"run": tag, "seed": int(str(path).split("seed")[-1].split(".")[0]),
                             "n_bins": nb, "n_min_cal": n_min, "rep": rep, "alpha": a, "method": name,
                             **metrics(C.predict_sets(St, T), y_test, bt3, minority)}
                        r.update({k: v for k, v in diag.items()} if name == "sabm" else
                                 {"n_req": diag["n_req"], "degenerate_rate_fixed": diag["degenerate_rate_fixed"],
                                  "n_cells": diag["n_cells"]})
                        rows.append(r)
                if matched:
                    # Reviewer objection D: a coverage gain bought by deferring more is not a
                    # gain.  Sweep alpha for each method and interpolate all of them to the
                    # operating point label-only Mondrian reaches at alpha = 0.10, matching
                    # first on deferral and then on minority coverage.
                    curves = {}
                    for name, fn in (("mondrian", C.mondrian), ("mondrian_x", C.mondrian_x), ("sabm", C.sabm)):
                        curves[name] = operating_curve(
                            lambda aa, fn=fn: C.predict_sets(St, fn(s_cal, yc, St, aa, bin_cal=bc, bin_test=bt)),
                            MATCH_GRID, y_test, bt3, minority)
                    for how in ("full_rate", "cov_minority"):
                        cmp_ = compare_at_matched(curves, "mondrian", 0.10, by=how)
                        cmp_ = cmp_.assign(run=tag, seed=int(str(path).split("seed")[-1].split(".")[0]),
                                           n_bins=nb, n_min_cal=n_min, rep=rep, matched_on=how)
                        matched_rows.append(cmp_)
    return (rows, matched_rows) if matched else rows


def summarize(df):
    keys = ["setting", "model", "n_bins", "n_min_cal", "alpha", "method"]
    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("seed", "rep") and c not in keys]
    g = df.groupby(keys, dropna=False)[num].mean().reset_index()
    g["n_runs"] = df.groupby(keys, dropna=False).size().values
    return g


def headline(summ):
    """The table that answers the reviewer: where SABM and fixed differ, who wins and why."""
    rows = []
    for (st, mdl, nb, nmin, a), g in summ.groupby(["setting", "model", "n_bins", "n_min_cal", "alpha"]):
        f = g[g.method == "mondrian_x"]; s = g[g.method == "sabm"]; m = g[g.method == "mondrian"]
        if f.empty or s.empty:
            continue
        f, s, m = f.iloc[0], s.iloc[0], (m.iloc[0] if not m.empty else None)
        rows.append({"setting": st, "model": mdl, "n_bins": nb, "n_min_cal": nmin, "alpha": a,
                     "n_req": s.get("n_req"), "merge_rate": s.get("merge_rate"),
                     "mean_bins_pooled": s.get("mean_bins_pooled"),
                     "degenerate_rate_fixed": f.get("degenerate_rate_fixed"),
                     "identical": bool(np.isclose(f.set_size, s.set_size, atol=1e-9) and
                                       np.isclose(f.cov_min_bin0, s.cov_min_bin0, atol=1e-9, equal_nan=True)),
                     "cov_min_bin0_fixed": f.cov_min_bin0, "cov_min_bin0_sabm": s.cov_min_bin0,
                     "cov_min_bin0_mondrian": m.cov_min_bin0 if m is not None else np.nan,
                     "singleton_fixed": f.singleton_rate, "singleton_sabm": s.singleton_rate,
                     "singleton_gain_sabm": s.singleton_rate - f.singleton_rate,
                     "full_fixed": f.full_rate, "full_sabm": s.full_rate,
                     "singleton_err_min_fixed": f.singleton_error_minority,
                     "singleton_err_min_sabm": s.singleton_error_minority,
                     "set_size_fixed": f.set_size, "set_size_sabm": s.set_size})
    return pd.DataFrame(rows)


def _expand(pattern):
    """Expand shell-style brace alternatives, e.g. a/{x,y}/b -> [a/x/b, a/y/b] (glob does not)."""
    i = pattern.find("{")
    if i < 0:
        return [pattern]
    j = pattern.index("}", i)
    return [p for alt in pattern[i + 1:j].split(",")
            for p in _expand(pattern[:i] + alt + pattern[j + 1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/raw/{bbbp,b3db}/{random,scaffold}/{rf,gin,chemprop}_ce/seed1??.npz",
                    help="glob; shell-style {a,b} alternatives are expanded")
    ap.add_argument("--n-bins", nargs="+", type=int, default=[2, 3, 5, 8])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--n-min", nargs="+", default=["all", 160, 80, 40, 20, 10])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--matched", action="store_true",
                    help="also sweep alpha and interpolate every method to label-only Mondrian's "
                         "operating point (matched deferral and matched minority coverage); "
                         "writes tables/adaptive_vs_fixed_matched.csv.  Slower.")
    a = ap.parse_args()
    files = sorted({f for p in _expand(a.runs) for f in glob.glob(str(REPO / p))})
    if a.max_runs:
        files = files[: a.max_runs]
    n_min = [x if x == "all" else int(x) for x in a.n_min]
    rows, mrows = [], []
    for i, f in enumerate(files):
        out = run_file(f, tuple(a.n_bins), tuple(a.alphas), tuple(n_min), a.draws, matched=a.matched)
        if a.matched:
            r, m = out; rows += r; mrows += m
        else:
            rows += out
        print(f"[{i+1}/{len(files)}] {f.split('/raw/')[1]}", flush=True)
    df = pd.DataFrame(rows)
    df["setting"] = df.run.str.split("/").str[:2].str.join("/")
    df["model"] = df.run.str.split("/").str[2]
    df.to_csv(REPO / "tables/adaptive_vs_fixed.csv", index=False)
    summ = summarize(df); summ.to_csv(REPO / "tables/adaptive_vs_fixed_summary.csv", index=False)
    head = headline(summ); head.to_csv(REPO / "tables/adaptive_vs_fixed_headline.csv", index=False)
    print("\n=== Where does adaptive merging ACTIVATE, and what does it buy? ===")
    act = head[head.merge_rate > 0].sort_values("singleton_gain_sabm", ascending=False)
    cols = ["setting", "model", "n_bins", "n_min_cal", "alpha", "n_req", "merge_rate", "degenerate_rate_fixed",
            "cov_min_bin0_fixed", "cov_min_bin0_sabm", "singleton_fixed", "singleton_sabm", "singleton_gain_sabm"]
    print(act[cols].head(25).round(3).to_string(index=False))
    print(f"\nconfigurations where SABM == fixed: {int(head.identical.sum())}/{len(head)}")
    if mrows:
        md = pd.concat(mrows, ignore_index=True)
        md["setting"] = md.run.str.split("/").str[:2].str.join("/")
        md.to_csv(REPO / "tables/adaptive_vs_fixed_matched.csv", index=False)
        ok = md[md.achievable & (md.matched_on == "full_rate")]
        print("\n=== AT MATCHED DEFERRAL: low-support minority coverage ===")
        print(ok.pivot_table(index=["n_bins", "method"], columns="n_min_cal",
                             values="cov_min_bin0").round(3).to_string())
        print("\n    reachability of label-only Mondrian's deferral rate (0 = the method cannot")
        print("    reach that operating point at ANY alpha, i.e. it is degenerate there)")
        print(md[md.matched_on == "full_rate"].pivot_table(index=["n_bins", "method"],
              columns="n_min_cal", values="achievable").round(2).to_string())
        print("\nwrote tables/adaptive_vs_fixed_matched.csv")
    print("wrote tables/adaptive_vs_fixed{,_summary,_headline}.csv")


if __name__ == "__main__":
    main()
