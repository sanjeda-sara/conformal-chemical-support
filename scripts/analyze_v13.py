"""Analyse any v1.3 Stage B output (confirmatory or extension grids).

usage: python scripts/analyze_v13.py results/processed/full_v13_conformal.csv [--tag confirmatory]
       [--selective results/processed/full_selective.csv] [--reference sabm]

Writes to tables/:
  <tag>_summary.csv     mean, sd, 95% t-interval and n_seeds of every key metric for every
                        dataset x regime x model x loss x score x method x alpha
  <tag>_paired_primary.csv  PRIMARY pre-specified family: endpoint = low-support minority
                        coverage, LAC, alpha = 0.10, CE, exchangeable settings, 3 contrasts
                        (class x support vs Mondrian; SABM vs Mondrian; SABM vs class x support).
                        Holm applied ONCE across the whole family.  This is the paper's
                        inferential claim.
  <tag>_paired.csv      EXPLORATORY family: SABM vs every other method on six endpoints,
                        Holm within each (endpoint, setting, model, loss) cell.  Descriptive
                        only; with ~48 methods per cell it cannot support a claim
  <tag>_headline.csv    the paper's Table 1 layout (LAC, alpha 0.1, CE)
  <tag>_gates.csv       Gate 3 / Gate 4 / no-go trigger evaluated as pre-registered
  <tag>_selective_summary.csv   non-conformal baselines and conformal selection
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.stats.stats import holm, mean_ci, paired_compare

REPO = Path(__file__).resolve().parents[1]
KEYS = ["dataset", "external", "regime", "model_name", "loss", "score", "method", "alpha"]
METRICS = ["coverage", "cov_minority", "cov_majority", "cov_min_bin0", "cov_min_bin1", "cov_min_bin2",
           "set_size", "singleton_rate", "full_rate", "empty_rate", "mcg", "cii", "acceptance_rate",
           "acc_on_accepted", "minority_acc_on_accepted", "worst_slab_cov", "worst_slab_cov_min",
           "ssc_min", "cov_min_tpsa0", "cov_min_tpsa2", "cov_min_rand0", "cov_min_rand1", "cov_min_rand2",
           "cost_fp1_fn2_d0.3", "keep_frac", "n_min_bin0", "n_cal_minority", "n_test_minority"]


def setting(df):
    ext = df["external"].fillna("").astype(str)
    return df["dataset"].astype(str) + np.where(ext != "", "→" + ext, "") + "/" + df["regime"].astype(str)


def summary(df):
    rows = []
    for k, g in df.groupby(KEYS, dropna=False):
        r = dict(zip(KEYS, k)); r["n_seeds"] = g.seed.nunique()
        for m in METRICS:
            if m in g:
                mu, lo, hi = mean_ci(g[m].values)
                r[m] = mu; r[m + "_lo"] = lo; r[m + "_hi"] = hi; r[m + "_sd"] = np.nanstd(g[m].values, ddof=1) if len(g) > 1 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows); out.insert(0, "setting", setting(out))
    return out


PRIMARY_CONTRASTS = [("mondrian_x", "mondrian"), ("sabm", "mondrian"), ("sabm", "mondrian_x")]
PRIMARY_ENDPOINT = "cov_min_bin0"
PRIMARY_SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
PRIMARY_MODELS = ["rf", "gin", "chemprop"]


PRIMARY_DOC = """Pre-specified PRIMARY inference family (frozen 2026-10-21, before the
confirmatory Stage B was analysed).  Everything outside it is EXPLORATORY and is reported
without multiplicity correction and labelled as such.

  endpoint   E3 = low-support minority coverage (cov_min_bin0)
  contrast   sabm - mondrian   (the paper's claim: support conditioning repairs the failure)
  score      LAC        alpha  0.10        loss  CE
  regimes    random and scaffold only (exchangeable / group-exchangeable); cluster and
             transfer are empirical-only and never enter the primary family
  cells      dataset x regime x backbone  ->  2 x 2 x 3 = 12 comparisons at most
  correction Holm across those 12 (and only those)

Secondary family (same correction, reported separately): E2 = set size, same cells.
Rationale for the size: with 20 paired seeds the smallest attainable Wilcoxon p is ~1e-6, so a
12-member family leaves ample room; the v1.2/v1.3 exploratory table of 600+ rows did not."""

# The three contrasts that decide the paper's claims.  Keeping the null contrast in the SAME
# family is deliberate: the reviewer's objection is that SABM may be indistinguishable from a
# fixed partition, so that comparison must be corrected alongside the ones we hope to reject.
PRIMARY_CONTRASTS = [("mondrian_x", "mondrian"),   # does support conditioning repair the failure?
                     ("sabm", "mondrian"),         # does the proposed calibrator repair it?
                     ("sabm", "mondrian_x")]       # does ADAPTATION add anything? (expected null)
PRIMARY = dict(endpoint="cov_min_bin0", score="lac", alpha=0.10, loss="ce",
               regimes=("random", "scaffold"), contrasts=PRIMARY_CONTRASTS)


def paired_primary(df, spec=PRIMARY):
    """The confirmatory test.  One small family: PRIMARY_CONTRASTS x setting x backbone on the
    pre-specified endpoint, Holm-corrected ONCE across the whole family.  Exploratory
    contrasts live in `paired()` and are never used for a claim in the paper."""
    d = df.copy()
    if "external" not in d:
        d["external"] = np.nan
    d = d[(d.score == spec["score"]) & np.isclose(d.alpha, spec["alpha"]) & (d.loss == spec["loss"])
          & d.regime.isin(spec["regimes"]) & d.external.isna()]
    d = d.assign(setting=setting(d))
    ep = spec["endpoint"]
    rows = []
    for (st, mdl), g in d.groupby(["setting", "model_name"]):
        for a_name, b_name in spec["contrasts"]:
            A = g[g.method == a_name].set_index("seed")
            B = g[g.method == b_name].set_index("seed")
            common = A.index.intersection(B.index)
            if len(common) < 3 or ep not in g:
                continue
            r = paired_compare(A.loc[common, ep].values, B.loc[common, ep].values)
            r.update({"setting": st, "model": mdl, "endpoint": ep,
                      "comparison": f"{a_name} - {b_name}", "family": "primary"})
            rows.append(r)
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["holm_p"] = holm(out.wilcoxon_p.fillna(1).values)
    out["family_size"] = len(out)
    out["significant_holm_0.05"] = out.holm_p < 0.05
    return out


def primary_family(df, spec=PRIMARY):
    """paired_primary plus a one-row verdict on the paper's two claims."""
    out = paired_primary(df, spec)
    if not len(out):
        return out, pd.DataFrame()
    rep = out[out.comparison == "mondrian_x - mondrian"]
    ada = out[out.comparison == "sabm - mondrian_x"]
    verdict = pd.DataFrame([{
        "family_size": int(out.family_size.iloc[0]),
        "repair_cells": int(len(rep)),
        "repair_cells_holm_sig": int(rep["significant_holm_0.05"].sum()),
        "repair_median_effect": float(rep.mean_diff.median()) if len(rep) else np.nan,
        "adaptation_cells": int(len(ada)),
        "adaptation_cells_holm_sig": int(ada["significant_holm_0.05"].sum()),
        "adaptation_median_effect": float(ada.mean_diff.median()) if len(ada) else np.nan,
        "CLAIM_support_conditioning_repairs": bool(len(rep) and rep["significant_holm_0.05"].sum() >= max(2, len(rep) // 2)),
        "CLAIM_adaptation_adds_coverage": bool(len(ada) and ada["significant_holm_0.05"].any()),
    }])
    return out, verdict


def paired(df, reference="sabm", score="lac", alpha=0.10,
           endpoints=("cov_min_bin0", "cov_minority", "set_size", "worst_slab_cov_min", "full_rate")):
    """All pairwise contrasts vs `reference`.  Holm is applied WITHIN each
    (endpoint, setting, model, loss) family, which is an EXPLORATORY correction: the family is
    large and the adjusted p-values are correspondingly conservative.  The confirmatory
    inference is `primary_family()` below; use that for any claim in the paper."""
    d = df[(df.score == score) & np.isclose(df.alpha, alpha)].copy()
    d["setting"] = setting(d)
    rows = []
    for (st, mdl, loss), g in d.groupby(["setting", "model_name", "loss"]):
        ref = g[g.method == reference].set_index("seed")
        if ref.empty:
            continue
        for other in sorted(set(g.method) - {reference}):
            o = g[g.method == other].set_index("seed")
            common = ref.index.intersection(o.index)
            if len(common) < 3:
                continue
            for ep in endpoints:
                if ep not in g:
                    continue
                r = paired_compare(ref.loc[common, ep].values, o.loc[common, ep].values)
                r.update({"setting": st, "model": mdl, "loss": loss, "endpoint": ep,
                          "comparison": f"{reference} - {other}", "other": other,
                          "family": "exploratory"})
                rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["holm_p"] = np.nan
        for _, g in out.groupby(["endpoint", "setting", "model", "loss"]):
            out.loc[g.index, "holm_p"] = holm(g.wilcoxon_p.fillna(1).values)
    return out


def headline(summ):
    s = summ[(summ.score == "lac") & np.isclose(summ.alpha, 0.10) & (summ.loss == "ce")]
    keep = ["marginal", "mondrian", "mondrian_pac", "mondrian_x", "sabm", "rc3p", "clustered_cells",
            "condqr_sim", "kandinsky_qr", "rlcp_sim", "lcp_knn", "interp_q90", "macro_cov", "kmm_mondrian",
            "weighted_mondrian"]
    s = s[s.method.isin(keep)]
    cols = ["setting", "model_name", "method", "n_seeds", "cov_minority", "cov_minority_lo", "cov_minority_hi",
            "cov_min_bin0", "cov_min_bin0_lo", "cov_min_bin0_hi", "cov_min_bin2", "set_size", "worst_slab_cov_min"]
    return s[cols].sort_values(["setting", "model_name", "method"])


def gates(summ, target=0.90):
    """Pre-registered Gate 3 / no-go: Mondrian low-support minority coverage CI upper end
    below target (failure demonstrated) per dataset x model, exchangeable regimes, LAC, CE."""
    s = summ[(summ.score == "lac") & np.isclose(summ.alpha, 0.10) & (summ.loss == "ce") &
             (summ.method == "mondrian") & summ.regime.isin(["random", "scaffold"]) & summ.external.isna()]
    s = s.assign(failure_shown=s.cov_min_bin0_hi < target, ci_includes_target=(s.cov_min_bin0_lo <= target) & (s.cov_min_bin0_hi >= target))
    nogo = all(s[(s.model_name.isin(["rf", "gin"]))].ci_includes_target) if len(s) else None
    s["NO_GO_TRIGGER(all RF/GIN CIs include 0.90)"] = nogo
    return s[["setting", "model_name", "n_seeds", "cov_min_bin0", "cov_min_bin0_lo", "cov_min_bin0_hi",
              "failure_shown", "ci_includes_target", "NO_GO_TRIGGER(all RF/GIN CIs include 0.90)"]]


def selective_summary(sel):
    sel = sel.copy(); sel["setting"] = setting(sel)
    num = [c for c in sel.columns if sel[c].dtype.kind in "fc" and c not in ("alpha", "q", "tau")]
    keys = ["setting", "model_name", "loss", "family", "rule", "calibrator"]
    return sel.groupby(keys, dropna=False)[num].mean().reset_index()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("conformal"); ap.add_argument("--selective", default=None)
    ap.add_argument("--tag", default=None); ap.add_argument("--reference", default="sabm")
    a = ap.parse_args()
    tag = a.tag or Path(a.conformal).stem.replace("_conformal", "")
    df = pd.read_csv(a.conformal)
    if "external" not in df:
        df["external"] = np.nan
    summ = summary(df); summ.to_csv(REPO / f"tables/{tag}_summary.csv", index=False)
    pr = paired(df, a.reference); pr.to_csv(REPO / f"tables/{tag}_paired.csv", index=False)
    prim, verdict = primary_family(df)
    prim.to_csv(REPO / f"tables/{tag}_paired_primary.csv", index=False)
    with open(REPO / f"tables/{tag}_primary_spec.txt", "w") as fh:
        fh.write(PRIMARY_DOC + "\n")
    if len(verdict):
        verdict.to_csv(REPO / f"tables/{tag}_primary_verdict.csv", index=False)
        print("\n== PRE-SPECIFIED PRIMARY FAMILY (Holm across these cells only) ==")
        print(prim[["setting", "model", "comparison", "n_pairs", "mean_diff", "ci_lo", "ci_hi",
                    "wilcoxon_p", "holm_p", "significant_holm_0.05"]].round(4).to_string(index=False))
        print(verdict.T.to_string(header=False))
    pp = paired_primary(df); pp.to_csv(REPO / f"tables/{tag}_paired_primary.csv", index=False)
    if len(pp):
        print("\n=== PRIMARY family (pre-specified, Holm across the whole family) ===")
        print(pp[["setting", "model", "comparison", "n_pairs", "mean_diff", "ci_lo", "ci_hi",
                  "wilcoxon_p", "holm_p", "significant_holm_0.05"]].round(4).to_string(index=False))
    headline(summ).to_csv(REPO / f"tables/{tag}_headline.csv", index=False)
    g = gates(summ); g.to_csv(REPO / f"tables/{tag}_gates.csv", index=False)
    print(g.round(3).to_string())
    sel_path = a.selective or a.conformal.replace("_conformal.csv", "_selective.csv")
    if Path(sel_path).exists():
        selective_summary(pd.read_csv(sel_path)).to_csv(REPO / f"tables/{tag}_selective_summary.csv", index=False)
    print("wrote tables/%s_{summary,paired,headline,gates}.csv" % tag)


if __name__ == "__main__":
    main()
