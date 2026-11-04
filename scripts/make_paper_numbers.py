"""Write every number the main paper uses as a LaTeX macro, with provenance.

Outputs
  paper/latex/generated/numbers.tex          \\expandafter\\def\\csname v@<key>\\endcsname{<value>}
  paper/latex/generated/numbers_provenance.csv   key, value, raw, source, filter, how
  paper/latex/generated/T_main.tex           main results table (cells are also keys)
In the paper:  \\V{key}.  An undefined key prints a red ??key, so a missing number is visible.

Every value here is checked by scripts/verify_paper_numbers.py, which recomputes from the Stage A
caches (results/raw/*.npz) and shares no code with this file or with the Stage B pipeline.
Confirmatory seeds 100-119 only.  BBBP rows always filter external = none (transfer is separate).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T = REPO / "tables"
OUT = REPO / "paper/latex/generated"
EXCH = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]
REC = []


def put(key, value, source, filt, how, fmt="{:.2f}"):
    key = key.replace("→", "to")
    s = value if isinstance(value, str) else fmt.format(value)
    if re.fullmatch(r"-0\.?0*", s):          # a value that rounds to zero prints without a sign
        s = s[1:]
    REC.append({"key": key, "value": s, "raw": value if not isinstance(value, str) else np.nan,
                "source": source, "filter": filt, "how": how})
    return s


def rng_(key, vals, source, filt, how, fmt="{:.2f}"):
    vals = np.asarray(vals, float)
    put(key + ".min", vals.min(), source, filt, "min of " + how, fmt)
    put(key + ".max", vals.max(), source, filt, "max of " + how, fmt)


def main():
    S = pd.read_csv(T / "confirmatory_summary.csv", low_memory=False)
    src = "tables/confirmatory_summary.csv"
    S = S[S.n_seeds == 20]
    lac = (S.score == "lac") & (S.alpha == 0.1) & (S.loss == "ce")
    exch = S.external.isna() & S.regime.isin(["random", "scaffold"])
    F = "score=lac, alpha=0.1, loss=ce, external=none, regime in {random,scaffold}, 12 cells"

    def cells(method, col, extra=None):
        m = lac & exch & (S.method == method)
        if extra is not None:
            m &= extra
        g = S[m].set_index(["setting", "model_name"])[col]
        assert len(g) == 12, (method, col, len(g))
        return g

    rng_("marg.cls", cells("marginal", "cov_minority"), src, F + ", method=marginal", "cov_minority")
    rng_("mond.cls", cells("mondrian", "cov_minority"), src, F + ", method=mondrian", "cov_minority")
    rng_("mond.low", cells("mondrian", "cov_min_bin0"), src, F + ", method=mondrian", "cov_min_bin0")
    rng_("mond.high", cells("mondrian", "cov_min_bin2"), src, F + ", method=mondrian", "cov_min_bin2")
    rng_("mx.low", cells("mondrian_x", "cov_min_bin0"), src, F + ", method=mondrian_x", "cov_min_bin0")
    rng_("mx.cls", cells("mondrian_x", "cov_minority"), src, F + ", method=mondrian_x", "cov_minority")
    rng_("size.cost", cells("mondrian_x", "set_size") - cells("mondrian", "set_size"), src, F,
         "set_size(mondrian_x) - set_size(mondrian)", "{:.3f}")
    rng_("mx.high", cells("mondrian_x", "cov_min_bin2"), src, F + ", method=mondrian_x", "cov_min_bin2")
    put("mx.high.ndrop", int((cells("mondrian_x", "cov_min_bin2") < cells("mondrian", "cov_min_bin2")).sum()), src, F,
        "count cells where high-support coverage falls under class x support", "{:d}")
    put("mx.high.nbelow", int((cells("mondrian_x", "cov_min_bin2") < 0.9).sum()), src, F + ", method=mondrian_x",
        "count cells with high-support point estimate < 0.90", "{:d}")
    put("mx.high.nciexcl", int((cells("mondrian_x", "cov_min_bin2_hi") < 0.9).sum()), src, F + ", method=mondrian_x",
        "count cells with high-support 95% CI upper end < 0.90", "{:d}")
    put("mx.low.nciexcl", int((cells("mondrian_x", "cov_min_bin0_hi") < 0.9).sum()), src, F + ", method=mondrian_x",
        "count cells with low-support 95% CI upper end < 0.90", "{:d}")
    spread = np.array([cells("mondrian", f"cov_min_rand{k}") for k in range(3)])
    rng_("rand.spread", spread.max(0) - spread.min(0), src, F + ", method=mondrian",
         "max-min over cov_min_rand0..2 per cell")
    gap = cells("mondrian", "cov_min_bin2") - cells("mondrian", "cov_min_bin0")
    rng_("mond.gap", gap, src, F + ", method=mondrian", "cov_min_bin2 - cov_min_bin0")
    put("n.lowcells.ciexcl", int((cells("mondrian", "cov_min_bin0_hi") < 0.9).sum()), src,
        F + ", method=mondrian", "count of cells whose cov_min_bin0 95% CI upper end < 0.90", "{:d}")
    # per-cell table values
    rows = []
    for st in EXCH:
        for md in MODELS:
            k = f"{st}.{md}"
            r = {}
            for meth, cols in (("marginal", ["cov_minority"]), ("mondrian", ["cov_minority", "cov_min_bin0", "cov_min_bin2", "set_size"]),
                               ("mondrian_x", ["cov_min_bin0", "set_size"])):
                for c in cols:
                    v = cells(meth, c).loc[(st, md)]
                    r[f"{meth}.{c}"] = put(f"t.{k}.{meth}.{c}", v, src, F + f", setting={st}, model={md}, method={meth}", c)
            rows.append((st, md, r))
    # empirical settings: transfer and cluster
    for st, filt in (("bbbp→b3db/scaffold", S.external == "b3db"), ("bbbp/cluster", S.external.isna()),
                     ("b3db/cluster", S.external.isna())):
        for md in MODELS:
            for meth, c in (("marginal", "cov_minority"), ("mondrian", "cov_minority"), ("mondrian", "cov_min_bin0"),
                            ("mondrian_x", "cov_min_bin0"), ("mondrian", "cov_majority"), ("marginal", "cov_majority"),
                            ("mondrian", "set_size"), ("mondrian_x", "set_size"), ("mondrian", "cov_min_bin2")):
                v = S[lac & filt & (S.setting == st) & (S.model_name == md) & (S.method == meth)][c]
                assert len(v) == 1, (st, md, meth, c)
                put(f"e.{st}.{md}.{meth}.{c}", float(v.iloc[0]), src, f"score=lac, alpha=0.1, loss=ce, setting={st}, model={md}, method={meth}", c)

    # curated dataset sizes (from the pipeline's standardized datasets) and the transfer set
    for ds in ("bbbp", "b3db"):
        df = pd.read_pickle(REPO / f"results/processed/{ds}.pkl")["df"]
        put(f"data.{ds}.n", f"{len(df):,}", f"results/processed/{ds}.pkl", "", "molecules after curation")
        put(f"data.{ds}.minfrac", 100 * (df.y == 0).mean(), f"results/processed/{ds}.pkl", "", "percent BBB- (y=0)", "{:.1f}")
        put(f"data.{ds}.nmin", f"{int((df.y == 0).sum()):,}", f"results/processed/{ds}.pkl", "", "BBB- molecules")
    zt = np.load(REPO / "results/raw/bbbp__to__b3db/scaffold/rf_ce/seed100.npz", allow_pickle=True)
    put("data.transfer.n", f"{len(zt['y_test']):,}", "results/raw/bbbp__to__b3db/scaffold/rf_ce/seed100.npz", "", "external test molecules")
    put("data.transfer.nmin", f"{int((zt['y_test'] == 0).sum()):,}", "same", "", "external BBB- molecules")

    # measured minority share of the low-support tertile, and the Prop. 1 floor
    for st in EXCH:
        g = S[lac & exch & (S.method == "mondrian") & (S.setting == st)]
        w = (g.n_min_bin0 / g.n_test_minority).mean()
        put(f"w.{st}", w, src, f"lac, 0.1, ce, mondrian, setting={st}", "mean over models of n_min_bin0/n_test_minority (seed means)")
        put(f"floor.{st}", 1 - 0.1 / w, src, f"setting={st}", "1 - 0.1 / w")

    # secondary scores, alphas, loss: does the low-support gap and its repair keep direction?
    for tag, m in (("aps", (S.score == "aps") & (S.alpha == 0.1)), ("raps", (S.score == "raps") & (S.alpha == 0.1)),
                   ("a05", (S.score == "lac") & (S.alpha == 0.05)), ("a20", (S.score == "lac") & (S.alpha == 0.2))):
        base = m & (S.loss == "ce") & exch
        a = S[base & (S.method == "mondrian_x")].set_index(["setting", "model_name"]).cov_min_bin0
        b = S[base & (S.method == "mondrian")].set_index(["setting", "model_name"]).cov_min_bin0
        d = (a - b).dropna()
        put(f"sec.{tag}.npos", int((d > 0).sum()), src, f"{tag}, ce, 12 cells", "count mean(mondrian_x bin0) > mean(mondrian bin0)", "{:d}")
        put(f"sec.{tag}.ncell", int(len(d)), src, f"{tag}", "cells", "{:d}")
        rng_(f"sec.{tag}.gain", d, src, f"{tag}, ce", "mondrian_x - mondrian cov_min_bin0")
        rng_(f"sec.{tag}.mondlow", b, src, f"{tag}, ce", "mondrian cov_min_bin0")
    # support-definition robustness: every taxonomy scored on the common VAL tertile grid
    for meth in ("mondrian_x_maccs", "mondrian_x_top5", "mondrian_kmeans3", "mondrian_tpsa", "mondrian_x_b2", "mondrian_x_b5",
                 "mondrian_x_random"):
        v = cells(meth, "cov_min_bin0")
        rng_(f"rob.{meth}", v, src, F + f", method={meth}", "cov_min_bin0 (common tertile grid)")
    # training loss (H1): marginal minority coverage wCE - CE
    for md in MODELS:
        m0 = (S.score == "lac") & (S.alpha == 0.1) & exch & (S.method == "marginal") & (S.model_name == md)
        a = S[m0 & (S.loss == "wce")].set_index("setting").cov_minority
        b = S[m0 & (S.loss == "ce")].set_index("setting").cov_minority
        rng_(f"loss.{md}", a - b, src, f"lac 0.1 marginal {md}", "cov_minority wce - ce")
        m1 = (S.score == "lac") & (S.alpha == 0.1) & exch & (S.model_name == md)
        def prem(l):
            q = S[m1 & (S.loss == l)].pivot_table(index="setting", columns="method", values="set_size")
            return q["mondrian"] - q["marginal"]
        rng_(f"prem.{md}", prem("wce") - prem("ce"), src, f"lac 0.1 {md}", "Mondrian premium wce - ce")

    # primary family
    P = pd.read_csv(T / "confirmatory_paired_primary.csv")
    sp = "tables/confirmatory_paired_primary.csv"
    assert len(P) == 36 and (P.family_size == 36).all()
    mx = P[P.comparison == "mondrian_x - mondrian"]
    put("fam.size", 36, sp, "all rows", "rows", "{:d}")
    rng_("gain", mx.mean_diff, sp, "comparison = mondrian_x - mondrian", "mean_diff", "{:.3f}")
    put("gain.npos", int((mx.mean_diff > 0).sum()), sp, "mondrian_x - mondrian", "count mean_diff > 0", "{:d}")
    put("gain.nsig", int(mx["significant_holm_0.05"].sum()), sp, "mondrian_x - mondrian", "count Holm < 0.05", "{:d}")
    put("gain.bbbpscaf.holm", float(mx[mx.setting == "bbbp/scaffold"].holm_p.max()), sp, "bbbp/scaffold", "max holm_p", "{:.3f}")
    rng_("gain.bbbpscaf", mx[mx.setting == "bbbp/scaffold"].mean_diff, sp, "bbbp/scaffold", "mean_diff", "{:.3f}")
    put("gain.sigmaxholm", float(mx[mx["significant_holm_0.05"]].holm_p.max()), sp, "significant cells", "max holm_p", "{:.3f}")
    rng_("gain.dz", mx.cohens_dz, sp, "mondrian_x - mondrian", "cohens_dz", "{:.2f}")
    sx = P[P.comparison == "sabm - mondrian_x"]
    put("sabmfixed.maxabs", float(sx.mean_diff.abs().max()), sp, "sabm - mondrian_x", "max |mean_diff|", "{:.4f}")
    put("sabmfixed.nsig", int(sx["significant_holm_0.05"].sum()), sp, "sabm - mondrian_x", "count Holm<0.05", "{:d}")

    # cluster bootstrap
    B = pd.read_csv(T / "primary_cluster_bootstrap.csv"); sb = "tables/primary_cluster_bootstrap.csv"
    for u in ("molecule", "scaffold"):
        g = B[(B.unit == u) & (B.comparison == "mondrian_x - mondrian")]
        put(f"boot.{u}.nsig", int(g["significant_holm_0.05"].sum()), sb, f"unit={u}, mondrian_x - mondrian", "count", "{:d}")
        put(f"boot.{u}.maxholm", float(g.holm_p.max()), sb, f"unit={u}", "max holm_p", "{:.4f}")
        rng_(f"boot.{u}.cilo", g.ci_lo, sb, f"unit={u}", "ci_lo", "{:.3f}")
    put("boot.B", int(B.B.iloc[0]), sb, "", "B", "{:d}")
    rng_("boot.nmol", B[B.unit == "molecule"].n_units, sb, "unit=molecule", "unique low-support BBB- molecules", "{:.0f}")

    # common grid (A4)
    G = pd.read_csv(T / "common_grid_summary.csv"); sg = "tables/common_grid_summary.csv"
    G["n_min_cal"] = G.n_min_cal.astype(str)
    G = G[G.n_seeds == 20]          # a subsample size counts for a cell only if reachable in all 20 splits
    a1 = G.alpha == 0.1
    for nm in ("all", "40", "20"):
        for meth in ("fixed_b2", "fixed_b3", "sabm_b3", "mondrian", "fixed_b5"):
            g = G[a1 & (G.n_min_cal == nm) & (G.method == meth)]
            rng_(f"cg.{nm}.{meth}.t0", g.cov_min_t0, sg, f"alpha=0.1, n_min_cal={nm}, method={meth}", "cov_min_t0 (seed mean)")
            rng_(f"cg.{nm}.{meth}.size", g.set_size, sg, f"alpha=0.1, n_min_cal={nm}, method={meth}", "set_size")
            rng_(f"cg.{nm}.{meth}.single", g.singleton_rate, sg, f"alpha=0.1, n_min_cal={nm}, method={meth}", "singleton_rate")
    gall = G[a1 & (G.n_min_cal == "all")].pivot_table(index=["setting", "model"], columns="method", values="cov_min_t0")
    put("cg.all.sabm_eq_b3", int(np.isclose(gall.sabm_b3, gall.fixed_b3, atol=1e-12).sum()), sg, "alpha=0.1, all",
        "count of cells with sabm_b3 == fixed_b3 on cov_min_t0", "{:d}")
    put("cg.ncells.all", int(len(gall)), sg, "alpha=0.1, all", "cells", "{:d}")
    gsz = G[a1 & (G.n_min_cal == "all")].pivot_table(index=["setting", "model"], columns="method", values="set_size")
    dsz = (gsz.fixed_b3 - gsz.sabm_b3)
    put("cg.all.sabm3.size_eq", int(np.isclose(dsz, 0, atol=1e-12).sum()), sg, "alpha=0.1, all",
        "cells where sabm_b3 and fixed_b3 have identical mean set size", "{:d}")
    put("cg.all.sabm3.size_maxdiff", float(dsz.max()), sg, "alpha=0.1, all", "largest set-size reduction of sabm_b3 vs fixed_b3", "{:.3f}")
    ginf = G[a1 & (G.n_min_cal == "all") & (G.method == "fixed_b3")]
    put("cg.all.fixed_b3.inf.max", float(ginf.inf_threshold_frac.max()), sg, "alpha=0.1, all, fixed_b3",
        "largest fraction of infinite-threshold cells", "{:.3f}")
    rng_("cg.ncalmin", G[a1 & (G.n_min_cal == "all") & (G.method == "mondrian")].n_cal_minority, sg, "all", "minority calibration size (seed mean)", "{:.0f}")
    # feasibility cliff: fixed-partition degeneracy, mean over the setting x model cells available at each size
    feas = {}
    for nm in ("10", "20", "40", "80", "160", "all"):
        for nb in (2, 3, 5, 8):
            g = G[a1 & (G.n_min_cal == nm) & (G.method == f"fixed_b{nb}")]
            feas[(nm, nb)] = put(f"fz.{nm}.b{nb}", g.degenerate_cell_frac.mean(), sg,
                                 f"alpha=0.1, n_min_cal={nm}, method=fixed_b{nb}",
                                 "mean over available cells of degenerate_cell_frac", "{:.3f}")
        put(f"fz.{nm}.ncells", int(G[a1 & (G.n_min_cal == nm) & (G.method == "fixed_b2")].shape[0]), sg,
            f"n_min_cal={nm}", "cells available", "{:d}")
    for nm in ("40", "all", "20"):
        for meth in ("fixed_b2", "sabm_b8", "fixed_b8", "mondrian"):
            g = G[a1 & (G.n_min_cal == nm) & (G.method == meth)]
            for c in ("singleton_rate", "set_size", "cov_min_t0"):
                put(f"cgm.{nm}.{meth}.{c}", g[c].mean(), sg, f"alpha=0.1, n_min_cal={nm}, method={meth}",
                    f"mean over cells of {c}", "{:.3f}")
    with open(OUT / "T_feas.tex", "w") as f:
        f.write("% GENERATED by scripts/make_paper_numbers.py\n\\begin{tabular}{rcccc}\n\\toprule\n"
                "\\bbbm\\ calibration size & 2 bins & 3 bins & 5 bins & 8 bins\\\\\n\\midrule\n")
        for nm in ("10", "20", "40", "80", "160"):
            f.write(f"{nm} & " + " & ".join(f"{feas[(nm, nb)]} ({int(nm) / nb:.1f})" for nb in (2, 3, 5, 8)) + "\\\\\n")
        f.write("realized (" + "\\V{cg.ncalmin.min}--\\V{cg.ncalmin.max}" + ") & " + " & ".join(feas[("all", nb)] for nb in (2, 3, 5, 8))
                + "\\\\\n\\bottomrule\n\\end{tabular}\n")

    # persistent failures (A1, A2)
    C = pd.read_csv(T / "persistent_failure_concordance_ce.csv"); sc = "tables/persistent_failure_concordance_ce.csv"
    C = C[C.miss_definition == "mondrian"].set_index("setting")
    for st in C.index:
        k = st.replace("→", "to")
        for col in ("n_cross_arch", "n_eligible", "n_majority_all3"):
            put(f"pf.{k}.{col}", int(C.loc[st, col]), sc, f"setting={st}, miss=mondrian", col, "{:d}")
        put(f"pf.{k}.null", float(C.loc[st, "null_mean"]), sc, f"setting={st}", "null_mean", "{:.2f}")
        put(f"pf.{k}.p", float(C.loc[st, "perm_p_one_sided"]), sc, f"setting={st}", "perm_p_one_sided", "{:.3f}")
        for b in MODELS:
            put(f"pf.{k}.{b}", int(C.loc[st, f"n_persistent_{b}"]), sc, f"setting={st}", f"n_persistent_{b}", "{:d}")
    CX = pd.read_csv(T / "persistent_failure_concordance_ce_excl3.csv"); CX = CX[CX.miss_definition == "mondrian"].set_index("setting")
    for st in CX.index:
        k = st.replace("→", "to")
        put(f"pfx.{k}.n_cross_arch", int(CX.loc[st, "n_cross_arch"]), "tables/persistent_failure_concordance_ce_excl3.csv",
            f"setting={st}, three curation-error records excluded", "n_cross_arch", "{:d}")
        put(f"pfx.{k}.null", float(CX.loc[st, "null_mean"]), "tables/persistent_failure_concordance_ce_excl3.csv", f"setting={st}", "null_mean", "{:.2f}")
    Dm = pd.read_csv(T / "persistent_failure_descriptors_majority_ce.csv"); sd = "tables/persistent_failure_descriptors_majority_ce.csv"
    Dp = pd.read_csv(T / "persistent_failure_descriptors_ce.csv"); sdp = "tables/persistent_failure_descriptors_ce.csv"
    for nm, D, s_ in (("maj", Dm, sd), ("str", Dp, sdp)):
        for _, r in D[D.descriptor.isin(["TPSA", "HBD", "MW", "cLogP", "mean_sim_train"])].iterrows():
            k = f"pd.{nm}.{r.setting.replace('→', 'to')}.{r.descriptor}"
            put(k + ".med", r.median_persistent, s_, f"setting={r.setting}, descriptor={r.descriptor}", "median_persistent", "{:.1f}" if r.descriptor in ("TPSA", "MW") else "{:.2f}")
            put(k + ".rest", r.median_rest, s_, f"setting={r.setting}, descriptor={r.descriptor}", "median_rest", "{:.1f}" if r.descriptor in ("TPSA", "MW") else "{:.2f}")
            if pd.notna(r.get("holm_p_primary_family", np.nan)):
                put(k + ".holm", r.holm_p_primary_family, s_, f"setting={r.setting}, descriptor={r.descriptor}", "holm_p_primary_family", "{:.3f}")
        tp = D[(D.descriptor == "TPSA") & D.holm_p_primary_family.notna()]
        put(f"pd.{nm}.tpsa.maxholm", float(tp.holm_p_primary_family.max()), s_, "TPSA, tested settings", "max holm", "{:.3f}")
        put(f"pd.{nm}.tpsa.ntested", int(len(tp)), s_, "TPSA", "settings tested (>=5 per group)", "{:d}")
        put(f"pd.{nm}.ntests", int(D.holm_p_primary_family.notna().sum()), s_, "primary four", "Holm family size", "{:d}")
    E = pd.read_csv(T / "persistent_failure_evidence_group_ce.csv"); se = "tables/persistent_failure_evidence_group_ce.csv"
    for _, r in E[E.n_persistent > 0].iterrows():
        k = r.setting.replace("→", "to")
        put(f"ev.{k}.p", r.fisher_p_two_sided, se, f"setting={r.setting}", "fisher_p_two_sided", "{:.2f}")
        put(f"ev.{k}.fp", r.frac_persistent_no_logBB, se, f"setting={r.setting}", "frac_persistent_no_logBB", "{:.2f}")
        put(f"ev.{k}.fr", r.frac_rest_no_logBB, se, f"setting={r.setting}", "frac_rest_no_logBB", "{:.2f}")

    # descriptor adjustment (A8)
    n_sup, n_full, cells_ = 0, 0, 0
    for md in MODELS:
        A = pd.read_csv(T / f"descriptor_adjustment_coefficients_confirmatory_{md}.csv")
        tp = A[A.term == "TPSA"]
        for st in EXCH:
            cells_ += 1
            n_sup += bool(tp[(tp.setting == st) & (tp.model_spec == "adjusted for support")].excludes_zero.iloc[0])
            n_full += bool(tp[(tp.setting == st) & (tp.model_spec == "adjusted for support + size/lipo")].excludes_zero.iloc[0])
        rng_(f"adj.{md}.unadj", tp[tp.model_spec == "unadjusted"].coef, f"tables/descriptor_adjustment_coefficients_confirmatory_{md}.csv", "term=TPSA, unadjusted", "coef")
        rng_(f"adj.{md}.sup", tp[tp.model_spec == "adjusted for support"].coef, f"tables/descriptor_adjustment_coefficients_confirmatory_{md}.csv", "term=TPSA, adjusted for support", "coef")
        pe = pd.read_csv(T / f"descriptor_adjustment_permutation_confirmatory_{md}.csv")
        put(f"adj.{md}.perm.nsig", int(pe.signal_beyond_support.sum()), f"tables/descriptor_adjustment_permutation_confirmatory_{md}.csv", "", "count signal_beyond_support", "{:d}")
    put("adj.nsup", n_sup, "tables/descriptor_adjustment_coefficients_confirmatory_*.csv", "TPSA, adjusted for support", "count CI excludes 0", "{:d}")
    put("adj.nfull", n_full, "tables/descriptor_adjustment_coefficients_confirmatory_*.csv", "TPSA, + size/lipo", "count CI excludes 0", "{:d}")
    put("adj.ncells", cells_, "", "", "backbone x setting cells", "{:d}")

    # screening utility (A3)
    U = pd.read_csv(T / "screening_utility_paired.csv"); su = "tables/screening_utility_paired.csv"
    u1 = (U.alpha == 0.1)
    for con in ("fixed_b3 - ad_reject", "fixed_b3 - conf_reject", "fixed_b3 - mondrian", "mondrian - marginal",
                "fixed_b3 - ad_reject_val", "fixed_b3 - conf_reject_val"):
        for ep in ("minority_false_accept_rate", "accepted_error", "deferral_rate", "low_support_accepted_error"):
            g = U[u1 & (U.contrast == con) & (U.endpoint == ep)]
            tag = con.replace("fixed_b3 - ", "").replace(" - ", "_")
            ex = g[g.setting != "bbbp→b3db/scaffold"]; tr = g[g.setting == "bbbp→b3db/scaffold"]
            rng_(f"su.{tag}.{ep}.exch", ex.mean_diff, su, f"alpha=0.1, {con}, {ep}, exchangeable", "mean_diff")
            rng_(f"su.{tag}.{ep}.tr", tr.mean_diff, su, f"alpha=0.1, {con}, {ep}, transfer", "mean_diff")
            put(f"su.{tag}.{ep}.nneg", int(((g.ci_hi < 0)).sum()), su, f"alpha=0.1, {con}, {ep}", "count ci_hi < 0 (of 15)", "{:d}")
    SU = pd.read_csv(T / "screening_utility_summary.csv")
    g = SU[(SU.alpha == 0.1) & (SU.method == "fixed_b3")]
    rng_("su.defer.exch", g[g.setting != "bbbp→b3db/scaffold"].deferral_rate, "tables/screening_utility_summary.csv", "fixed_b3, exchangeable", "deferral_rate")
    for meth in ("fixed_b3", "ad_reject_val", "conf_reject_val"):
        g = SU[(SU.alpha == 0.1) & (SU.method == meth) & (SU.setting == "bbbp→b3db/scaffold")]
        rng_(f"su.defer.tr.{meth}", g.deferral_rate, "tables/screening_utility_summary.csv", f"{meth}, transfer", "deferral_rate")

    # ------------------------------------------------------------------------------------------------
    # EXPLORATORY analyses added 31 Oct 2026 (reviewer defenses). Not part of the 36-member family.
    # ------------------------------------------------------------------------------------------------
    # D3: existing conditional / localized conformal methods on the same caches (Stage B rows)
    CB = pd.read_csv(T / "conditional_baselines.csv"); scb = "tables/conditional_baselines.csv"
    assert (CB.n_cells == 12).all(), CB[CB.n_cells != 12]
    for _, r in CB.iterrows():
        f_ = f"lac, alpha=0.1, ce, 12 exchangeable cells, method={r.method}"
        put(f"cb.{r.method}.low.min", r.low_min, scb, f_, "min over cells of seed-mean cov_min_bin0")
        put(f"cb.{r.method}.low.max", r.low_max, scb, f_, "max over cells of seed-mean cov_min_bin0")
        put(f"cb.{r.method}.high", r.high_mean, scb, f_, "mean over cells of seed-mean cov_min_bin2")
        put(f"cb.{r.method}.nfail", int(r.n_cells_low_ci_below_090), scb, f_, "cells whose cov_min_bin0 95% t-CI upper end < 0.90", "{:d}")
        put(f"cb.{r.method}.size", r.set_size_mean, scb, f_, "mean over cells of set_size")
    # same-cell-size control: class x Tanimoto vs class x random strata (exploratory family of 12)
    RS = pd.read_csv(T / "random_strata_contrast.csv"); srs = "tables/random_strata_contrast.csv"
    assert len(RS) == 12 and (RS.n_pairs == 20).all()
    f_ = "lac, alpha=0.1, ce, 12 cells, mondrian_x - mondrian_x_random, low-support BBB- coverage"
    rng_("rs.gain", RS.mean_diff, srs, f_, "mean_diff", "{:.3f}")
    put("rs.npos", int((RS.mean_diff > 0).sum()), srs, f_, "cells with mean_diff > 0", "{:d}")
    put("rs.nsig", int((RS.holm_p < 0.05).sum()), srs, f_, "cells with Holm p < 0.05 (family of 12)", "{:d}")
    put("rs.fam", int(RS.family_size.iloc[0]), srs, f_, "Holm family size", "{:d}")
    put("rs.nonsig.maxholm", float(RS[RS.holm_p >= 0.05].holm_p.max()), srs, f_, "largest Holm p among non-significant cells", "{:.3f}")
    put("rs.nonsig.n", int((RS.holm_p >= 0.05).sum()), srs, f_, "non-significant cells", "{:d}")
    rng_("rs.randlow", RS["mondrian_x_random.cov_min_bin0"], srs, "random strata", "low-support BBB- coverage")
    rng_("rs.cellsize.low", RS["mondrian_x_random.cov_min_bin0"] - RS["mondrian.cov_min_bin0"], srs, "random strata - Mondrian",
         "low-support BBB- coverage", "{:.3f}")
    rng_("rs.cellsize.high", RS["mondrian_x_random.cov_min_bin2"] - RS["mondrian.cov_min_bin2"], srs, "random strata - Mondrian",
         "high-support BBB- coverage", "{:.3f}")
    put("rs.randhigh.nup", int((RS["mondrian_x_random.cov_min_bin2"] > RS["mondrian.cov_min_bin2"]).sum()), srs,
        "random strata vs Mondrian", "cells where high-support coverage rises", "{:d}")
    rng_("rs.randsize", RS["mondrian_x_random.set_size"], srs, "random strata", "mean set size")
    for ds in ("bbbp", "b3db"):
        g = RS[RS.setting.str.startswith(ds + "/")]
        rng_(f"rs.cellsize.low.{ds}", g["mondrian_x_random.cov_min_bin0"] - g["mondrian.cov_min_bin0"], srs,
             f"{ds}, random strata - Mondrian", "low-support BBB- coverage", "{:.3f}")
        # finite-sample inflation scale 1/(n_r + 1) for three random strata of a minority calibration set of size n
        ncal = S[lac & exch & (S.method == "mondrian") & S.setting.str.startswith(ds + "/")].groupby("setting").n_cal_minority.mean()
        rng_(f"th.csb.{ds}", 1 / (ncal / 3 + 1), "tables/confirmatory_summary.csv", f"{ds}, method=mondrian",
             "1/(n_cal_minority/3 + 1), n = seed-mean minority calibration size per setting", "{:.3f}")
    rng_("rs.mxsize", RS["mondrian_x.set_size"], srs, "class x Tanimoto", "mean set size")
    # model competence (context): held-out discrimination of each backbone
    PM = pd.read_csv(T / "classifier_metrics.csv"); spm = "tables/classifier_metrics.csv"
    ind = PM[~PM.setting.str.contains("→")]; trn = PM[PM.setting.str.contains("→")]
    assert len(ind) == 12 and (PM.n_seeds == 20).all()
    for c, nm in (("auroc_mean", "auroc"), ("auprc_minority_mean", "auprc"), ("balanced_accuracy_mean", "balacc"),
                  ("minority_recall_mean", "minrecall"), ("minority_prevalence_mean", "prev")):
        rng_(f"perf.{nm}", ind[c], spm, "12 in-domain cells, seed means", c)
        rng_(f"perf.tr.{nm}", trn[c], spm, "transfer, 3 backbones, seed means", c)
    # stronger screening baseline: Mondrian plus a rejection filter at the deferral of class x support
    CR = pd.read_csv(T / "calibrated_rejection_paired.csv"); scr = "tables/calibrated_rejection_paired.csv"
    for pol in ("mond_ad", "mond_conf", "mond_ad_val", "mond_conf_val"):
        for ep in ("minority_false_accept_rate", "accepted_error", "deferral_rate"):
            g = CR[(CR.contrast == f"fixed_b3 - {pol}") & (CR.endpoint == ep)]
            ex, tr = g[~g.setting.str.contains("→")], g[g.setting.str.contains("→")]
            f_ = f"fixed_b3 - {pol}, {ep}"
            rng_(f"cr.{pol}.{ep}.exch", ex.mean_diff, scr, f_ + ", 12 in-domain cells", "mean_diff", "{:.3f}")
            rng_(f"cr.{pol}.{ep}.tr", tr.mean_diff, scr, f_ + ", transfer", "mean_diff", "{:.3f}")
            put(f"cr.{pol}.{ep}.nlow", int((g.ci_hi < 0).sum()), scr, f_ + ", 15 cells", "cells with seed-bootstrap CI below 0", "{:d}")
            put(f"cr.{pol}.{ep}.nhigh", int((g.ci_lo > 0).sum()), scr, f_ + ", 15 cells", "cells with seed-bootstrap CI above 0", "{:d}")
    CRs = pd.read_csv(T / "calibrated_rejection_summary.csv")
    put("cr.unmatched.frac", float((CRs[CRs.policy == "fixed_b3"].extra_deferrals_needed < 0).mean()), "tables/calibrated_rejection_summary.csv",
        "fixed_b3 rows (setting x model)", "fraction of cells whose mean extra deferral is negative", "{:.2f}")
    # concordance against a covariate-stratified null, and appearance counts
    CC = pd.read_csv(T / "concordance_conditional.csv").set_index("setting"); scc = "tables/concordance_conditional.csv"
    for st in CC.index:
        k = st.replace("→", "to")
        put(f"cc.{k}.nullcov", float(CC.loc[st, "null_cov_mean"]), scc, f"setting={st}", "null mean, strata k x support x TPSA", "{:.2f}")
        put(f"cc.{k}.pcov", float(CC.loc[st, "p_cov"]), scc, f"setting={st}", "permutation p, covariate strata", "{:.4f}")
        for c in ("n_kge3", "n_kge4", "n_k2"):
            put(f"cc.{k}.{c}", int(CC.loc[st, c]), scc, f"setting={st}", c, "{:d}")
        if pd.notna(CC.loc[st, "k_median"]):
            put(f"cc.{k}.kmed", float(CC.loc[st, "k_median"]), scc, f"setting={st}", "median appearances of strict failures", "{:.1f}")
            put(f"cc.{k}.kmax", int(CC.loc[st, "k_max"]), scc, f"setting={st}", "max appearances of strict failures", "{:d}")
    # B3DB-internal name conflicts
    NC = pd.read_csv(T / "b3db_name_conflicts.csv"); snc = "tables/b3db_name_conflicts.csv"
    for _, r in NC.iterrows():
        k = f"nc.{r.setting}.{r.definition}"
        put(k + ".ratefail", 100 * r.rate_fail, snc, f"setting={r.setting}, {r.definition}", "percent of failures with a same-name BBB+ record", "{:.0f}")
        put(k + ".raterest", 100 * r.rate_rest, snc, f"setting={r.setting}, {r.definition}", "percent of other eligible BBB- molecules", "{:.0f}")
        put(k + ".p", r.fisher_p, snc, f"setting={r.setting}, {r.definition}", "Fisher p", "{:.2f}")
        put(k + ".nfail", int(r.fail_conflict), snc, f"setting={r.setting}, {r.definition}", "failures with a same-name BBB+ record", "{:d}")
    put("nc.nconflict.min", int(NC.n_conflict.min()), snc, "b3db settings", "eligible BBB- molecules with a same-name BBB+ record", "{:d}")
    put("nc.nconflict.max", int(NC.n_conflict.max()), snc, "b3db settings", "eligible BBB- molecules with a same-name BBB+ record", "{:d}")
    # D4: post-hoc recalibration invariance (Prop. S3)
    I = pd.read_csv(T / "posthoc_invariance_check.csv"); si = "tables/posthoc_invariance_check.csv"
    dcols = ["max_abs_diff_low", "max_abs_diff_set_size", "max_abs_diff_cov_minority"]
    cl = I[I.base != "marginal"]
    put("inv.ts.max", float(cl[cl.posthoc.str.endswith("_ts")][dcols].values.max()), si, "class-conditional rows, temperature scaling",
        "max |difference| over cells and endpoints", "{:.4f}")
    put("inv.la.max", float(cl[cl.posthoc.str.endswith("_la")][dcols].values.max()), si, "class-conditional rows, logit adjustment",
        "max |difference| over cells and endpoints", "{:.4f}")
    put("inv.la.lowmax", float(cl[cl.posthoc.str.endswith("_la")].max_abs_diff_low.max()), si, "logit adjustment",
        "max |difference| in low-support BBB- coverage", "{:.4f}")
    for ph in ("marginal_ts", "marginal_la"):
        mg = I[I.posthoc == ph]
        assert len(mg) == 1, ph
        put(f"inv.{ph}.max", float(mg[dcols].values.max()), si, f"marginal vs {ph}", "max |difference| over cells and endpoints", "{:.4f}")
        put(f"inv.{ph}.cov", float(mg.max_abs_diff_cov_minority.iloc[0]), si, f"marginal vs {ph}", "max |difference| in cov_minority", "{:.2f}")
        put(f"inv.{ph}.low", float(mg.max_abs_diff_low.iloc[0]), si, f"marginal vs {ph}", "max |difference| in low-support BBB- coverage", "{:.2f}")
    # D5: continuous support curve
    SL = pd.read_csv(T / "support_curve_slopes.csv"); ssl = "tables/support_curve_slopes.csv"
    for meth, tg in (("mondrian", "mond"), ("mondrian_x", "mx")):
        g = SL[SL.method == meth]
        rng_(f"sc.{tg}.slope", g["slope_per_0.1"], ssl, f"method={meth}", "logistic slope of coverage per +0.1 Tanimoto")
        put(f"sc.{tg}.npos", int((g.ci_lo > 0).sum()), ssl, f"method={meth}", "cells with split-bootstrap CI above 0", "{:d}")
        put(f"sc.{tg}.nneg", int((g.ci_hi < 0).sum()), ssl, f"method={meth}", "cells with split-bootstrap CI below 0", "{:d}")
        put(f"sc.{tg}.nflat", int((~g.ci_excludes_0.astype(bool)).sum()), ssl, f"method={meth}", "cells whose CI includes 0", "{:d}")
    # D7: support-dependent calibration at a fixed score
    SC = pd.read_csv(T / "support_calibration_coefs.csv"); ssc = "tables/support_calibration_coefs.csv"
    for ds in ("bbbp", "b3db"):
        g = SC[SC.setting.str.startswith(ds + "/")]
        rng_(f"scal.{ds}.coef", g["coef_support_per_0.1"], ssc, f"{ds}, 6 cells", "support coefficient per +0.1 Tanimoto")
        put(f"scal.{ds}.nneg", int((g.support_ci_hi < 0).sum()), ssc, f"{ds}", "cells with CI below 0", "{:d}")
        put(f"scal.{ds}.npos", int((g.support_ci_lo > 0).sum()), ssc, f"{ds}", "cells with CI above 0", "{:d}")
    # D8: mechanism (asymmetric degradation of the true-class probability)
    M = pd.read_csv(T / "mechanism_summary.csv"); sm = "tables/mechanism_summary.csv"
    assert len(M) == 12
    rng_("mech.p2.min", M.P2_drop_p_true_min_high_to_low, sm, "12 cells", "BBB- true-class probability, high minus low tertile")
    rng_("mech.p2.maj", M.P2_drop_p_true_maj_high_to_low, sm, "12 cells", "BBB+ true-class probability, high minus low tertile")
    put("mech.p2.n", int((M.P2_drop_p_true_min_high_to_low > M.P2_drop_p_true_maj_high_to_low).sum()), sm, "12 cells",
        "cells where the BBB- drop exceeds the BBB+ drop", "{:d}")
    rng_("mech.covmajlow", M.cov_maj_low, sm, "12 cells, class-conditional", "low-support BBB+ coverage")
    rng_("mech.covminlow", M.cov_min_low, sm, "12 cells, class-conditional", "low-support BBB- coverage")
    put("mech.p1.nmajok", int(M.P1_maj_low_at_or_above_090.astype(bool).sum()), sm, "12 cells",
        "cells whose low-support BBB+ coverage t-CI lower end >= 0.88", "{:d}")
    put("mech.p3.n", int(M.P3_low_gap_negative.astype(bool).sum()), sm, "12 cells", "cells with low-tertile calibration gap CI below 0", "{:d}")
    put("mech.p3.nb3db", int(M[M.run.str.startswith("b3db")].P3_low_gap_negative.astype(bool).sum()), sm, "6 B3DB cells",
        "cells with low-tertile calibration gap CI below 0", "{:d}")
    short = lambda run: run.replace("tox21_sr_are", "tox21")
    MG = pd.read_csv(T / "mechanism_generality_summary.csv"); smg = "tables/mechanism_generality_summary.csv"
    for _, r in MG.iterrows():
        put(f"mechg.{short(r.run)}.p2min", r.P2_drop_p_true_min_high_to_low, smg, f"run={r.run}, rf", "minority true-class drop, high - low")
        put(f"mechg.{short(r.run)}.p2maj", r.P2_drop_p_true_maj_high_to_low, smg, f"run={r.run}, rf", "majority true-class drop, high - low")
    # D2 / R2: other imbalanced tasks, and the prior-work protocol on confirmatory seeds
    for tg, fam in (("gen", "generality"), ("tc", "tursun_confirm")):
        GS = pd.read_csv(T / f"{fam}_summary.csv"); GP = pd.read_csv(T / f"{fam}_paired.csv")
        ss_, sp_ = f"tables/{fam}_summary.csv", f"tables/{fam}_paired.csv"
        key = (lambda r: f"tc.{r.model}") if tg == "tc" else (lambda r: f"gen.{short(r.run)}")
        assert (GS.n_seeds == 20).all()
        for _, r in GS.iterrows():
            for c in ("marginal.cls", "mondrian.cls", "mondrian.low", "mondrian.high", "mondrian_x.low", "mondrian_x.high",
                      "mondrian.size", "mondrian_x.size"):
                put(f"{key(r)}.{c}", r[c], ss_, f"run={r.run}, model={r.model}", f"seed mean of {c}")
            put(f"{key(r)}.ncalmin", r.n_cal_min, ss_, f"run={r.run}, model={r.model}", "minority calibration size (seed mean)", "{:.1f}")
            put(f"{key(r)}.ntestminlow", r.n_test_min_low, ss_, f"run={r.run}, model={r.model}", "low-support minority test molecules (seed mean)", "{:.1f}")
        for _, r in GP.iterrows():
            f_ = f"run={r.run}, model={r.model}, class x support - class-conditional, low-support minority coverage"
            put(f"{key(r)}.gain", r.mean_diff, sp_, f_, "mean_diff", "{:.3f}")
            put(f"{key(r)}.cilo", r.ci_lo, sp_, f_, "ci_lo (seed bootstrap)", "{:.3f}")
            put(f"{key(r)}.cihi", r.ci_hi, sp_, f_, "ci_hi (seed bootstrap)", "{:.3f}")
            put(f"{key(r)}.holm", r.holm_p, sp_, f_, f"holm_p within this family of {int(r.family_size)}", "{:.4f}")
        put(f"{tg}.fam", int(len(GP)), sp_, "", "Holm family size", "{:d}")
    # D1: can CAL alone flag the failure? (leave-one-out coverage on CAL)
    CA = pd.read_csv(T / "cal_audit_flags.csv").iloc[0]; sca = "tables/cal_audit_flags.csv"
    put("ca.rho", CA.spearman_split, sca, "240 splits", "Spearman, CAL LOO vs TEST low-support coverage")
    put("ca.sens", CA.sensitivity, sca, "flag < 0.85", "sensitivity")
    put("ca.spec", CA.specificity, sca, "flag < 0.85", "specificity")
    put("ca.cellagree", int(CA.cell_agreement), sca, "12 cells", "cells where CAL and TEST CI flags agree", "{:d}")
    put("ca.signagree", CA.split_gap_sign_agree, sca, "240 splits", "fraction of splits where CAL and TEST high-low gap signs agree")
    put("ca.nsplits", int(CA.n_splits), sca, "", "splits with both CAL and TEST low-support values", "{:d}")
    # A2 (dataset level): BBBP / B3DB label audit
    LO = pd.read_csv(T / "label_audit_overlap.csv").iloc[0]; slo = "tables/label_audit_overlap.csv"
    put("la.nshared", f"{int(LO.n_shared_connectivity):,}", slo, "", "BBBP molecules whose connectivity key occurs in B3DB")
    put("la.overlap", 100 * LO.frac_bbbp_in_b3db, slo, "", "percent of BBBP in B3DB", "{:.1f}")
    put("la.agree", 100 * LO.label_agreement, slo, "", "percent label agreement on shared molecules", "{:.1f}")
    put("la.ndis", int(LO.n_disagree), slo, "", "label disagreements", "{:d}")
    put("la.negpos", int(LO.n_bbbp_neg_b3db_pos), slo, "", "BBBP BBB- but B3DB BBB+", "{:d}")
    put("la.posneg", int(LO.n_bbbp_pos_b3db_neg), slo, "", "BBBP BBB+ but B3DB BBB-", "{:d}")
    LF = pd.read_csv(T / "label_audit_failures.csv"); slf = "tables/label_audit_failures.csv"
    for _, r in LF.iterrows():
        k = f"la.{r.setting}.{r.definition}"
        put(k + ".p", r.fisher_p, slf, f"setting={r.setting}, {r.definition}", "Fisher exact p (two-sided)", "{:.3f}")
        put(k + ".fdis", int(r.failures_disagree), slf, f"setting={r.setting}, {r.definition}", "failures among label conflicts", "{:d}")
        put(k + ".ndis", int(r.n_disagree), slf, f"setting={r.setting}, {r.definition}", "eligible molecules with a label conflict", "{:d}")
        if np.isfinite(r.fisher_odds_ratio) and r.fisher_odds_ratio > 0:
            put(k + ".or", r.fisher_odds_ratio, slf, f"setting={r.setting}, {r.definition}", "Fisher odds ratio", "{:.1f}")

    # Figure 1 (scripts/make_fig1_overview.py): representative setting, threshold-relative scores, traced molecule
    f1p = REPO / "tables/fig1_values.csv"
    if f1p.exists():
        F1 = dict(pd.read_csv(f1p, dtype=str).values)
        src1 = "tables/fig1_values.csv"
        flt1 = f"{F1['sel.setting']}, {F1['sel.model']}, LAC, alpha=0.1, CE, seeds 100-119"
        for k in ("sel.gap", "sel.all12.gap", "A.cls", "A.low", "A.mid", "A.high", "A.gap", "C.low.mond", "C.low.cxs",
                  "C.high.mond", "C.high.cxs", "C.size.mond", "C.size.cxs", "B.shift.low.min", "B.shift.low.max",
                  "B.shift.high.min", "B.shift.high.max", "mol.score", "mol.qclass", "mol.qstratum"):
            put(f"fig1.{k}", float(F1[k]), src1, flt1, k, "{:.3f}")
        for k in ("A.cls.lo", "A.cls.hi", "A.low.lo", "A.low.hi", "A.high.lo", "A.high.hi", "C.low.cxs.lo", "C.low.cxs.hi",
                  "C.high.cxs.lo", "C.high.cxs.hi"):
            put(f"fig1.{k}", float(F1[k]), src1, flt1, "95% CI over splits: " + k, "{:.3f}")
        for k in ("low", "mid", "high"):
            put(f"fig1.B.excl.{k}", 100 * (1 - float(F1[f"B.F0.{k}"])), src1, flt1,
                f"% of pooled {k}-support BBB- test appearances above their split's class threshold", "{:.1f}")
            put(f"fig1.B.n.{k}", int(F1[f"B.n.{k}"]), src1, flt1, f"{k}-support BBB- test appearances", "{:d}")
        for k in ("C.full.mond", "C.full.cxs"):
            put(f"fig1.{k}", 100 * float(F1[k]), src1, flt1, "% two-label sets, all test compounds", "{:.1f}")
        for k in ("B.shift.low.npos", "B.shift.high.nneg", "B.rescued", "B.lost", "B.rescued.low", "B.rescued.high",
                  "B.lost.low", "B.lost.high", "mol.n", "mol.rescued", "mol.seed"):
            put(f"fig1.{k}", int(F1[k]), src1, flt1, k, "{:d}")
        for k in ("B.ncal.low.min", "B.ncal.low.max", "B.nmol.low", "B.nmol.mid", "B.nmol.high", "B.rescued.mid", "B.lost.mid"):
            put(f"fig1.{k}", int(F1[k]), src1, flt1, k, "{:d}")
        put("fig1.sel.gap12.low", float(F1["sel.gap12.low"]), src1, "all 12 cells", "smallest gap", "{:.3f}")
        put("fig1.sel.gap12.high", float(F1["sel.gap12.high"]), src1, "all 12 cells", "largest gap", "{:.3f}")
        put("fig1.mol.id", F1["mol.id"], src1, flt1, "traced molecule identifier")
        put("fig1.mol.nn_id", F1["mol.nn_id"], src1, flt1, "nearest TRAIN compound of the traced molecule")
        for k in ("B.exclint.low", "B.exclint.high", "mol.no", "mol.cid", "nn.no", "nn.cid", "mol.nref", "mol.ncal", "mol.in_bbbp"):
            put(f"fig1.{k}", int(F1[k]), src1, flt1, k, "{:d}")
        for k in ("A.excl.low", "A.excl.mid", "A.excl.high"):
            put(f"fig1.{k}", float(F1[k]), src1, flt1, "100 x (1 - split-mean coverage)", "{:.1f}")
        put("fig1.mol.excess", float(F1["mol.excess"]), src1, flt1, "score minus class threshold, traced split", "{:+.3f}")
        put("fig1.nn.sim2", float(F1["nn.sim2"]), src1, flt1, "second-highest Tanimoto to TRAIN", "{:.3f}")
        put("fig1.mol.smax.min", float(F1["mol.smax.min"]), src1, flt1, "over test appearances", "{:.2f}")
        put("fig1.mol.smax.max", float(F1["mol.smax.max"]), src1, flt1, "over test appearances", "{:.2f}")
        put("fig1.mol.smax", float(F1["mol.smax"]), src1, flt1, "max Tanimoto of the traced molecule to TRAIN", "{:.2f}")
        put("fig1.mol.name", F1["mol.name"], src1, flt1, "traced molecule (" + F1["mol.id"] + ")")
        put("fig1.sel.label", F1["sel.label"], src1, flt1, "selected setting")
        put("fig1.all12.label", F1["sel.all12.label"], src1, "all 12 cells", "lower median of the 12 gaps")

    # write
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(REC)
    assert R.key.is_unique, R.key[R.key.duplicated()].tolist()
    R.to_csv(OUT / "numbers_provenance.csv", index=False)
    with open(OUT / "numbers.tex", "w") as f:
        f.write("% GENERATED by scripts/make_paper_numbers.py. Do not edit. Provenance: numbers_provenance.csv\n")
        for _, r in R.iterrows():
            f.write(f"\\expandafter\\def\\csname v@{r.key}\\endcsname{{{r.value}}}\n")
    # main table
    lab = {"bbbp/random": "BBBP random", "bbbp/scaffold": "BBBP scaffold", "b3db/random": "B3DB random", "b3db/scaffold": "B3DB scaffold"}
    nm = {"rf": "RF", "gin": "GIN", "chemprop": "Chemprop"}
    with open(OUT / "T_main.tex", "w") as f:
        f.write("% GENERATED by scripts/make_paper_numbers.py\n")
        f.write("\\begin{tabular}{llccccccc}\n\\toprule\n& & Marginal & \\multicolumn{3}{c}{Class-only Mondrian} & Class$\\times$support & \\multicolumn{2}{c}{Set size}\\\\\n"
                "\\cmidrule(lr){4-6}\\cmidrule(lr){8-9}\nSetting & Backbone & class & class & low & high & low & Mondrian & C$\\times$S\\\\\n\\midrule\n")
        for i, (st, md, r) in enumerate(rows):
            f.write(f"{lab[st] if i % 3 == 0 else ''} & {nm[md]} & {r['marginal.cov_minority']} & {r['mondrian.cov_minority']} & "
                    f"{r['mondrian.cov_min_bin0']} & {r['mondrian.cov_min_bin2']} & {r['mondrian_x.cov_min_bin0']} & "
                    f"{r['mondrian.set_size']} & {r['mondrian_x.set_size']}\\\\" + ("\\midrule\n" if i % 3 == 2 and i < 11 else "\n"))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {len(R)} keys to {OUT / 'numbers.tex'}")


if __name__ == "__main__":
    main()
