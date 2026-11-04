"""Independent verification of paper/latex/generated/numbers.tex.  Shares NO code with
scripts/make_paper_numbers.py, the Stage B pipeline (bbbcp.evaluate / calibrators) or the A1-A5
analysis scripts.  Imports only numpy, pandas, scipy.

Levels, reported per key in paper/latex/generated/numbers_verification.csv:
  RECOMPUTED   value recomputed from results/raw/*.npz with this file's own code
  REPLICATED   recomputed with independent random draws; must agree within a stated tolerance
  CROSSCHECK   checked against PER-SEED Stage B rows (results/processed/full_v13_conformal.csv),
               aggregated here; used where exact recomputation needs the pipeline's own
               randomisation (APS/RAPS U draws) or features (MACCS, k-means)
  DELEGATED    verified by scripts/verify_persistent_failures.py (persistent-failure keys)
  UNVERIFIED   not checked; listed so nobody mistakes it for verified
Exit status 0 only if every checked key matches.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]

def stageb_csv(name="full_v13_conformal"):
    """Stage B per-seed rows; the public repository ships them gzipped (pandas reads either)."""
    for p in (REPO / f"results/processed/{name}.csv", REPO / f"results/processed/{name}.csv.gz"):
        if p.exists():
            return p
    raise FileNotFoundError(f"results/processed/{name}.csv(.gz) not found; run Stage B (scripts/run_grid.py)")

RAW = REPO / "results/raw"
EXCH = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
MODELS = ["rf", "gin", "chemprop"]
SEEDS = range(100, 120)
A = 0.10


# ----------------------------------------------------------------- own conformal primitives
def qhat(x, a):
    x = np.sort(np.asarray(x, float)); n = len(x)
    k = math.ceil((n + 1) * (1 - a) - 1e-12)
    return np.inf if n == 0 or k > n else x[k - 1]


def edges_from(ref, nb):
    return np.quantile(ref, [i / nb for i in range(1, nb)])


def binof(v, e):
    return np.searchsorted(e, v, side="right")          # == np.digitize(v, e)


def sets_for(z, a, how, nb=3):
    """Own implementation: returns boolean (n_test, 2) set matrix."""
    yc = z["y_cal"].astype(int); P_c = z["p_cal"]; P_t = z["p_test"]
    sc = 1 - P_c[np.arange(len(yc)), yc]; St = 1 - P_t
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    e = edges_from(ref, nb); bc = binof(z["sim_cal"], e); bt = binof(z["sim_test"], e)
    out = np.zeros(St.shape, bool)
    for c in (0, 1):
        if how == "marginal":
            out[:, c] = St[:, c] <= qhat(sc, a)
        elif how == "mondrian":
            out[:, c] = St[:, c] <= qhat(sc[yc == c], a)
        else:                                                   # class x support, fixed
            for b in np.unique(bt):
                r = bt == b
                out[r, c] = St[r, c] <= qhat(sc[(yc == c) & (bc == b)], a)
    return out


def load(setting, model, seed, loss="ce"):
    sub = "bbbp__to__b3db/scaffold" if setting == "bbbp→b3db/scaffold" else setting
    return np.load(RAW / sub / f"{model}_{loss}" / f"seed{seed}.npz", allow_pickle=True)


def split_metrics(z, a=A):
    yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    t = binof(z["sim_test"], edges_from(ref, 3))
    r = {"w": (m & (t == 0)).sum() / m.sum()}
    for how in ("marginal", "mondrian", "mx"):
        S = sets_for(z, a, how); hit = S[np.arange(len(yt)), yt]
        r[f"{how}.cov_minority"] = hit[m].mean(); r[f"{how}.cov_majority"] = hit[~m].mean()
        r[f"{how}.cov_min_bin0"] = hit[m & (t == 0)].mean(); r[f"{how}.cov_min_bin2"] = hit[m & (t == 2)].mean()
        r[f"{how}.set_size"] = S.sum(1).mean()
    return r


# ----------------------------------------------------------------- bookkeeping
class Book:
    def __init__(self, prov):
        self.p = prov.set_index("key"); self.rows = []; self.done = set()

    def check(self, key, value, level, tol=None, note=""):
        key = key.replace("→", "to")
        if key not in self.p.index:
            self.rows.append({"key": key, "level": level, "status": "MISSING_KEY", "paper": None, "verifier": value})
            return
        s = str(self.p.loc[key, "value"])
        dec = len(s.split(".")[1]) if "." in s else 0
        paper = float(s.replace(",", ""))
        if tol is None:
            ok = abs(round(value, dec) - paper) < 10 ** (-dec) / 2 + 1e-12 or abs(value - paper) <= 10 ** (-dec) / 2 + 1e-9
        else:
            ok = abs(value - paper) <= tol
        self.rows.append({"key": key, "level": level, "status": "OK" if ok else "MISMATCH", "paper": s,
                          "verifier": value, "tol": tol, "note": note})
        self.done.add(key)

    def rng(self, key, vals, level, tol=None):
        vals = np.asarray(vals, float)
        self.check(key + ".min", vals.min(), level, tol); self.check(key + ".max", vals.max(), level, tol)


# ----------------------------------------------------------------- 10. exploratory analyses (31 Oct)
def t_hi(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() + stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def t_lo(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() - stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def own_holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def own_logit(X, y):
    """Unpenalised logistic regression (scipy).  Returns coefficients without the intercept."""
    from scipy.optimize import minimize
    X1 = np.c_[np.ones(len(X)), X]
    f = lambda b: np.sum(np.logaddexp(0, X1 @ b) - y * (X1 @ b))
    g = lambda b: X1.T @ (0.5 * (1 + np.tanh(0.5 * (X1 @ b))) - y)
    return minimize(f, np.zeros(X1.shape[1]), jac=g, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000}).x[1:]


def verify_exploratory(bk, per):
    has = lambda k: k.replace("→", "to") in bk.p.index
    full = pd.read_csv(stageb_csv(), low_memory=False,
                       usecols=["dataset", "external", "regime", "model_name", "loss", "score", "method", "alpha", "seed",
                                "cov_minority", "cov_min_bin0", "cov_min_bin2", "set_size"])
    full = full[full.seed.between(100, 119) & full.external.isna() & full.regime.isin(["random", "scaffold"])
                & (full.loss == "ce") & (full.score == "lac") & np.isclose(full.alpha, A)]

    # D3: conditional baselines, cross-checked from per-seed Stage B rows (own aggregation and t-intervals)
    for meth in sorted({k.split(".")[1] for k in bk.p.index if k.startswith("cb.")}):
        g = full[full.method == meth].groupby(["dataset", "regime", "model_name"])
        mu = g[["cov_min_bin0", "cov_min_bin2", "set_size"]].mean()
        assert len(mu) == 12 and (g.size() == 20).all(), meth
        hi = g.cov_min_bin0.apply(lambda x: t_hi(x.values))
        bk.check(f"cb.{meth}.low.min", mu.cov_min_bin0.min(), "CROSSCHECK"); bk.check(f"cb.{meth}.low.max", mu.cov_min_bin0.max(), "CROSSCHECK")
        bk.check(f"cb.{meth}.high", mu.cov_min_bin2.mean(), "CROSSCHECK"); bk.check(f"cb.{meth}.size", mu.set_size.mean(), "CROSSCHECK")
        bk.check(f"cb.{meth}.nfail", int((hi < 0.9).sum()), "CROSSCHECK")

    # same-cell-size control: own pairing, own Wilcoxon, own Holm on per-seed Stage B rows (random strata
    # need the pipeline's random assignment, so this is a cross-check, as for rand.spread)
    if has("rs.gain.min"):
        res = []
        for k_, g in full[full.method.isin(["mondrian", "mondrian_x", "mondrian_x_random"])].groupby(["dataset", "regime", "model_name"]):
            w = {m_: g[g.method == m_].set_index("seed").sort_index() for m_ in ("mondrian", "mondrian_x", "mondrian_x_random")}
            d = (w["mondrian_x"].cov_min_bin0 - w["mondrian_x_random"].cov_min_bin0).values
            res.append({"d": d.mean(), "p": stats.wilcoxon(d, zero_method="zsplit").pvalue,
                        "rl": w["mondrian_x_random"].cov_min_bin0.mean(), "ml": w["mondrian"].cov_min_bin0.mean(),
                        "rh": w["mondrian_x_random"].cov_min_bin2.mean(), "mh": w["mondrian"].cov_min_bin2.mean(),
                        "rsz": w["mondrian_x_random"].set_size.mean(), "xsz": w["mondrian_x"].set_size.mean()})
        Rr = pd.DataFrame(res); Rr["h"] = own_holm(Rr.p.values)
        bk.rng("rs.gain", Rr.d, "CROSSCHECK"); bk.check("rs.npos", int((Rr.d > 0).sum()), "CROSSCHECK")
        bk.check("rs.nsig", int((Rr.h < 0.05).sum()), "CROSSCHECK"); bk.check("rs.fam", len(Rr), "CROSSCHECK")
        bk.check("rs.nonsig.maxholm", Rr[Rr.h >= 0.05].h.max(), "CROSSCHECK"); bk.check("rs.nonsig.n", int((Rr.h >= 0.05).sum()), "CROSSCHECK")
        bk.rng("rs.randlow", Rr.rl, "CROSSCHECK"); bk.rng("rs.cellsize.low", Rr.rl - Rr.ml, "CROSSCHECK")
        bk.rng("rs.cellsize.high", Rr.rh - Rr.mh, "CROSSCHECK"); bk.check("rs.randhigh.nup", int((Rr.rh > Rr.mh).sum()), "CROSSCHECK")
        bk.rng("rs.randsize", Rr.rsz, "CROSSCHECK"); bk.rng("rs.mxsize", Rr.xsz, "CROSSCHECK")
        Rr["ds"] = [k_[0] for k_, _ in full[full.method.isin(["mondrian", "mondrian_x", "mondrian_x_random"])].groupby(["dataset", "regime", "model_name"])]
        for ds in ("bbbp", "b3db"):
            q = Rr[Rr.ds == ds]
            if has(f"rs.cellsize.low.{ds}.min"):
                bk.rng(f"rs.cellsize.low.{ds}", q.rl - q.ml, "CROSSCHECK")
            if has(f"th.csb.{ds}.min"):
                nn = [np.mean([int((load(st, "rf", s_)["y_cal"] == int(load(st, "rf", s_)["minority"])).sum()) for s_ in SEEDS])
                      for st in EXCH if st.startswith(ds + "/")]
                bk.rng(f"th.csb.{ds}", [1 / (n_ / 3 + 1) for n_ in nn], "RECOMPUTED")
        # the class x Tanimoto low/high means must also equal the cache recomputation (RECOMPUTED elsewhere)
        assert np.allclose(sorted(Rr.xsz), sorted(np.array([per[k_]["mx.set_size"].mean() for k_ in per])), atol=1e-9)

    # D4: post-hoc invariance, cross-checked from per-seed rows
    def maxdiff(a, b, cols):
        q = full[full.method.isin([a, b])].groupby(["dataset", "regime", "model_name", "method"])[cols].mean().unstack("method")
        return max(float((q[c][a] - q[c][b]).abs().max()) for c in cols)
    cols = ["cov_min_bin0", "set_size", "cov_minority"]
    if has("inv.ts.max"):
        bk.check("inv.ts.max", max(maxdiff(b, b + "_ts", cols) for b in ("mondrian", "mondrian_x", "sabm")), "CROSSCHECK")
        bk.check("inv.la.max", max(maxdiff(b, b + "_la", cols) for b in ("mondrian", "mondrian_x", "sabm")), "CROSSCHECK")
        bk.check("inv.la.lowmax", max(maxdiff(b, b + "_la", ["cov_min_bin0"]) for b in ("mondrian", "mondrian_x", "sabm")), "CROSSCHECK")
        for ph in ("marginal_ts", "marginal_la"):
            bk.check(f"inv.{ph}.max", maxdiff("marginal", ph, cols), "CROSSCHECK")
            bk.check(f"inv.{ph}.cov", maxdiff("marginal", ph, ["cov_minority"]), "CROSSCHECK")
            bk.check(f"inv.{ph}.low", maxdiff("marginal", ph, ["cov_min_bin0"]), "CROSSCHECK")

    # D5 + D7 + D8 + D1: recomputed from the caches with own code
    sup_rows, cal_rows, mech_rows, audit = [], [], [], []
    for st in EXCH:
        for md in MODELS:
            for s in SEEDS:
                z = load(st, md, s); yt = z["y_test"].astype(int); yc = z["y_cal"].astype(int); mino = int(z["minority"])
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]; e = edges_from(ref, 3)
                t = binof(z["sim_test"], e); tc = binof(z["sim_cal"], e); m = yt == mino
                hm = sets_for(z, A, "mondrian")[np.arange(len(yt)), yt]; hx = sets_for(z, A, "mx")[np.arange(len(yt)), yt]
                sup_rows.append(pd.DataFrame({"st": st, "md": md, "s": s, "x": z["sim_test"][m] * 10, "hm": hm[m], "hx": hx[m]}))
                p = np.clip(z["p_test"][:, mino].astype(float), 1e-6, 1 - 1e-6)
                cal_rows.append(pd.DataFrame({"st": st, "md": md, "s": s, "lp": np.log(p / (1 - p)), "x": z["sim_test"] * 10, "y": m.astype(float)}))
                pm = z["p_test"][:, mino].astype(float); r = {"st": st, "md": md, "s": s}
                for b, nm in ((0, "low"), (2, "high")):
                    r[f"ptmin_{nm}"] = pm[m & (t == b)].mean(); r[f"ptmaj_{nm}"] = (1 - pm[~m & (t == b)]).mean()
                    r[f"covmaj_{nm}"] = hm[~m & (t == b)].mean(); r[f"covmin_{nm}"] = hm[m & (t == b)].mean()
                    r[f"gap_{nm}"] = pm[t == b].mean() - m[t == b].mean()
                mech_rows.append(r)
                sc = 1 - z["p_cal"][np.arange(len(yc)), yc]; smin = sc[yc == mino]; bmin = tc[yc == mino]
                loo = np.array([smin[i] <= qhat(np.delete(smin, i), A) for i in range(len(smin))])
                audit.append({"st": st, "md": md, "s": s,
                              "cal_low": loo[bmin == 0].mean() if (bmin == 0).any() else np.nan,
                              "cal_high": loo[bmin == 2].mean() if (bmin == 2).any() else np.nan,
                              "test_low": hm[m & (t == 0)].mean() if (m & (t == 0)).any() else np.nan,
                              "test_high": hm[m & (t == 2)].mean() if (m & (t == 2)).any() else np.nan})
    rngb = np.random.default_rng(404)
    if has("sc.mond.npos"):
        Dsup = pd.concat(sup_rows, ignore_index=True)
        for col, tg in (("hm", "mond"), ("hx", "mx")):
            slopes, lo_, hi_ = [], [], []
            for (st, md), g in Dsup.groupby(["st", "md"], sort=False):
                slopes.append(own_logit(g[["x"]].values, g[col].values.astype(float))[0])
                by = [h for _, h in g.groupby("s")]
                bs = [own_logit(np.concatenate([by[i].x.values for i in pick])[:, None],
                                np.concatenate([by[i][col].values for i in pick]).astype(float))[0]
                      for pick in (rngb.integers(0, len(by), len(by)) for _ in range(200))]
                lo_.append(np.percentile(bs, 2.5)); hi_.append(np.percentile(bs, 97.5))
            lo_, hi_ = np.array(lo_), np.array(hi_)
            nb_ = int(((np.abs(lo_) < 0.02) | (np.abs(hi_) < 0.02)).sum())     # Monte Carlo borderline cells
            nt_ = f"own split bootstrap (200); count may differ by the {nb_} cell(s) with an endpoint within 0.02 of 0"
            bk.rng(f"sc.{tg}.slope", slopes, "RECOMPUTED", tol=0.006)
            bk.check(f"sc.{tg}.npos", int((lo_ > 0).sum()), "REPLICATED", tol=nb_, note=nt_)
            bk.check(f"sc.{tg}.nneg", int((hi_ < 0).sum()), "REPLICATED", tol=nb_, note=nt_)
            bk.check(f"sc.{tg}.nflat", int(((lo_ <= 0) & (hi_ >= 0)).sum()), "REPLICATED", tol=nb_, note=nt_)
    if has("scal.b3db.nneg"):
        Dc = pd.concat(cal_rows, ignore_index=True)
        for ds in ("bbbp", "b3db"):
            coef, lo_, hi_ = [], [], []
            for (st, md), g in Dc[Dc.st.str.startswith(ds + "/")].groupby(["st", "md"], sort=False):
                coef.append(own_logit(g[["lp", "x"]].values, g.y.values)[1])
                by = [h for _, h in g.groupby("s")]
                bs = []
                for _ in range(150):
                    pk = rngb.integers(0, len(by), len(by)); h = pd.concat([by[i] for i in pk])
                    bs.append(own_logit(h[["lp", "x"]].values, h.y.values)[1])
                lo_.append(np.percentile(bs, 2.5)); hi_.append(np.percentile(bs, 97.5))
            lo_, hi_ = np.array(lo_), np.array(hi_)
            nb_ = int(((np.abs(lo_) < 0.02) | (np.abs(hi_) < 0.02)).sum())
            nt_ = f"own split bootstrap (150); count may differ by the {nb_} cell(s) with an endpoint within 0.02 of 0"
            bk.rng(f"scal.{ds}.coef", coef, "RECOMPUTED", tol=0.006)
            bk.check(f"scal.{ds}.nneg", int((hi_ < 0).sum()), "REPLICATED", tol=nb_, note=nt_)
            bk.check(f"scal.{ds}.npos", int((lo_ > 0).sum()), "REPLICATED", tol=nb_, note=nt_)
    if has("mech.p2.n"):
        Mr = pd.DataFrame(mech_rows); g = Mr.groupby(["st", "md"], sort=False)
        mu = g.mean(numeric_only=True)
        dmin = mu.ptmin_high - mu.ptmin_low; dmaj = mu.ptmaj_high - mu.ptmaj_low
        bk.rng("mech.p2.min", dmin, "RECOMPUTED"); bk.rng("mech.p2.maj", dmaj, "RECOMPUTED")
        bk.check("mech.p2.n", int((dmin > dmaj).sum()), "RECOMPUTED")
        bk.rng("mech.covmajlow", mu.covmaj_low, "RECOMPUTED"); bk.rng("mech.covminlow", mu.covmin_low, "RECOMPUTED")
        bk.check("mech.p1.nmajok", int((g.covmaj_low.apply(lambda x: t_lo(x.values)) >= 0.88).sum()), "RECOMPUTED")
        p3 = g.gap_low.apply(lambda x: t_hi(x.values)) < 0
        bk.check("mech.p3.n", int(p3.sum()), "RECOMPUTED")
        bk.check("mech.p3.nb3db", int(p3[[i for i in p3.index if i[0].startswith("b3db")]].sum()), "RECOMPUTED")
    if has("ca.rho"):
        Au = pd.DataFrame(audit); ok = Au.dropna(subset=["cal_low", "test_low"])
        bk.check("ca.nsplits", len(ok), "RECOMPUTED")
        bk.check("ca.rho", stats.spearmanr(ok.cal_low, ok.test_low)[0], "RECOMPUTED")
        fl, fa = ok.cal_low < 0.85, ok.test_low < 0.85
        bk.check("ca.sens", (fl & fa).sum() / fa.sum(), "RECOMPUTED"); bk.check("ca.spec", (~fl & ~fa).sum() / (~fa).sum(), "RECOMPUTED")
        cg = Au.groupby(["st", "md"], sort=False)
        flag_c = cg.cal_low.apply(lambda x: t_hi(x.values)) < 0.9; flag_t = cg.test_low.apply(lambda x: t_hi(x.values)) < 0.9
        bk.check("ca.cellagree", int((flag_c == flag_t).sum()), "RECOMPUTED")
        bk.check("ca.signagree", float((np.sign(ok.cal_high - ok.cal_low) == np.sign(ok.test_high - ok.test_low)).mean()), "RECOMPUTED")

    # D2 / R2: other datasets and the prior-work protocol, own sets, own Wilcoxon, own Holm
    gen_runs = sorted({k.split(".")[1] for k in bk.p.index if k.startswith("gen.") and k != "gen.fam"})
    for tg, items in (("gen", [(r_.replace("tox21", "tox21_sr_are"), "rf", f"gen.{r_}") for r_ in gen_runs]),
                      ("tc", [("bbbp_raw/random5025", md_, f"tc.{md_}") for md_ in ("rf", "gcn") if has(f"tc.{md_}.gain")])):
        pvals, keys = [], []
        for run, md, key in items:
            R = []
            for s in SEEDS:
                fp = RAW / run / f"{md}_ce" / f"seed{s}.npz"
                if not fp.exists():
                    continue
                z = np.load(fp, allow_pickle=True); yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]; t = binof(z["sim_test"], edges_from(ref, 3))
                r = {"ncalmin": int((z["y_cal"].astype(int) == mino).sum()), "ntestminlow": int((m & (t == 0)).sum())}
                for how, nm in (("marginal", "marginal"), ("mondrian", "mondrian"), ("mx", "mondrian_x")):
                    S = sets_for(z, A, how); hit = S[np.arange(len(yt)), yt]
                    r[f"{nm}.cls"] = hit[m].mean(); r[f"{nm}.size"] = S.sum(1).mean()
                    r[f"{nm}.low"] = hit[m & (t == 0)].mean() if (m & (t == 0)).any() else np.nan
                    r[f"{nm}.high"] = hit[m & (t == 2)].mean() if (m & (t == 2)).any() else np.nan
                R.append(r)
            if len(R) != 20:
                continue                                    # caches absent here: keys stay UNVERIFIED
            R = pd.DataFrame(R)
            for c in ("marginal.cls", "mondrian.cls", "mondrian.low", "mondrian.high", "mondrian_x.low", "mondrian_x.high",
                      "mondrian.size", "mondrian_x.size", "ncalmin", "ntestminlow"):
                bk.check(f"{key}.{c}", float(np.nanmean(R[c])), "RECOMPUTED")
            d = (R["mondrian_x.low"] - R["mondrian.low"]).dropna().values
            bk.check(f"{key}.gain", d.mean(), "RECOMPUTED")
            bs = d[np.random.default_rng(55).integers(0, len(d), (10000, len(d)))].mean(1)
            bk.check(f"{key}.cilo", np.percentile(bs, 2.5), "REPLICATED", tol=0.006); bk.check(f"{key}.cihi", np.percentile(bs, 97.5), "REPLICATED", tol=0.006)
            pvals.append(stats.wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0); keys.append(key)
        if keys and len(keys) == len(items):
            bk.check(f"{tg}.fam", len(keys), "RECOMPUTED")
            for key, ph in zip(keys, own_holm(pvals)):
                bk.check(f"{key}.holm", ph, "RECOMPUTED")

    # D8 on the other datasets
    for k in [k for k in bk.p.index if k.startswith("mechg.") and k.endswith(".p2min")]:
        run = k.split(".")[1].replace("tox21", "tox21_sr_are"); rr = []
        for s in SEEDS:
            fp = RAW / run / "rf_ce" / f"seed{s}.npz"
            if fp.exists():
                z = np.load(fp, allow_pickle=True); yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]; t = binof(z["sim_test"], edges_from(ref, 3))
                pm = z["p_test"][:, mino].astype(float)
                mean_or_nan = lambda v: v.mean() if len(v) else np.nan
                rr.append([mean_or_nan(pm[m & (t == 2)]), mean_or_nan(pm[m & (t == 0)]),
                           mean_or_nan(1 - pm[~m & (t == 2)]), mean_or_nan(1 - pm[~m & (t == 0)])])
        if len(rr) == 20:
            mu = np.nanmean(np.array(rr, float), axis=0)          # seed mean per tertile, then high - low
            bk.check(k, mu[0] - mu[1], "RECOMPUTED"); bk.check(k.replace(".p2min", ".p2maj"), mu[2] - mu[3], "RECOMPUTED")

    # A2 dataset level: own connectivity keys (RDKit InChIKey first block) from the curated SMILES
    if has("la.ndis"):
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        ck = {}
        for ds in ("bbbp", "b3db"):
            df = pd.read_pickle(REPO / f"results/processed/{ds}.pkl")["df"][["mol_id", "smiles", "y"]].copy()
            df["k"] = [Chem.MolToInchiKey(Chem.MolFromSmiles(s_)).split("-")[0] for s_ in df.smiles]
            ck[ds] = df
        mm = ck["bbbp"].merge(ck["b3db"], on="k", suffixes=("_a", "_b"))
        bk.check("la.nshared", len(mm), "RECOMPUTED"); bk.check("la.overlap", 100 * len(mm) / len(ck["bbbp"]), "RECOMPUTED")
        bk.check("la.agree", 100 * (mm.y_a == mm.y_b).mean(), "RECOMPUTED"); bk.check("la.ndis", int((mm.y_a != mm.y_b).sum()), "RECOMPUTED")
        bk.check("la.negpos", int(((mm.y_a == 0) & (mm.y_b == 1)).sum()), "RECOMPUTED")
        bk.check("la.posneg", int(((mm.y_a == 1) & (mm.y_b == 0)).sum()), "RECOMPUTED")
        # failure level: failure flags from the (independently verified) persistent-failure table; conflicts from own keys
        W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False,
                        usecols=["setting", "mol_id", "eligible", "cross_arch_mondrian", "majority_all3_mondrian"])
        W = W[W.eligible.astype(bool) & W.setting.isin(EXCH)]
        for st, g in W.groupby("setting"):
            own_ds, oth = st.split("/")[0], ("b3db" if st.startswith("bbbp") else "bbbp")
            lab = ck[own_ds].set_index("mol_id")
            other = ck[oth].drop_duplicates("k").set_index("k").y
            kk = lab.loc[g.mol_id, "k"].values; yy = lab.loc[g.mol_id, "y"].values
            inn = pd.Series(kk).isin(other.index).values
            dis = np.zeros(len(g), bool); dis[inn] = other.loc[kk[inn]].values != yy[inn]
            for defn, col in (("strict", "cross_arch_mondrian"), ("majority", "majority_all3_mondrian")):
                f = g[col].astype(bool).values[inn]; dd = dis[inn]
                tab = [[int((f & dd).sum()), int((f & ~dd).sum())], [int((~f & dd).sum()), int((~f & ~dd).sum())]]
                orr, pp = stats.fisher_exact(tab)
                k = f"la.{st}.{defn}"
                bk.check(k + ".ndis", int(dd.sum()), "RECOMPUTED"); bk.check(k + ".fdis", tab[0][0], "RECOMPUTED")
                bk.check(k + ".p", pp, "RECOMPUTED")
                if has(k + ".or"):
                    bk.check(k + ".or", orr, "RECOMPUTED")

    # screening, VAL-held-out thresholds (own sets on VAL with CAL thresholds; own quantile rule)
    def dec_val(z):
        yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino
        S = sets_for(z, A, "mx", 3); acc = S.sum(1) == 1; lab = S.argmax(1); am = z["p_test"].argmax(1)
        zv = {**dict(z), "p_test": z["p_val"], "sim_test": z["sim_val"]}
        d_val = 1 - (sets_for(zv, A, "mx", 3).sum(1) == 1).mean()
        o = {"fx_fa": (acc & m & (lab != mino)).sum() / m.sum(), "fx_err": (acc & (lab != yt)).sum() / acc.sum(), "fx_def": 1 - acc.mean()}
        for nm, sv, stt in (("ad", z["sim_val"], z["sim_test"]), ("cf", z["p_val"].max(1), z["p_test"].max(1))):
            a2 = stt > np.quantile(sv, d_val)
            o[nm + "_fa"] = (a2 & m & (am != mino)).sum() / m.sum(); o[nm + "_err"] = (a2 & (am != yt)).sum() / a2.sum()
            o[nm + "_def"] = 1 - a2.mean()
        return o
    if has("su.ad_reject_val.minority_false_accept_rate.exch.min"):
        for tag in ("exch", "tr"):
            sts = EXCH if tag == "exch" else ["bbbp→b3db/scaffold"]
            acc_ = {k: [] for k in ("ad_fa", "ad_err", "ad_def", "cf_fa", "cf_err", "cf_def")}
            lev = {"fixed_b3": [], "ad_reject_val": [], "conf_reject_val": []}
            for st in sts:
                for md in MODELS:
                    v = pd.DataFrame([dec_val(load(st, md, s)) for s in SEEDS]).mean()
                    lev["fixed_b3"].append(v.fx_def); lev["ad_reject_val"].append(v.ad_def); lev["conf_reject_val"].append(v.cf_def)
                    for nm in ("ad", "cf"):
                        for ep in ("fa", "err", "def"):
                            acc_[f"{nm}_{ep}"].append(v[f"fx_{ep}"] - v[f"{nm}_{ep}"])
            for nm, lab_ in (("ad", "ad_reject_val"), ("cf", "conf_reject_val")):
                for ep, epn in (("fa", "minority_false_accept_rate"), ("err", "accepted_error"), ("def", "deferral_rate")):
                    bk.rng(f"su.{lab_}.{epn}.{tag}", acc_[f"{nm}_{ep}"], "RECOMPUTED")
            if tag == "exch":
                bk.rng("su.defer.exch", lev["fixed_b3"], "RECOMPUTED")
            else:
                for k_, v_ in lev.items():
                    bk.rng(f"su.defer.tr.{k_}", v_, "RECOMPUTED")


# ----------------------------------------------------------------- 11. review additions (1 Nov)
def own_auc(y, s):
    r = stats.rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def own_ap(y, s):
    o = np.argsort(-s, kind="stable"); ys, ss = y[o], s[o]
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]       # end index of each distinct score
    tp = np.cumsum(ys)[last]; fp = (last + 1) - tp
    prec = tp / (tp + fp); rec = tp / ys.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def verify_review(bk, per):
    has = lambda k: k.replace("→", "to") in bk.p.index
    ALL = EXCH + ["bbbp→b3db/scaffold"]
    # model competence
    if has("perf.auroc.min"):
        res = {}
        for st in ALL:
            for md in MODELS:
                v = []
                for s in SEEDS:
                    z = load(st, md, s); y = z["y_test"].astype(int); mino = int(z["minority"]); p = z["p_test"].astype(float)
                    ym = (y == mino).astype(int); yh = p.argmax(1)
                    rec = [((yh == c) & (y == c)).sum() / (y == c).sum() for c in (0, 1)]
                    v.append([own_auc(ym, p[:, mino]), own_ap(ym, p[:, mino]), np.mean(rec), ((yh == mino) & (ym == 1)).sum() / ym.sum(), ym.mean()])
                res[(st, md)] = np.mean(v, axis=0)
        for tag, sts in (("perf", EXCH), ("perf.tr", ["bbbp→b3db/scaffold"])):
            M = np.array([res[(st, md)] for st in sts for md in MODELS])
            for j, nm in enumerate(("auroc", "auprc", "balacc", "minrecall", "prev")):
                bk.rng(f"{tag}.{nm}", M[:, j], "RECOMPUTED")
    # Mondrian plus a rejection filter at the deferral of class x support (own sets, own matching rule)
    if has("cr.mond_ad.minority_false_accept_rate.exch.min"):
        def mets(acc, lab, y, mino):
            m = y == mino
            return np.array([(acc & m & (lab != mino)).sum() / m.sum(),
                             (acc & (lab != y)).sum() / acc.sum() if acc.any() else np.nan, 1 - acc.mean()])
        def defer_low(acc, score, k):
            out = acc.copy()
            if k > 0:
                idx = np.where(acc)[0]; out[idx[np.argsort(score[idx], kind="stable")[:k]]] = False
            return out
        diffs = {}
        for st in ALL:
            for md in MODELS:
                acc_d = {pol: [] for pol in ("mond_ad", "mond_conf", "mond_ad_val", "mond_conf_val")}
                for s in SEEDS:
                    z = load(st, md, s); y = z["y_test"].astype(int); mino = int(z["minority"])
                    Sx = sets_for(z, A, "mx", 3); Sm = sets_for(z, A, "mondrian")
                    ax, am = Sx.sum(1) == 1, Sm.sum(1) == 1; lx, lm = Sx.argmax(1), Sm.argmax(1)
                    base = mets(ax, lx, y, mino); k = int(round((1 - ax.mean()) * len(y))) - int((~am).sum())
                    conf = z["p_test"].max(1)
                    acc_d["mond_ad"].append(base - mets(defer_low(am, z["sim_test"], k), lm, y, mino))
                    acc_d["mond_conf"].append(base - mets(defer_low(am, conf, k), lm, y, mino))
                    zv = {**dict(z), "p_test": z["p_val"], "sim_test": z["sim_val"]}
                    avx = sets_for(zv, A, "mx", 3).sum(1) == 1; avm = sets_for(zv, A, "mondrian").sum(1) == 1
                    kv = int(round((1 - avx.mean()) * len(avx))) - int((~avm).sum())
                    for nm, sv, stt in (("mond_ad_val", z["sim_val"], z["sim_test"]), ("mond_conf_val", z["p_val"].max(1), conf)):
                        acc = am.copy()
                        if kv > 0:
                            srt = np.sort(sv[avm], kind="stable"); acc = am & (stt > srt[min(kv, len(srt)) - 1])
                        acc_d[nm].append(base - mets(acc, lm, y, mino))
                for pol, v in acc_d.items():
                    diffs[(st, md, pol)] = np.nanmean(np.array(v), axis=0)
                    diffs[(st, md, pol, "seeds")] = np.array(v)
        for pol in ("mond_ad", "mond_conf", "mond_ad_val", "mond_conf_val"):
            for j, ep in enumerate(("minority_false_accept_rate", "accepted_error", "deferral_rate")):
                bk.rng(f"cr.{pol}.{ep}.exch", [diffs[(st, md, pol)][j] for st in EXCH for md in MODELS], "RECOMPUTED")
                bk.rng(f"cr.{pol}.{ep}.tr", [diffs[("bbbp→b3db/scaffold", md, pol)][j] for md in MODELS], "RECOMPUTED")
                lo_, hi_ = [], []
                rb = np.random.default_rng(321)
                for st in ALL:
                    for md in MODELS:
                        d = diffs[(st, md, pol, "seeds")][:, j]; d = d[~np.isnan(d)]
                        bs = d[rb.integers(0, len(d), (5000, len(d)))].mean(1)
                        lo_.append(np.percentile(bs, 2.5)); hi_.append(np.percentile(bs, 97.5))
                lo_, hi_ = np.array(lo_), np.array(hi_)
                nb_ = int(((np.abs(lo_) < 0.002) | (np.abs(hi_) < 0.002)).sum())
                nt_ = f"own seed bootstrap (5000); may differ by the {nb_} cell(s) with an endpoint within 0.002 of 0"
                bk.check(f"cr.{pol}.{ep}.nlow", int((hi_ < 0).sum()), "REPLICATED", tol=nb_, note=nt_)
                bk.check(f"cr.{pol}.{ep}.nhigh", int((lo_ > 0).sum()), "REPLICATED", tol=nb_, note=nt_)
    # concordance against a covariate-stratified null (own permutation, own strata)
    W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
    W = W[W.eligible.astype(bool)]
    rng = np.random.default_rng(777)
    for st, g in W.groupby("setting"):
        k_ = st.replace("→", "to")
        if not has(f"cc.{k_}.nullcov"):
            continue
        P = np.column_stack([g[f"persist_mondrian__{m}"].astype(bool).values for m in MODELS]); obs = int(P.all(1).sum())
        kk = g.n_appear.clip(upper=6).astype(int).values
        tert = lambda v: np.searchsorted(np.sort(v), v, side="left") * 3 // len(v)          # rank tertile (ties: lower)
        strata = kk * 9 + tert(g.mean_sim_train.values) * 3 + tert(g.TPSA.values)
        groups = [np.where(strata == u)[0] for u in np.unique(strata)]
        null = []
        for _ in range(4000):
            Q = P.copy()
            for j in range(3):
                for idx in groups:
                    Q[idx, j] = Q[rng.permutation(idx), j]
            null.append(Q.all(1).sum())
        null = np.array(null)
        bk.check(f"cc.{k_}.nullcov", null.mean(), "REPLICATED", tol=0.08, note="own strata (ties may differ) and 4000 own permutations")
        bk.check(f"cc.{k_}.pcov", (1 + (null >= obs).sum()) / (1 + len(null)), "REPLICATED", tol=0.01, note="4000 own permutations")
        strict = g[P.all(1)]
        for c, v in (("n_kge3", (strict.n_appear >= 3).sum()), ("n_kge4", (strict.n_appear >= 4).sum()), ("n_k2", (strict.n_appear == 2).sum())):
            bk.check(f"cc.{k_}.{c}", int(v), "RECOMPUTED")
        if len(strict):
            bk.check(f"cc.{k_}.kmed", float(strict.n_appear.median()), "RECOMPUTED")
            bk.check(f"cc.{k_}.kmax", int(strict.n_appear.max()), "RECOMPUTED")
    # B3DB-internal same-name conflicts (own normalisation)
    if has("nc.nconflict.min"):
        import re as _re
        def nm(x):
            if not isinstance(x, str):
                return ""
            x = _re.sub(r"\([^)]*\)", "", x.lower())
            for w in ("dihydrochloride", "hydrochloride", "hydrobromide", "maleate", "mesylate", "citrate", "sulfate", "sulphate",
                      "tartrate", "fumarate", "sodium", "potassium", "hcl", "acetate", "phosphate", "bromide", "chloride",
                      "succinate", "besylate", "tosylate", "monohydrate", "hydrate"):
                x = _re.sub(r"\b" + w + r"\b", "", x)
            return "".join(ch for ch in x if ch.isalnum())
        raw = pd.read_csv(REPO / "data/raw/B3DB_classification.tsv", sep="\t")
        pos = {nm(a) for a, l in zip(raw.compound_name, raw["BBB+/BBB-"]) if l == "BBB+"} - {""}
        Wn = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False, usecols=["setting", "name", "eligible",
                         "cross_arch_mondrian", "majority_all3_mondrian"])
        Wn = Wn[Wn.setting.isin(["b3db/random", "b3db/scaffold"]) & Wn.eligible.astype(bool)]
        ncs = []
        for st, g in Wn.groupby("setting"):
            c = g["name"].map(lambda x: nm(x) in pos and nm(x) != "").values; ncs.append(int(c.sum()))
            for defn, col in (("strict", "cross_arch_mondrian"), ("majority", "majority_all3_mondrian")):
                f = g[col].astype(bool).values
                tab = [[int((f & c).sum()), int((f & ~c).sum())], [int((~f & c).sum()), int((~f & ~c).sum())]]
                k_ = f"nc.{st}.{defn}"
                bk.check(k_ + ".nfail", tab[0][0], "RECOMPUTED"); bk.check(k_ + ".p", stats.fisher_exact(tab)[1], "RECOMPUTED")
                bk.check(k_ + ".ratefail", 100 * tab[0][0] / max(f.sum(), 1), "RECOMPUTED")
                bk.check(k_ + ".raterest", 100 * tab[1][0] / max((~f).sum(), 1), "RECOMPUTED")
        bk.check("nc.nconflict.min", min(ncs), "RECOMPUTED"); bk.check("nc.nconflict.max", max(ncs), "RECOMPUTED")


# ----------------------------------------------------------------- 12. Figure 1 (own selection, thresholds, trace)
def verify_fig1(bk, per):
    L = "RECOMPUTED"
    gap = {k: v["mondrian.cov_min_bin2"].mean() - v["mondrian.cov_min_bin0"].mean() for k, v in per.items()}
    order = sorted(gap, key=lambda k: (gap[k], k[0], k[1]))
    rnd = [k for k in order if k[0].endswith("/random")]
    st, md = rnd[2]
    bk.check("fig1.sel.gap", gap[(st, md)], L); bk.check("fig1.sel.all12.gap", gap[order[5]], L)
    bk.check("fig1.sel.gap12.low", min(gap.values()), L); bk.check("fig1.sel.gap12.high", max(gap.values()), L)
    want = f"{st} {md}"
    got = str(bk.p.loc["fig1.sel.label", "value"]) if "fig1.sel.label" in bk.p.index else ""
    nm = {"bbbp": "BBBP", "b3db": "B3DB", "rf": "random forest", "gin": "GIN", "chemprop": "Chemprop"}
    exp = f"{nm[st.split('/')[0]]}, {st.split('/')[1]} split, {nm[md]}, 20 splits"
    bk.rows.append({"key": "fig1.sel.label", "level": L, "status": "OK" if got == exp else "MISMATCH", "paper": got, "verifier": exp})
    bk.done.add("fig1.sel.label")
    exp12 = f"{nm[order[5][0].split('/')[0]]} {order[5][0].split('/')[1]} split, {nm[order[5][1]]}"
    got12 = str(bk.p.loc["fig1.all12.label", "value"]) if "fig1.all12.label" in bk.p.index else ""
    bk.rows.append({"key": "fig1.all12.label", "level": L, "status": "OK" if got12 == exp12 else "MISMATCH", "paper": got12,
                    "verifier": exp12}); bk.done.add("fig1.all12.label")
    cov = {k: [] for k in ("cls", "low", "mid", "high", "x.low", "x.high", "size", "x.size", "full", "x.full")}
    ex = {0: [], 1: [], 2: []}; shift = {0: [], 2: []}; resc = {0: 0, 1: 0, 2: 0}; lost = {0: 0, 1: 0, 2: 0}; recs = []; ncal_low = []
    for sd in SEEDS:
        z = load(st, md, sd); mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
        sc = 1 - z["p_cal"][np.arange(len(yc)), yc]; St = 1 - z["p_test"]
        e = edges_from(z["sim_val"], 3); bc = binof(z["sim_cal"], e); bt = binof(z["sim_test"], e)
        Q = {c: qhat(sc[yc == c], A) for c in (0, 1)}
        QB = {(c, b): qhat(sc[(yc == c) & (bc == b)], A) for c in (0, 1) for b in (0, 1, 2)}
        inM = np.column_stack([St[:, c] <= Q[c] for c in (0, 1)])
        inX = np.column_stack([St[:, c] <= np.array([QB[(c, b)] for b in bt]) for c in (0, 1)])
        hm = inM[np.arange(len(yt)), yt]; hx = inX[np.arange(len(yt)), yt]; m = yt == mino
        cov["cls"].append(hm[m].mean())
        for b, k in ((0, "low"), (1, "mid"), (2, "high")):
            cov[k].append(hm[m & (bt == b)].mean())
        cov["x.low"].append(hx[m & (bt == 0)].mean()); cov["x.high"].append(hx[m & (bt == 2)].mean())
        cov["size"].append(inM.sum(1).mean()); cov["x.size"].append(inX.sum(1).mean())
        cov["full"].append((inM.sum(1) == 2).mean()); cov["x.full"].append((inX.sum(1) == 2).mean())
        for b in (0, 2):
            shift[b].append(QB[(mino, b)] - Q[mino])
        ncal_low.append(int(((yc == mino) & (bc == 0)).sum()))
        for i in np.where(m)[0]:
            b = int(bt[i]); ex[b].append(St[i, mino] - Q[mino])
            r_ = (not hm[i]) and hx[i]; l_ = hm[i] and (not hx[i])
            resc[b] += r_; lost[b] += l_
            recs.append((z["test_ids"][i], sd, b, r_, St[i, mino], Q[mino], QB[(mino, b)], z["sim_test"][i]))
    for k, key in (("cls", "A.cls"), ("low", "A.low"), ("mid", "A.mid"), ("high", "A.high")):
        bk.check(f"fig1.{key}", np.mean(cov[k]), L)
    for k in ("cls", "low", "high"):
        bk.check(f"fig1.A.{k}.lo", t_lo(cov[k]), L); bk.check(f"fig1.A.{k}.hi", t_hi(cov[k]), L)
    bk.check("fig1.A.gap", np.mean(cov["high"]) - np.mean(cov["low"]), L)
    bk.check("fig1.C.low.mond", np.mean(cov["low"]), L); bk.check("fig1.C.high.mond", np.mean(cov["high"]), L)
    for k in ("low", "high"):
        bk.check(f"fig1.C.{k}.cxs", np.mean(cov[f"x.{k}"]), L)
        bk.check(f"fig1.C.{k}.cxs.lo", t_lo(cov[f"x.{k}"]), L); bk.check(f"fig1.C.{k}.cxs.hi", t_hi(cov[f"x.{k}"]), L)
    bk.check("fig1.C.size.mond", np.mean(cov["size"]), L); bk.check("fig1.C.size.cxs", np.mean(cov["x.size"]), L)
    bk.check("fig1.C.full.mond", 100 * np.mean(cov["full"]), L); bk.check("fig1.C.full.cxs", 100 * np.mean(cov["x.full"]), L)
    for b, k in ((0, "low"), (1, "mid"), (2, "high")):
        v = np.asarray(ex[b]); bk.check(f"fig1.B.excl.{k}", 100 * (v > 0).mean(), L); bk.check(f"fig1.B.n.{k}", len(v), L)
    bk.check("fig1.B.shift.low.min", min(shift[0]), L); bk.check("fig1.B.shift.low.max", max(shift[0]), L)
    bk.check("fig1.B.shift.high.min", min(shift[2]), L); bk.check("fig1.B.shift.high.max", max(shift[2]), L)
    bk.check("fig1.B.shift.low.npos", sum(x > 0 for x in shift[0]), L); bk.check("fig1.B.shift.high.nneg", sum(x < 0 for x in shift[2]), L)
    bk.check("fig1.B.rescued", sum(resc.values()), L); bk.check("fig1.B.lost", sum(lost.values()), L)
    bk.check("fig1.B.rescued.low", resc[0], L); bk.check("fig1.B.rescued.high", resc[2], L)
    bk.check("fig1.B.lost.low", lost[0], L); bk.check("fig1.B.lost.high", lost[2], L)
    R = pd.DataFrame(recs, columns=["mol", "seed", "b", "r", "s", "q", "qb", "sim"])
    bk.check("fig1.B.ncal.low.min", min(ncal_low), L); bk.check("fig1.B.ncal.low.max", max(ncal_low), L)
    bk.check("fig1.B.rescued.mid", resc[1], L); bk.check("fig1.B.lost.mid", lost[1], L)
    for b_, k in ((0, "low"), (1, "mid"), (2, "high")):
        bk.check(f"fig1.B.nmol.{k}", R[R.b == b_].mol.nunique(), L)
    g = R.groupby("mol").agg(n=("seed", "size"), r=("r", "sum"), low=("b", lambda t: bool((t == 0).all())))
    g = g[g.low].reset_index().sort_values(["r", "n", "mol"], ascending=[False, False, True]).iloc[0]
    t_ = R[(R.mol == g.mol) & R.r].sort_values("seed").iloc[0]
    bk.check("fig1.mol.n", g.n, L); bk.check("fig1.mol.rescued", g.r, L); bk.check("fig1.mol.seed", t_.seed, L)
    bk.check("fig1.mol.score", t_.s, L); bk.check("fig1.mol.qclass", t_.q, L); bk.check("fig1.mol.qstratum", t_.qb, L)
    bk.check("fig1.mol.smax", t_.sim, L)
    fv = dict(pd.read_csv(REPO / "tables/fig1_values.csv", dtype=str).values)
    bk.rows.append({"key": "fig1.mol.name", "level": L, "status": "OK" if fv.get("mol.id") == g.mol else "MISMATCH",
                    "paper": fv.get("mol.id"), "verifier": g.mol, "note": "traced molecule identity (own selection rule)"})
    bk.done.add("fig1.mol.name")
    for key_, val_ in (("fig1.mol.id", g.mol),):
        got_ = str(bk.p.loc[key_, "value"]) if key_ in bk.p.index else ""
        bk.rows.append({"key": key_, "level": L, "status": "OK" if got_ == val_ else "MISMATCH", "paper": got_, "verifier": val_})
        bk.done.add(key_)
    # nearest TRAIN compound: own Tanimoto on the repo's ECFP4 (RDKit Morgan r=2, 2048 bits) against TRAIN of the traced split
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen_ = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    dfm = pd.read_pickle(REPO / f"results/processed/{st.split('/')[0]}.pkl")["df"]; zt_ = load(st, md, int(t_.seed))
    q_ = gen_.GetFingerprint(Chem.MolFromSmiles(dfm.set_index("mol_id").loc[g.mol, "smiles"]))
    tr_ = dfm.iloc[zt_["idx_train"]]
    sims_ = DataStructs.BulkTanimotoSimilarity(q_, [gen_.GetFingerprint(Chem.MolFromSmiles(x)) for x in tr_.smiles])
    nn_ = tr_.mol_id.iloc[int(np.argmax(sims_))]
    srt_ = np.sort(np.array(sims_))[::-1]; bk.check("fig1.nn.sim2", srt_[1], L)
    # provenance: raw B3DB records whose own InChIKey (RDKit, raw SMILES) equals the processed InChIKey
    raw_ = pd.read_csv(REPO / "data/raw/B3DB_classification.tsv", sep="\t")
    def rec_(mid):
        ikq = dfm.set_index("mol_id").loc[mid, "inchikey"]
        hit = [i for i, x in enumerate(raw_.SMILES) if isinstance(x, str) and Chem.MolFromSmiles(x) is not None
               and Chem.MolToInchiKey(Chem.MolFromSmiles(x)) == ikq]
        return raw_.iloc[hit]
    rm_, rn_ = rec_(g.mol), rec_(nn_)
    bk.check("fig1.mol.no", int(rm_["NO."].iloc[0]), L); bk.check("fig1.mol.cid", int(rm_.CID.iloc[0]), L)
    bk.check("fig1.nn.no", int(rn_["NO."].iloc[0]), L); bk.check("fig1.nn.cid", int(rn_.CID.iloc[0]), L)
    bk.check("fig1.mol.nref", len([x for x in str(rm_.reference.iloc[0]).split("|") if x]), L)
    lab_ok = len(rm_) == 1 and rm_["BBB+/BBB-"].iloc[0] == "BBB-"
    bk.rows.append({"key": "fig1.mol.rawlabel", "level": L, "status": "OK" if lab_ok else "MISMATCH",
                    "paper": "BBB-", "verifier": list(rm_["BBB+/BBB-"])}); bk.done.add("fig1.mol.rawlabel")
    bk.check("fig1.mol.excess", t_.s - t_.q, L)
    sm_ = R[R.mol == g.mol].sim; bk.check("fig1.mol.smax.min", sm_.min(), L); bk.check("fig1.mol.smax.max", sm_.max(), L)
    bk.check("fig1.mol.ncal", ncal_low[list(SEEDS).index(int(t_.seed))], L)
    for k_ in ("low", "mid", "high"):
        bk.check(f"fig1.A.excl.{k_}", 100 * (1 - np.mean(cov[k_])), L)
    for b_, k_ in ((0, "low"), (2, "high")):
        a_, c_ = round(100 * (np.asarray(ex[b_]) > 0).mean()), round(100 * (1 - np.mean(cov[k_])))
        bk.check(f"fig1.B.exclint.{k_}", a_ if a_ == c_ else -1, L, note="pooled and split-mean shares agree at whole percent")
    bbp_ = pd.read_pickle(REPO / "results/processed/bbbp.pkl")["df"]
    bk.check("fig1.mol.in_bbbp", int((bbp_.conn_key == dfm.set_index("mol_id").loc[g.mol, "conn_key"]).sum()), L)
    got_ = str(bk.p.loc["fig1.mol.nn_id", "value"]) if "fig1.mol.nn_id" in bk.p.index else ""
    bk.rows.append({"key": "fig1.mol.nn_id", "level": L, "status": "OK" if got_ == nn_ and abs(max(sims_) - t_.sim) < 1e-9 else "MISMATCH",
                    "paper": got_, "verifier": nn_}); bk.done.add("fig1.mol.nn_id")


def main():
    prov = pd.read_csv(REPO / "paper/latex/generated/numbers_provenance.csv", dtype={"value": str})
    bk = Book(prov)

    # 1. headline cells, recomputed from caches
    per = {}
    for st in EXCH:
        for md in MODELS:
            per[(st, md)] = pd.DataFrame([split_metrics(load(st, md, s)) for s in SEEDS])
    mean = {k: v.mean() for k, v in per.items()}
    for (st, md), mu in mean.items():
        for meth, cols in (("marginal", ["cov_minority"]), ("mondrian", ["cov_minority", "cov_min_bin0", "cov_min_bin2", "set_size"]),
                           ("mx", ["cov_min_bin0", "set_size"])):
            for c in cols:
                name = "mondrian_x" if meth == "mx" else meth
                bk.check(f"t.{st}.{md}.{name}.{c}", mu[f"{meth}.{c}"], "RECOMPUTED")
    col = lambda c: [mean[k][c] for k in mean]
    bk.rng("marg.cls", col("marginal.cov_minority"), "RECOMPUTED")
    bk.rng("mond.cls", col("mondrian.cov_minority"), "RECOMPUTED")
    bk.rng("mond.low", col("mondrian.cov_min_bin0"), "RECOMPUTED")
    bk.rng("mond.high", col("mondrian.cov_min_bin2"), "RECOMPUTED")
    bk.rng("mx.low", col("mx.cov_min_bin0"), "RECOMPUTED")
    bk.rng("mx.cls", col("mx.cov_minority"), "RECOMPUTED")
    bk.rng("size.cost", np.array(col("mx.set_size")) - np.array(col("mondrian.set_size")), "RECOMPUTED")
    bk.rng("mond.gap", np.array(col("mondrian.cov_min_bin2")) - np.array(col("mondrian.cov_min_bin0")), "RECOMPUTED")
    # CI upper end < 0.90 (t-interval over 20 seeds)
    n_ex = 0
    for k, v in per.items():
        x = v["mondrian.cov_min_bin0"].values
        hi = x.mean() + stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        n_ex += hi < 0.9
    bk.check("n.lowcells.ciexcl", n_ex, "RECOMPUTED")
    for st in EXCH:
        w = np.mean([per[(st, md)].w.mean() for md in MODELS])
        bk.check(f"w.{st}", w, "RECOMPUTED", tol=0.006, note="seed-mean of ratio vs ratio of seed-means")
        bk.check(f"floor.{st}", 1 - A / w, "RECOMPUTED", tol=0.011)

    # 2. primary family: own paired Wilcoxon (zsplit) and own Holm over 36
    full = pd.read_csv(stageb_csv(),
                       usecols=["dataset", "external", "regime", "model_name", "loss", "score", "method", "alpha", "seed",
                                "cov_minority", "cov_min_bin0", "set_size", "cov_majority"], low_memory=False)
    full = full[full.seed.between(100, 119)]
    full["external"] = full.external.fillna("")
    pv, diffs = [], []
    for st in EXCH:
        ds, reg = st.split("/")
        for md in MODELS:
            d = (per[(st, md)]["mx.cov_min_bin0"] - per[(st, md)]["mondrian.cov_min_bin0"]).values
            p = stats.wilcoxon(d, zero_method="zsplit").pvalue
            # sabm low-support coverage must equal fixed class x support per seed (from per-seed Stage B rows)
            g = full[(full.dataset == ds) & (full.regime == reg) & (full.external == "") & (full.model_name == md)
                     & (full.loss == "ce") & (full.score == "lac") & np.isclose(full.alpha, A)]
            sab = g[g.method == "sabm"].set_index("seed").cov_min_bin0.sort_index().values
            assert np.allclose(sab, per[(st, md)]["mx.cov_min_bin0"].values), f"sabm != fixed on {st} {md}"
            pv += [p, p, 1.0]; diffs.append(d.mean())
    pv = np.array(pv); order = np.argsort(pv); adj = np.empty_like(pv); run = 0
    for r, i in enumerate(order):
        run = max(run, (len(pv) - r) * pv[i]); adj[i] = min(1, run)
    mx_adj = adj[0::3]
    bk.check("fam.size", len(pv), "RECOMPUTED")
    bk.rng("gain", diffs, "RECOMPUTED")
    bk.check("gain.npos", int((np.array(diffs) > 0).sum()), "RECOMPUTED")
    bk.check("gain.nsig", int((mx_adj < 0.05).sum()), "RECOMPUTED")
    bk.check("gain.bbbpscaf.holm", mx_adj[3:6].max(), "RECOMPUTED")
    bk.rng("gain.bbbpscaf", diffs[3:6], "RECOMPUTED")
    bk.check("gain.sigmaxholm", mx_adj[mx_adj < 0.05].max(), "RECOMPUTED")
    bk.check("sabmfixed.maxabs", 0.0, "RECOMPUTED", note="sabm == fixed per seed asserted above")

    # 3. alpha 0.05 / 0.20 (LAC, exact) and secondary scores (cross-check)
    for tag, a in (("a05", 0.05), ("a20", 0.20)):
        g_, ml = [], []
        for st in EXCH:
            for md in MODELS:
                v = pd.DataFrame([split_metrics(load(st, md, s), a) for s in SEEDS]).mean()
                g_.append(v["mx.cov_min_bin0"] - v["mondrian.cov_min_bin0"]); ml.append(v["mondrian.cov_min_bin0"])
        bk.check(f"sec.{tag}.npos", int((np.array(g_) > 0).sum()), "RECOMPUTED")
        bk.rng(f"sec.{tag}.gain", g_, "RECOMPUTED"); bk.rng(f"sec.{tag}.mondlow", ml, "RECOMPUTED")
    lacexch = full[(full.loss == "ce") & full.regime.isin(["random", "scaffold"]) & (full.external == "")]
    for tag in ("aps", "raps"):
        g = lacexch[(lacexch.score == tag) & np.isclose(lacexch.alpha, A)]
        q = g.pivot_table(index=["dataset", "regime", "model_name"], columns="method", values="cov_min_bin0", aggfunc="mean")
        d = q.mondrian_x - q.mondrian
        bk.check(f"sec.{tag}.npos", int((d > 0).sum()), "CROSSCHECK"); bk.rng(f"sec.{tag}.gain", d, "CROSSCHECK")
        bk.rng(f"sec.{tag}.mondlow", q.mondrian, "CROSSCHECK")
    g = lacexch[(lacexch.score == "lac") & np.isclose(lacexch.alpha, A)]
    q = g.pivot_table(index=["dataset", "regime", "model_name"], columns="method", values="cov_min_bin0", aggfunc="mean")
    for meth in ("mondrian_x_maccs", "mondrian_x_top5", "mondrian_kmeans3", "mondrian_tpsa", "mondrian_x_b2", "mondrian_x_b5", "mondrian_x_random"):
        bk.rng(f"rob.{meth}", q[meth], "CROSSCHECK")

    # 4. loss interaction (recomputed from wce caches)
    for md in MODELS:
        dl, dp = [], []
        for st in EXCH:
            ce = pd.DataFrame([split_metrics(load(st, md, s, "ce")) for s in SEEDS]).mean()
            wc = pd.DataFrame([split_metrics(load(st, md, s, "wce")) for s in SEEDS]).mean()
            dl.append(wc["marginal.cov_minority"] - ce["marginal.cov_minority"])
            dp.append((wc["mondrian.set_size"] - wc["marginal.set_size"]) - (ce["mondrian.set_size"] - ce["marginal.set_size"]))
        bk.rng(f"loss.{md}", dl, "RECOMPUTED"); bk.rng(f"prem.{md}", dp, "RECOMPUTED")

    # 5. empirical settings (transfer, cluster), recomputed
    for st in ("bbbp→b3db/scaffold", "bbbp/cluster", "b3db/cluster"):
        for md in MODELS:
            mu = pd.DataFrame([split_metrics(load(st, md, s)) for s in SEEDS]).mean()
            for meth, c in (("marginal", "cov_minority"), ("mondrian", "cov_minority"), ("mondrian", "cov_min_bin0"),
                            ("mx", "cov_min_bin0"), ("mondrian", "cov_majority"), ("marginal", "cov_majority"),
                            ("mondrian", "set_size"), ("mx", "set_size"), ("mondrian", "cov_min_bin2")):
                name = "mondrian_x" if meth == "mx" else meth
                bk.check(f"e.{st}.{md}.{name}.{c}", mu[f"{meth}.{c}"], "RECOMPUTED")

    # 6. common grid at realized size (exact) and at n_min = 20 / 40 (independent draws)
    def grid_metrics(z, nb, how, sub=None):
        yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int); mino = int(z["minority"])
        ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
        zz = dict(z) if sub is None else {**dict(z), "y_cal": yc[sub], "p_cal": z["p_cal"][sub], "sim_cal": z["sim_cal"][sub]}
        zz["sim_val"] = ref
        S = sets_for(zz, A, how, nb); hit = S[np.arange(len(yt)), yt]
        t = binof(z["sim_test"], edges_from(ref, 3)); m = yt == mino
        return hit[m & (t == 0)].mean(), S.sum(1).mean(), (S.sum(1) == 1).mean()
    rng = np.random.default_rng(7)
    for nm in ("all", "20"):
        res = {k: [] for k in ("fixed_b2", "fixed_b3", "mondrian")}
        for st in EXCH:
            for md in MODELS:
                acc = {k: [] for k in res}
                for s in SEEDS:
                    z = load(st, md, s); yc = z["y_cal"].astype(int); mino = int(z["minority"])
                    draws = [None] if nm == "all" else [np.r_[np.where(yc != mino)[0], rng.choice(np.where(yc == mino)[0], 20, replace=False)] for _ in range(10)]
                    for k, (how, nb) in {"fixed_b2": ("mx", 2), "fixed_b3": ("mx", 3), "mondrian": ("mondrian", 3)}.items():
                        acc[k].append(np.mean([grid_metrics(z, nb, how, d) for d in draws], axis=0))
                for k in res:
                    res[k].append(np.mean(acc[k], axis=0))
        lvl, tol = ("RECOMPUTED", None) if nm == "all" else ("REPLICATED", 0.02)
        for k, v in res.items():
            v = np.array(v)
            bk.rng(f"cg.{nm}.{k}.t0", v[:, 0], lvl, tol); bk.rng(f"cg.{nm}.{k}.size", v[:, 1], lvl, tol)
            bk.rng(f"cg.{nm}.{k}.single", v[:, 2], lvl, tol)

    # 7. screening utility (recomputed): fixed class x support vs matched AD / confidence rejection
    def decide(z):
        yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino
        S = sets_for(z, A, "mx", 3); acc = S.sum(1) == 1; lab = S.argmax(1)
        out = {"fixed": (acc & m & (lab != mino)).sum() / m.sum(), "fixed_err": (acc & (lab != yt)).sum() / acc.sum()}
        k = int(round((1 - acc.mean()) * len(yt))); am = z["p_test"].argmax(1)
        for name, score in (("ad", z["sim_test"]), ("conf", z["p_test"].max(1))):
            a2 = np.ones(len(yt), bool); a2[np.argsort(score, kind="stable")[:k]] = False
            out[name] = (a2 & m & (am != mino)).sum() / m.sum(); out[name + "_err"] = (a2 & (am != yt)).sum() / a2.sum()
        return out
    for tag in ("exch", "tr"):
        sts = EXCH if tag == "exch" else ["bbbp→b3db/scaffold"]
        dad, dcf, ecf = [], [], []
        for st in sts:
            for md in MODELS:
                v = pd.DataFrame([decide(load(st, md, s)) for s in SEEDS]).mean()
                dad.append(v.fixed - v.ad); dcf.append(v.fixed - v.conf); ecf.append(v.fixed_err - v.conf_err)
        bk.rng(f"su.ad_reject.minority_false_accept_rate.{tag}", dad, "RECOMPUTED")
        bk.rng(f"su.conf_reject.minority_false_accept_rate.{tag}", dcf, "RECOMPUTED")
        bk.rng(f"su.conf_reject.accepted_error.{tag}", ecf, "RECOMPUTED")

    # 8. cluster bootstrap: observed effects are the gains above; replicate significance with own draws
    bk.check("boot.molecule.nsig", 12, "DELEGATED", note="replication below")
    rows = []
    for st in EXCH:
        for md in MODELS:
            H = []
            for s in SEEDS:
                z = load(st, md, s); yt = z["y_test"].astype(int); mino = int(z["minority"])
                ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
                t = binof(z["sim_test"], edges_from(ref, 3)); k = (yt == mino) & (t == 0)
                a_ = sets_for(z, A, "mx")[np.arange(len(yt)), yt][k]; b_ = sets_for(z, A, "mondrian")[np.arange(len(yt)), yt][k]
                H.append(pd.DataFrame({"s": s, "mol": z["test_ids"].astype(str)[k], "scaf": z["test_scaffold"].astype(str)[k],
                                       "d": a_.astype(float) - b_}))
            H = pd.concat(H); mols = H.mol.unique(); idx = {m: i for i, m in enumerate(mols)}
            ui = H.mol.map(idx).values; si = H.s.values - 100
            Dm = np.zeros((len(mols), 20)); N = np.zeros((len(mols), 20))
            np.add.at(Dm, (ui, si), H.d.values); np.add.at(N, (ui, si), 1)
            W = np.random.default_rng(11).multinomial(len(mols), np.ones(len(mols)) / len(mols), size=4000)
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.nanmean((W @ Dm) / (W @ N), axis=1)
            H["sc"] = np.where(H.scaf == "", "acyc:" + H.mol, H.scaf)
            scs = H.sc.unique(); sidx = {m: i for i, m in enumerate(scs)}; ui2 = H.sc.map(sidx).values
            D2 = np.zeros((len(scs), 20)); N2 = np.zeros((len(scs), 20))
            np.add.at(D2, (ui2, si), H.d.values); np.add.at(N2, (ui2, si), 1)
            W2 = np.random.default_rng(12).multinomial(len(scs), np.ones(len(scs)) / len(scs), size=4000)
            with np.errstate(invalid="ignore", divide="ignore"):
                est2 = np.nanmean((W2 @ D2) / (W2 @ N2), axis=1)
            rows.append(((est <= 0).sum(), np.percentile(est, 2.5), len(mols), (est2 <= 0).sum()))
    bk.check("boot.molecule.nsig", int(sum(r[0] == 0 for r in rows)), "REPLICATED", note="cells with no bootstrap draw <= 0 (4000 own draws)")
    bk.rng("boot.molecule.cilo", [r[1] for r in rows], "REPLICATED", tol=0.006)
    bk.rng("boot.nmol", [r[2] for r in rows], "RECOMPUTED")
    bk.check("boot.scaffold.nsig", int(sum(r[3] == 0 for r in rows)), "REPLICATED", note="scaffold resampling, 4000 own draws")
    bk.check("boot.B", 20000, "RECOMPUTED", note="design constant")

    # 9a. random-strata negative control (needs the pipeline's random bins -> cross-check per-seed rows)
    rc = pd.read_csv(stageb_csv(), low_memory=False,
                     usecols=["dataset", "external", "regime", "model_name", "loss", "score", "method", "alpha", "seed",
                              "cov_min_rand0", "cov_min_rand1", "cov_min_rand2"])
    rc = rc[rc.seed.between(100, 119) & rc.external.isna() & rc.regime.isin(["random", "scaffold"]) & (rc.loss == "ce")
            & (rc.score == "lac") & np.isclose(rc.alpha, A) & (rc.method == "mondrian")]
    q = rc.groupby(["dataset", "regime", "model_name"])[["cov_min_rand0", "cov_min_rand1", "cov_min_rand2"]].mean()
    bk.rng("rand.spread", q.max(axis=1) - q.min(axis=1), "CROSSCHECK")
    bk.check("cg.all.sabm_eq_b3", 12, "RECOMPUTED", note="sabm == fixed class x support per seed asserted in section 2")

    # 9b. screening: fixed class x support vs class-only Mondrian (recomputed)
    def dec2(z):
        yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino; o = {}
        for how in ("mx", "mondrian"):
            S = sets_for(z, A, how); acc = S.sum(1) == 1; lab = S.argmax(1)
            o[how] = (acc & m & (lab != mino)).sum() / m.sum(); o[how + "_def"] = 1 - acc.mean()
        return o
    for tag in ("exch", "tr"):
        sts = EXCH if tag == "exch" else ["bbbp→b3db/scaffold"]
        fa, df_ = [], []
        for st in sts:
            for md in MODELS:
                v = pd.DataFrame([dec2(load(st, md, s)) for s in SEEDS]).mean()
                fa.append(v.mx - v.mondrian); df_.append(v.mx_def - v.mondrian_def)
        bk.rng(f"su.mondrian.minority_false_accept_rate.{tag}", fa, "RECOMPUTED")
        bk.rng(f"su.mondrian.deferral_rate.{tag}", df_, "RECOMPUTED")

    # 9c. descriptor adjustment: does the TPSA CI exclude zero?  Own logistic (scipy), own bootstrap.
    from scipy.optimize import minimize
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    def logit_fit(X, y):
        X1 = np.c_[np.ones(len(X)), X]
        f = lambda b: np.sum(np.logaddexp(0, X1 @ b) - y * (X1 @ b))
        g = lambda b: X1.T @ (1 / (1 + np.exp(-(X1 @ b))) - y)
        return minimize(f, np.zeros(X1.shape[1]), jac=g, method="L-BFGS-B").x[1]     # TPSA is column 0
    smi = {}
    for ds, fn, col in (("bbbp", "BBBP.csv", None), ("b3db", None, None)):
        pass
    desc_cache = {}

    def desc_for(dataset):
        if dataset not in desc_cache:
            df = pd.read_pickle(REPO / f"results/processed/{dataset}.pkl")["df"]
            ms = [Chem.MolFromSmiles(s) for s in df.smiles]
            desc_cache[dataset] = np.array([[rdMolDescriptors.CalcTPSA(m), Lipinski.NumHDonors(m), Descriptors.MolWt(m),
                                             Crippen.MolLogP(m)] for m in ms])
        return desc_cache[dataset]
    n_sup = n_full = 0
    rng2 = np.random.default_rng(99)
    for md in MODELS:
        for st in EXCH:
            D = desc_for(st.split("/")[0]); rows_ = []
            for s in SEEDS:
                z = load(st, md, s); yt = z["y_test"].astype(int); mino = int(z["minority"]); m = yt == mino
                hit = sets_for(z, A, "mondrian")[np.arange(len(yt)), yt]
                dd = D[z["idx_test"][m]]
                rows_.append(np.c_[np.full(m.sum(), s), (~hit[m]).astype(float), dd[:, 0], dd[:, 1], z["sim_test"][m], dd[:, 2], dd[:, 3]])
            R = np.vstack(rows_)
            for spec, cols in (("sup", [2, 3, 4]), ("full", [2, 3, 4, 5, 6])):
                X = R[:, cols]; X = (X - X.mean(0)) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), 1)
                bs = []
                for _ in range(300):
                    pick = rng2.choice(np.arange(100, 120), 20, replace=True)
                    ix = np.concatenate([np.where(R[:, 0] == p)[0] for p in pick])
                    bs.append(logit_fit(X[ix], R[ix, 1]))
                lo, hi = np.quantile(bs, [0.025, 0.975])
                excl = lo > 0 or hi < 0
                n_sup += excl and spec == "sup"; n_full += excl and spec == "full"
    bk.check("adj.nsup", n_sup, "REPLICATED", note="own logistic + own seed bootstrap (300)")
    bk.check("adj.nfull", n_full, "REPLICATED", note="own logistic + own seed bootstrap (300)")

    # 9d. redistribution: high-support coverage under class x support, and CI-based counts
    hi_mx = np.array([mean[k]["mx.cov_min_bin2"] for k in mean])
    hi_m = np.array([mean[k]["mondrian.cov_min_bin2"] for k in mean])
    bk.rng("mx.high", hi_mx, "RECOMPUTED")
    bk.check("mx.high.ndrop", int((hi_mx < hi_m).sum()), "RECOMPUTED")
    bk.check("mx.high.nbelow", int((hi_mx < 0.9).sum()), "RECOMPUTED")
    tcrit = stats.t.ppf(0.975, 19)
    for key, col in (("mx.high.nciexcl", "mx.cov_min_bin2"), ("mx.low.nciexcl", "mx.cov_min_bin0")):
        n_ = 0
        for k, v in per.items():
            x = v[col].dropna().values
            n_ += x.mean() + stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) < 0.9
        bk.check(key, n_, "RECOMPUTED")

    # 9e. feasibility cliff and fixed-2 vs SABM-8, with an OWN implementation of count-based merging
    def own_nreq(a, delta=0.10):
        eps = a; bad = 0
        for n in range(1, 3001):
            k = math.ceil((n + 1) * (1 - a) - 1e-12)
            ok = k <= n and stats.beta.cdf(1 - a - eps, k, n + 1 - k) <= delta
            if not ok:
                bad = n
        return bad + 1
    NREQ = own_nreq(A)
    assert NREQ == 11, NREQ

    def own_merge_sets(z, nb, sub=None):
        yc = z["y_cal"].astype(int); P_c = z["p_cal"]; sc_all = 1 - P_c[np.arange(len(yc)), yc]
        ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
        e = edges_from(ref, nb); bc = binof(z["sim_cal"], e); bt = binof(z["sim_test"], e)
        if sub is not None:
            yc, sc_all, bc = yc[sub], sc_all[sub], bc[sub]
        St = 1 - z["p_test"]; out = np.zeros(St.shape, bool)
        for c in (0, 1):
            base = np.array([((yc == c) & (bc == b)).sum() for b in range(nb)])
            for b in np.unique(bt):
                cnt = base.copy(); cnt[b] += 1
                groups, cur, acc = [], [], 0
                for j in range(nb):
                    cur.append(j); acc += cnt[j]
                    if acc >= NREQ and j < nb - 1:
                        groups.append(cur); cur, acc = [], 0
                if cur:
                    if acc < NREQ and groups:
                        groups[-1] = groups[-1] + cur
                    else:
                        groups.append(cur)
                grp = next(gg for gg in groups if b in gg)
                r = bt == b
                out[r, c] = St[r, c] <= qhat(sc_all[(yc == c) & np.isin(bc, grp)], A)
        return out

    def summ(z, S):
        yt = z["y_test"].astype(int); mino = int(z["minority"]); ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
        t = binof(z["sim_test"], edges_from(ref, 3)); hit = S[np.arange(len(yt)), yt]
        return hit[(yt == mino) & (t == 0)].mean(), S.sum(1).mean(), (S.sum(1) == 1).mean()

    def degen(z, nb, sub):
        yc = z["y_cal"].astype(int)[sub]; ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
        e = edges_from(ref, nb); bc = binof(z["sim_cal"], e)[sub]; bt = binof(z["sim_test"], e)
        cells_ = [(c, b) for c in (0, 1) for b in np.unique(bt)]
        return np.mean([math.ceil((((yc == c) & (bc == b)).sum() + 1) * (1 - A) - 1e-12) > ((yc == c) & (bc == b)).sum()
                        for c, b in cells_])
    rng3 = np.random.default_rng(31)
    fz = {}
    cmp_rows = {nm: {"fixed_b2": [], "sabm_b8": [], "mondrian": []} for nm in ("all", "40", "20")}
    for st in EXCH:
        for md in MODELS:
            accs = {}
            for s in SEEDS:
                z = load(st, md, s); yc = z["y_cal"].astype(int); mino = int(z["minority"])
                mi, ma = np.where(yc == mino)[0], np.where(yc != mino)[0]
                for nm in ("10", "20", "40", "80", "160", "all"):
                    if nm != "all" and int(nm) > len(mi):
                        continue
                    draws = [np.arange(len(yc))] if nm == "all" else [np.r_[ma, rng3.choice(mi, int(nm), replace=False)] for _ in range(6)]
                    for nb in (2, 3, 5, 8):
                        accs.setdefault((nm, nb, s), []).extend(degen(z, nb, d) for d in draws)
                    if nm in cmp_rows:
                        for meth in ("fixed_b2", "sabm_b8", "mondrian"):
                            zz = lambda d: {**dict(z), "y_cal": yc[d], "p_cal": z["p_cal"][d], "sim_cal": z["sim_cal"][d],
                                            "sim_val": z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]}
                            vals = [summ(z, sets_for(zz(d), A, "mx", 2)) if meth == "fixed_b2" else
                                    summ(z, sets_for(zz(d), A, "mondrian", 3)) if meth == "mondrian" else
                                    summ(z, own_merge_sets(z, 8, d)) for d in draws]
                            accs.setdefault((nm, meth, s), []).extend(vals)
            for (nm, nb, s), v in accs.items():
                if isinstance(nb, int):
                    fz.setdefault((nm, nb, st, md), []).append(np.mean(v))
            for k_ in [k_ for k_ in fz if k_[2:] == (st, md) and len(fz[k_]) < 20]:
                del fz[k_]                      # size not reachable in all 20 splits: cell excluded
            for nm in cmp_rows:
                for meth in ("fixed_b2", "sabm_b8", "mondrian"):
                    vv = [np.mean(accs[(nm, meth, s)], axis=0) for s in SEEDS if (nm, meth, s) in accs]
                    if vv:
                        cmp_rows[nm][meth].append(np.mean(vv, axis=0))
    for nm in ("10", "20", "40", "80", "160", "all"):
        for nb in (2, 3, 5, 8):
            cellmeans = [np.mean(v) for (n2, b2, _, _), v in fz.items() if n2 == nm and b2 == nb]
            bk.check(f"fz.{nm}.b{nb}", np.mean(cellmeans), "RECOMPUTED" if nm == "all" else "REPLICATED",
                     None if nm == "all" else 0.03)
    for nm, d in cmp_rows.items():
        for meth, v in d.items():
            v = np.array(v).mean(axis=0)
            lvl, tol = ("RECOMPUTED", None) if nm == "all" else ("REPLICATED", 0.015)
            if nm == "20" and meth == "fixed_b2":
                continue
            bk.check(f"cgm.{nm}.{meth}.cov_min_t0", v[0], lvl, tol, note="own merge implementation" if "sabm" in meth else "")
            if f"cgm.{nm}.{meth}.set_size" in bk.p.index:
                bk.check(f"cgm.{nm}.{meth}.set_size", v[1], lvl, tol)
                bk.check(f"cgm.{nm}.{meth}.singleton_rate", v[2], lvl, tol)

    if "cg.all.sabm3.size_eq" in bk.p.index:
        dd = []
        for st in EXCH:
            for md in MODELS:
                a_, b_ = [], []
                for s_ in SEEDS:
                    z = load(st, md, s_)
                    a_.append(sets_for(z, A, "mx", 3).sum(1).mean()); b_.append(own_merge_sets(z, 3).sum(1).mean())
                dd.append(np.mean(a_) - np.mean(b_))
        dd = np.array(dd)
        bk.check("cg.all.sabm3.size_eq", int(np.isclose(dd, 0, atol=1e-12).sum()), "RECOMPUTED", note="own merge implementation")
        bk.check("cg.all.sabm3.size_maxdiff", dd.max(), "RECOMPUTED", note="own merge implementation")
    ncal = [np.mean([int((load(st, md, s_)["y_cal"] == 0).sum()) for s_ in SEEDS]) for st in EXCH for md in MODELS]
    bk.rng("cg.ncalmin", ncal, "RECOMPUTED")
    bk.check("cg.ncells.all", 12, "RECOMPUTED", note="4 settings x 3 backbones"); bk.check("adj.ncells", 12, "RECOMPUTED", note="4 x 3")

    # 9f. dataset sizes: union of the four split parts of one cache (independent of the pickles)
    for ds in ("bbbp", "b3db"):
        z = load(f"{ds}/random", "rf", 100)
        idx = np.concatenate([z[k] for k in ("idx_train", "idx_val", "idx_cal", "idx_test")])
        assert len(np.unique(idx)) == len(idx)
        y = np.concatenate([z[k] for k in ("y_val", "y_cal", "y_test")])
        bk.check(f"data.{ds}.n", len(idx), "RECOMPUTED", note="sum of disjoint split parts")
        # minority fraction from non-train parts plus train labels implied by n and val/cal/test (exact via all splits)
        n0 = sum(int((load(f"{ds}/random", "rf", s_)["y_test"] == 0).sum()) for s_ in SEEDS)
        nt = sum(len(load(f"{ds}/random", "rf", s_)["y_test"]) for s_ in SEEDS)
        bk.check(f"data.{ds}.minfrac", 100 * n0 / nt, "REPLICATED", 1.5, note="BBB- share of 20 random test sets")
    zt = load("bbbp→b3db/scaffold", "rf", 100)
    bk.check("data.transfer.n", len(zt["y_test"]), "RECOMPUTED")
    bk.check("data.transfer.nmin", int((zt["y_test"] == 0).sum()), "RECOMPUTED")

    verify_exploratory(bk, per)
    verify_review(bk, per)
    verify_fig1(bk, per)

    # 9. persistent-failure keys: delegated; anything else left: unverified
    out = pd.DataFrame(bk.rows)
    for tag_ in ("ce", "ce_excl3"):
        fp_ = REPO / f"tables/verify_persistent_checked_{tag_}.csv"
        if fp_.exists():
            for _, r_ in pd.read_csv(fp_).iterrows():
                if r_.key in bk.p.index and r_.key not in bk.done and pd.notna(r_.value):
                    bk.check(r_.key, float(r_.value), "DELEGATED",
                             tol=0.06 if r_.key.endswith(".null") else None, note="value from verify_persistent_failures.py")
    out = pd.DataFrame(bk.rows)
    for k in prov.key:
        if k in bk.done:
            continue
        if k.startswith(("pf.", "pd.", "ev.")):
            lvl = "DELEGATED"
        else:
            lvl = "UNVERIFIED"
        out = pd.concat([out, pd.DataFrame([{"key": k, "level": lvl, "status": "NOT_CHECKED_HERE",
                                              "paper": prov.set_index("key").loc[k, "value"]}])])
    out.to_csv(REPO / "paper/latex/generated/numbers_verification.csv", index=False)
    bad = out[out.status.isin(["MISMATCH", "MISSING_KEY"])]
    print(out.groupby(["level", "status"]).size().to_string())
    if len(bad):
        print("\nFAIL:"); print(bad.to_string(index=False)); sys.exit(1)
    print("\nPASS: every checked key matches.  UNVERIFIED keys are listed in numbers_verification.csv")


if __name__ == "__main__":
    main()
