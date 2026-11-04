"""Stage B: conformal evaluation on cached predictions (no refitting).

v1.3 (2026-10-21) adds, without changing any v1.2 method or number:
  * SOTA conformal baselines (conformal/sota.py)
  * ablation calibrators with alternative taxonomies (conformal/ablations.py)
  * probability variants: post-hoc logit adjustment (_la) and temperature scaling (_ts)
  * the margin score
  * extra metrics: Tursunbadalov MCG/CII/acceptance, size-stratified coverage, worst-slab
    coverage, minority coverage by TPSA bin and by random (negative-control) bin
  * evaluate_selective(): non-conformal baselines and conformal selection
Everything that needs fingerprints or SMILES (Tanimoto kernels, MACCS, descriptors, KMM) is
computed from `meta` supplied by scripts/run_grid.py; methods whose inputs are missing are
skipped, so old callers keep working.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.conformal import calibrators as C
from bbbcp.conformal.ablations import ABLATIONS
from bbbcp.conformal.scores import score_matrix
from bbbcp.conformal.sota import SCORE_FREE, SOTA, bbse_weights, kde_weights, kmm_weights, skmm_select
from bbbcp.metrics.conditional import size_stratified, tursunbadalov_metrics, worst_slab_coverage
from bbbcp.metrics.metrics import screening_cost, set_metrics

SCORES = ("lac", "aps", "raps", "margin")
ALPHAS = (0.05, 0.10, 0.20)
CORE = ("marginal", "mondrian", "mondrian_pac", "mondrian_x", "sabm", "sabm_abs05", "weighted",
        "weighted_mondrian")
METHOD_SETS = {
    "core": CORE,
    "sota": tuple(SOTA),
    "ablation": tuple(ABLATIONS),
    "all": CORE + tuple(SOTA) + tuple(ABLATIONS),
    "v12": CORE,                                 # exactly the v1.2 method list
}
PROB_VARIANT_METHODS = ("marginal", "mondrian", "mondrian_x", "sabm")
RAPS_KW = {"lam": 0.1, "k_reg": 1}  # fixed in advance (not tuned on cal/test)
COSTS = [(1, 1, 0.3), (1, 2, 0.3), (1, 5, 0.3), (2, 1, 0.3), (1, 2, 0.1), (1, 2, 0.6)]
HEAVY = {"lcp_tanimoto", "lcp_knn", "rlcp_sim", "kmm_weighted", "kmm_mondrian", "skmm_mondrian", "kde_weighted",
         "condqr_sim", "kandinsky_qr", "weighted", "weighted_mondrian", "labelshift_bbse"}
TPSA_EDGES = (60.0, 90.0)


def sim_bins(sim_ref, sim_x, n_bins=3):
    """Support bins with edges fixed from a REFERENCE sample (VAL by default; independent of
    cal/test).  If the reference is empty (no VAL split), the caller must pass CAL x-values."""
    edges = np.quantile(sim_ref, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(sim_x, edges), edges


def _scores(p_cal, y_cal, p_test, kind, rng):
    u_cal = rng.uniform(size=len(p_cal)); u_test = rng.uniform(size=len(p_test))
    kw = RAPS_KW if kind == "raps" else {}
    Sc = score_matrix(p_cal, kind, u_cal, **kw)
    St = score_matrix(p_test, kind, u_test, **kw)
    return Sc[np.arange(len(y_cal)), y_cal], St


def naive_sets(p_test, alpha):
    """Non-conformal baseline: add labels by decreasing probability until the cumulative
    model probability reaches 1 - alpha (trusts the model's probabilities)."""
    above = score_matrix(p_test, "aps_det") - p_test
    return above < 1 - alpha


def _entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def _prior(z, K=2):
    y = z["y_val"].astype(int) if len(z["y_val"]) else z["y_cal"].astype(int)
    c = np.bincount(y, minlength=K).astype(float) + 1.0
    return c / c.sum()


def _logit_adjust(P, prior):
    Q = P / prior[None, :]
    return Q / Q.sum(1, keepdims=True)


def build_context(z, meta=None, seed=0, minority=0, heavy=True):
    """Everything the calibrators may need, computed WITHOUT test labels."""
    meta = meta or {}
    rng = np.random.default_rng(seed)
    y_cal = z["y_cal"].astype(int)
    has_val = len(z["sim_val"]) > 0
    ref = z["sim_val"] if has_val else z["sim_cal"]
    ctx = {"P_cal": z["p_cal"], "P_test": z["p_test"], "sim_cal": z["sim_cal"], "sim_test": z["sim_test"],
           "prior": _prior(z), "rng": rng}
    ctx["bin_cal"], edges = sim_bins(ref, z["sim_cal"])
    ctx["bin_test"], _ = sim_bins(ref, z["sim_test"])
    ctx["_edges"] = edges
    for k in (2, 5):
        ctx[f"bins_b{k}_cal"], _ = sim_bins(ref, z["sim_cal"], k)
        ctx[f"bins_b{k}_test"], _ = sim_bins(ref, z["sim_test"], k)
    ctx["bins_caledges_cal"], _ = sim_bins(z["sim_cal"], z["sim_cal"])
    ctx["bins_caledges_test"], _ = sim_bins(z["sim_cal"], z["sim_test"])
    # entropy and predicted-label taxonomies (functions of x through the fixed model)
    if has_val and "p_val" in z.files:
        e_ref = _entropy(z["p_val"])
    else:
        e_ref = _entropy(z["p_cal"])
    ctx["bins_entropy_cal"], _ = sim_bins(e_ref, _entropy(z["p_cal"]))
    ctx["bins_entropy_test"], _ = sim_bins(e_ref, _entropy(z["p_test"]))
    ctx["bins_predlabel_cal"] = z["p_cal"].argmax(1)
    ctx["bins_predlabel_test"] = z["p_test"].argmax(1)
    ctx["bins_random_cal"] = rng.integers(0, 3, len(y_cal))
    ctx["bins_random_test"] = rng.integers(0, 3, len(z["y_test"]))
    # descriptors -> TPSA taxonomy and slab features
    if "desc_cal" in meta:
        tp_c, tp_t = meta["desc_cal"][:, 2], meta["desc_test"][:, 2]
        ctx["tpsa_bin_cal"] = np.digitize(tp_c, TPSA_EDGES)
        ctx["tpsa_bin_test"] = np.digitize(tp_t, TPSA_EDGES)
        mu, sd = meta["desc_cal"].mean(0), meta["desc_cal"].std(0) + 1e-9
        ctx["_F_test"] = (meta["desc_test"] - mu) / sd
    # fingerprint-based support variants and kernels
    if "X_cal" in meta and "X_train" in meta:
        from bbbcp.chemistry.fingerprints import support_measures, tanimoto_matrix
        Xtr = meta["X_train"]
        _, t5c = support_measures(meta["X_cal"], Xtr)
        _, t5t = support_measures(meta["X_test"], Xtr)
        t5v = support_measures(meta["X_val"], Xtr)[1] if has_val and "X_val" in meta else t5c
        ctx["bins_top5_cal"], _ = sim_bins(t5v, t5c)
        ctx["bins_top5_test"], _ = sim_bins(t5v, t5t)
        if "M_train" in meta:
            mc = support_measures(meta["M_cal"], meta["M_train"])[0]
            mt = support_measures(meta["M_test"], meta["M_train"])[0]
            mv = support_measures(meta["M_val"], meta["M_train"])[0] if has_val else mc
            ctx["bins_maccs_cal"], _ = sim_bins(mv, mc)
            ctx["bins_maccs_test"], _ = sim_bins(mv, mt)
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=32, random_state=0).fit(Xtr.astype(np.float32))
        km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(svd.transform(Xtr.astype(np.float32)))
        ctx["bins_kmeans3_cal"] = km.predict(svd.transform(meta["X_cal"].astype(np.float32)))
        ctx["bins_kmeans3_test"] = km.predict(svd.transform(meta["X_test"].astype(np.float32)))
        if heavy:
            ctx["Tc_tan"] = tanimoto_matrix(meta["X_test"], meta["X_cal"])
            Zall = np.vstack([meta["X_cal"], meta["X_test"]]).astype(np.float32)
            Z = TruncatedSVD(n_components=16, random_state=0).fit_transform(Zall)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            Zc, Zt = Z[: len(meta["X_cal"])], Z[len(meta["X_cal"]):]
            w, sig = kmm_weights(Zc, Zt)
            ctx["kmm_cal"], ctx["kmm_test"] = w, np.ones(len(Zt))
            keep = skmm_select(Zc, Zt, w, sig)
            w2, _ = kmm_weights(Zc, Zt[keep], sigma=sig)
            ctx["skmm_cal"], ctx["skmm_test"], ctx["skmm_keep"] = w2, np.ones(len(Zt)), keep
            ctx["kde_cal"], ctx["kde_test"] = kde_weights(Zc[:, :8], Zt[:, :8])
            ctx["w_cal"], ctx["w_test"] = C.density_ratio(meta["X_cal"], meta["X_test"])
    elif "X_cal" in meta and heavy:
        ctx["w_cal"], ctx["w_test"] = C.density_ratio(meta["X_cal"], meta["X_test"])
    ctx["bbse_w"] = bbse_weights(z["p_cal"], y_cal, z["p_test"])
    return ctx


def _available(m, ctx):
    need = {"lcp_tanimoto": "Tc_tan", "lcp_knn": "Tc_tan", "kmm_weighted": "kmm_cal", "kmm_mondrian": "kmm_cal",
            "skmm_mondrian": "skmm_cal", "kde_weighted": "kde_cal", "weighted": "w_cal",
            "weighted_mondrian": "w_cal", "kandinsky_qr": "tpsa_bin_cal",
            "mondrian_tpsa": "tpsa_bin_cal", "sabm_tpsa": "tpsa_bin_cal",
            "mondrian_x_top5": "bins_top5_cal", "sabm_top5": "bins_top5_cal",
            "mondrian_x_maccs": "bins_maccs_cal", "mondrian_kmeans3": "bins_kmeans3_cal"}
    return need.get(m) is None or need[m] in ctx


def _row_metrics(sets, y_test, a, minority, ctx, slab):
    r = set_metrics(sets, y_test, minority)
    for cfp, cfn, cd in COSTS:
        r[f"cost_fp{cfp}_fn{cfn}_d{cd}"] = screening_cost(sets, y_test, cfp, cfn, cd)
    r.update(tursunbadalov_metrics(sets, y_test, a, minority))
    r.update(size_stratified(sets, y_test))
    hit = sets[np.arange(len(y_test)), y_test]
    edges = ctx["_edges"]
    for b in range(len(edges) + 1):
        mb = ctx["bin_test"] == b
        r[f"cov_bin{b}"] = hit[mb].mean() if mb.any() else np.nan
        mm = mb & (y_test == minority)
        r[f"cov_min_bin{b}"] = hit[mm].mean() if mm.any() else np.nan
        r[f"n_min_bin{b}"] = int(mm.sum())
        r[f"size_bin{b}"] = sets[mb].sum(1).mean() if mb.any() else np.nan
    for key, tag in (("tpsa_bin_test", "tpsa"), ("bins_random_test", "rand")):
        if key in ctx:
            for b in range(3):
                mm = (ctx[key] == b) & (y_test == minority)
                r[f"cov_min_{tag}{b}"] = hit[mm].mean() if mm.any() else np.nan
                r[f"n_min_{tag}{b}"] = int(mm.sum())
    if slab and "_F_test" in ctx:
        r["worst_slab_cov"] = worst_slab_coverage(ctx["_F_test"], hit)
        r["worst_slab_cov_min"] = worst_slab_coverage(ctx["_F_test"], hit, mask=y_test == minority)
    return r


def evaluate_file(path: Path, methods=METHOD_SETS["v12"], scores=SCORES[:3], alphas=ALPHAS,
                  extra_meta=None, minority=0, prob_variants=("_mcdropout",), score_free=False,
                  slab=False, heavy=True):
    """Defaults reproduce v1.2 exactly (same methods, scores, variants).  scripts/run_grid.py
    passes methods=METHOD_SETS['all'], scores=SCORES, prob_variants=('_mcdropout', '_la', '_ts'),
    score_free=True, slab=True for the v1.3 grid."""
    if isinstance(methods, str):
        methods = METHOD_SETS[methods]
    z = np.load(path, allow_pickle=False)
    if "minority" in z.files:
        minority = int(z["minority"])
    y_cal, y_test = z["y_cal"].astype(int), z["y_test"].astype(int)
    seed = zlib.crc32(Path(path).name.encode() + str(Path(path).parent).split('results')[-1].encode())
    ctx = build_context(z, extra_meta, seed=seed % (2 ** 31), minority=minority,
                        heavy=heavy and any(m in HEAVY for m in methods))
    rows = []
    prob_sets = [("", z["p_cal"], z["p_test"])]
    if "_mcdropout" in prob_variants and "p_cal_mc" in z.files:
        prob_sets.append(("_mcdropout", z["p_cal_mc"], z["p_test_mc"]))
    if "_la" in prob_variants:
        pr = ctx["prior"]
        prob_sets.append(("_la", _logit_adjust(z["p_cal"], pr), _logit_adjust(z["p_test"], pr)))
    if "_ts" in prob_variants and len(z["y_val"]):
        from bbbcp.uncertainty.selective import temperature_scale
        pc, _ = temperature_scale(z["p_val"], z["y_val"].astype(int), z["p_cal"])
        pt, Tt = temperature_scale(z["p_val"], z["y_val"].astype(int), z["p_test"])
        prob_sets.append(("_ts", pc, pt))
    for suffix, p_cal, p_test in prob_sets:
        cx = dict(ctx, P_cal=p_cal, P_test=p_test)
        for kind in scores:
            rng = np.random.default_rng(seed + zlib.crc32(kind.encode()) % 1000)
            s_cal, S_test = _scores(p_cal, y_cal, p_test, kind, rng)
            for a in alphas:
                for m in methods:
                    if suffix == "_mcdropout" and m not in ("marginal", "mondrian"):
                        continue
                    if suffix in ("_la", "_ts") and m not in PROB_VARIANT_METHODS:
                        continue
                    if not _available(m, cx):
                        continue
                    fn = C.CALIBRATORS.get(m) or SOTA.get(m) or ABLATIONS[m]
                    T = fn(s_cal, y_cal, S_test, a, **cx)
                    sets = C.predict_sets(S_test, T)
                    yt = y_test
                    keep = getattr(fn, "keep", None) if m == "skmm_mondrian" else None
                    cxx = cx
                    if keep is not None:
                        sets, yt = sets[keep], y_test[keep]
                        cxx = dict(cx, bin_test=cx["bin_test"][keep])
                        for k2 in ("tpsa_bin_test", "bins_random_test"):
                            if k2 in cx:
                                cxx[k2] = cx[k2][keep]
                        if "_F_test" in cx:
                            cxx["_F_test"] = cx["_F_test"][keep]
                    r = {"method": m + suffix, "score": kind, "alpha": a,
                         **_row_metrics(sets, yt, a, minority, cxx,
                                        slab and np.isclose(a, 0.10) and kind in ("lac", "aps"))}
                    if keep is not None:
                        r["keep_frac"] = float(keep.mean())
                    if m.startswith("sabm") and hasattr(C.sabm, "last_level") and m in ("sabm", "sabm_abs05"):
                        r["sabm_mean_bins_pooled"] = float(C.sabm.last_level.mean())
                    rows.append(r)
        if not suffix:
            for a in alphas:
                sets = naive_sets(p_test, a)
                rows.append({"method": "naive_probability", "score": "none", "alpha": a,
                             **_row_metrics(sets, y_test, a, minority, ctx, False)})
            if score_free:
                for name, fn in SCORE_FREE.items():
                    for a in alphas:
                        T = fn(None, y_cal, None, a, **cx)
                        sets = C.predict_sets(fn.S_override, T)
                        rows.append({"method": name, "score": "pas", "alpha": a,
                                     **_row_metrics(sets, y_test, a, minority, ctx,
                                                    slab and np.isclose(a, 0.10))})
    out = pd.DataFrame(rows)
    out["n_cal"] = len(y_cal); out["n_cal_minority"] = int((y_cal == minority).sum())
    out["n_test"] = len(y_test); out["n_test_minority"] = int((y_test == minority).sum())
    out["minority_label"] = minority
    out["bin_edges"] = ";".join(f"{e:.3f}" for e in ctx["_edges"])
    return out


def evaluate_selective(path: Path, extra_meta=None, minority=0, alphas=ALPHAS):
    """Non-conformal baselines (B15-B17, A12 Venn-Abers) and conformal selection (A14)."""
    from bbbcp.uncertainty import selective as S
    z = np.load(path, allow_pickle=False)
    if "minority" in z.files:
        minority = int(z["minority"])
    y_val, y_cal, y_test = (z[k].astype(int) for k in ("y_val", "y_cal", "y_test"))
    p_val, p_cal, p_test = z["p_val"], z["p_cal"], z["p_test"]
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    bin_test, edges = sim_bins(ref, z["sim_test"])
    probs = {"raw": p_test}
    if len(y_val) > 10:
        probs["ts"], _ = S.temperature_scale(p_val, y_val, p_test)
        probs["platt"] = S.platt(p_val, y_val, p_test)
        probs["isotonic"] = S.isotonic(p_val, y_val, p_test)
    va = S.venn_abers(p_cal, y_cal, p_test)
    probs["venn_abers"] = np.column_stack([1 - va[:, 2], va[:, 2]])
    rows = []
    for name, P in probs.items():
        base = {"calibrator": name, "ece": S.ece(P, y_test), "ece_minority": S.classwise_ece(P, y_test, minority)}
        for a in alphas:
            acc = P.max(1) >= 1 - a
            sets = naive_sets(P, a)
            met = set_metrics(sets, y_test, minority)
            rows.append({"family": "confidence_threshold", "rule": f"maxprob>={1 - a:.2f}", "alpha": a, **base,
                         **S.selective_metrics(P, y_test, acc, minority),
                         "naive_cov_min": met["cov_minority"], "naive_set_size": met["set_size"],
                         **{f"acc_low_support": S.selective_metrics(P[bin_test == 0], y_test[bin_test == 0],
                                                                     acc[bin_test == 0], minority)["acc_on_accepted"]}})
        for unc_name, unc in (("1-maxprob", 1 - P.max(1)), ("entropy", _entropy(P))):
            rows.append({"family": "error_detection", "rule": unc_name, "calibrator": name,
                         **S.error_detection(unc, P, y_test)})
    for tname, tau in (("val_tertile1", edges[0]), ("val_median", float(np.quantile(ref, 0.5)))):
        acc = z["sim_test"] >= tau
        rows.append({"family": "ad_threshold", "rule": tname, "tau": tau, "calibrator": "raw",
                     **S.selective_metrics(p_test, y_test, acc, minority)})
    rows.append({"family": "error_detection", "rule": "1-maxTanimoto", "calibrator": "raw",
                 **S.error_detection(1 - z["sim_test"], p_test, y_test)})
    for q in (0.1, 0.2):
        rows.append({"family": "conformal_selection", "rule": f"BH q={q}", "q": q, "calibrator": "raw",
                     **S.conformal_selection(p_cal, y_cal, p_test, y_test, target=1,
                                             q=q, strata=bin_test)})
    return pd.DataFrame(rows)


def support_sweep(path: Path, n_list=(10, 20, 40, 80, 160), R=100, alphas=ALPHAS,
                  scores=("lac", "aps"), methods=("marginal", "mondrian", "mondrian_pac", "sabm"),
                  minority=0, seed=0):
    """Controlled calibration-support experiment: keep every majority calibration point,
    subsample n_min minority calibration points without replacement (R draws each).
    Test set fixed.  Measures how each calibrator behaves as minority support shrinks."""
    z = np.load(path)
    y_cal, y_test = z["y_cal"].astype(int), z["y_test"].astype(int)
    bin_cal_all, _ = sim_bins(z["sim_val"], z["sim_cal"])
    bin_test, _ = sim_bins(z["sim_val"], z["sim_test"])
    min_idx = np.where(y_cal == minority)[0]; maj_idx = np.where(y_cal != minority)[0]
    rng = np.random.default_rng(seed)
    rows = []
    for kind in scores:
        for n_min in n_list:
            if n_min > len(min_idx):
                continue
            for r in range(R):
                sub = np.concatenate([maj_idx, rng.choice(min_idx, n_min, replace=False)])
                s_cal, S_test = _scores(z["p_cal"][sub], y_cal[sub], z["p_test"], kind, rng)
                for a in alphas:
                    for m in methods:
                        T = C.CALIBRATORS[m](s_cal, y_cal[sub], S_test, a, bin_cal=bin_cal_all[sub],
                                             bin_test=bin_test)
                        sets = C.predict_sets(S_test, T)
                        met = set_metrics(sets, y_test, minority)
                        rows.append({"score": kind, "n_min_cal": n_min, "rep": r, "alpha": a,
                                     "method": m, **{k: met[k] for k in (
                                         "coverage", "cov_minority", "cov_majority", "worst_class_cov",
                                         "set_size", "singleton_rate", "singleton_error", "full_rate",
                                         "empty_rate")},
                                     "cost_fp1_fn2_d0.3": screening_cost(sets, y_test, 1, 2, 0.3)})
    return pd.DataFrame(rows)
