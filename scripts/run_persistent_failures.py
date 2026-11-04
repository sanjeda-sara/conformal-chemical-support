"""Cross-architecture persistent-failure analysis (paper Figure 3).  No model is refit.

Question: which BBB- molecules are confidently mis-handled by RF, GIN AND Chemprop across the
20 confirmatory splits (seeds 100-119), and what do they look like chemically?

Everything is rebuilt from the Stage A caches (results/raw/<setting>/<model>_<loss>/seed*.npz):
cached probabilities, labels, split indices, support similarities.  Conformal thresholds are
recomputed with the SAME calibrators Stage B used (LAC, alpha = 0.10).  LAC is deterministic,
so every per-split coverage reproduced here must equal the Stage B CSV exactly; the script
checks that (tables/persistent_failure_verification_<loss>.csv) and stops if it does not.

Definitions (fixed 2026-10-29, written BEFORE looking at the confirmatory per-molecule data)
  miss (primary)   the class-conditional (Mondrian) LAC set at alpha = 0.10 EXCLUDES the true
                   label BBB-, i.e. the set is {BBB+} or empty.  This is the method that
                   restores minority coverage on average, so a molecule it still misses is a
                   failure the paper's headline method does not fix.
  miss (secondary) argmax misclassification; class x support Mondrian (mondrian_x) exclusion;
                   marginal exclusion.  Reported alongside, never substituted for the primary.
  unit             UNIQUE molecule (mol_id) within a setting.  A molecule tested in k of the 20
                   splits contributes ONE row with k appearances, never k independent rows.
  eligible         k >= MIN_APPEAR (default 2) appearances.  Splits are matched across
                   backbones (same seed -> same split), so k is the same for all three.
  persistent       eligible and missed at EVERY appearance by that backbone.
  cross-arch.      persistent for all three backbones (RF, GIN, Chemprop).
  sensitivity      'majority': missed in >= half of appearances by all three backbones.

Inference (exploratory, labelled as such in the paper)
  * concordance: is the all-three overlap larger than if the three backbones failed on
    unrelated molecules?  Null = permute each backbone's persistent indicator independently,
    within strata of k, among eligible molecules.  A large excess says failures are a property
    of the MOLECULE, not of the architecture.
  * characterisation: cross-architecture persistent vs all other eligible BBB- molecules on the
    four pre-specified descriptors TPSA, HBD, MW, cLogP (Mann-Whitney, Holm across descriptors x
    settings), median difference with a Murcko-scaffold cluster bootstrap CI.  Support (mean
    max-Tanimoto to TRAIN), other descriptors and scaffolds are descriptive only.
  * annotation: compound name, and the label/logBB of the same connectivity key in the OTHER
    dataset.  A BBBP BBB- molecule that B3DB calls BBB+ is a label-disagreement candidate, which
    must be reported separately from any biological reading.

usage (from repo root, PYTHONPATH=$PWD/src):
  python scripts/run_persistent_failures.py                      # CE, seeds 100-119
  python scripts/run_persistent_failures.py --loss wce           # sensitivity
  python scripts/run_persistent_failures.py --stageb none        # skip the Stage B check
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from bbbcp.chemistry.fingerprints import DESCRIPTOR_NAMES
from bbbcp.conformal import calibrators as C
from bbbcp.evaluate import sim_bins
from bbbcp.pipeline import RAW, REPO, features, prepare
from bbbcp.stats.stats import holm

# setting label -> (results/raw sub-path, source dataset, target dataset, dataset, external, regime)
SETTINGS = {
    "bbbp/random":         ("bbbp/random", "bbbp", "bbbp", "bbbp", None, "random"),
    "bbbp/scaffold":       ("bbbp/scaffold", "bbbp", "bbbp", "bbbp", None, "scaffold"),
    "b3db/random":         ("b3db/random", "b3db", "b3db", "b3db", None, "random"),
    "b3db/scaffold":       ("b3db/scaffold", "b3db", "b3db", "b3db", None, "scaffold"),
    "bbbp→b3db/scaffold":  ("bbbp__to__b3db/scaffold", "bbbp", "b3db", "bbbp", "b3db", "scaffold"),
}
PRIMARY_SETTINGS = ["bbbp/random", "bbbp/scaffold", "b3db/random", "b3db/scaffold"]
BACKBONES = ["rf", "gin", "chemprop"]
MISS_KINDS = ["mondrian", "mondrian_x", "marginal", "argmax"]
PRIMARY_MISS = "mondrian"
PRIMARY_DESC = ["TPSA", "HBD", "MW", "cLogP"]
OTHER_DESC = ["HBA", "RotB", "AromRings", "FormalChargeAbs", "FracCSP3"]
ALPHA = 0.10
N_PERM = 5000
N_BOOT = 2000


# ------------------------------------------------------------------------------ per split
def split_records(path: Path, alpha: float, exclude=frozenset()):
    """Per-test-molecule hit/miss for one cached split.  Returns (molecule rows, check row)."""
    # allow_pickle: under pandas 3, Stage A stores test_ids / test_scaffold as object arrays.
    # These are our own Stage A files, not downloaded data.
    z = np.load(path, allow_pickle=True)
    minority = int(z["minority"]) if "minority" in z.files else 0
    y_cal, y_test = z["y_cal"].astype(int), z["y_test"].astype(int)
    p_cal, p_test = z["p_cal"], z["p_test"]
    sim_cal, sim_test = z["sim_cal"], z["sim_test"]
    test_ids, test_scaf, idx_test = z["test_ids"].astype(str), z["test_scaffold"].astype(str), z["idx_test"].astype(int)
    if exclude:
        # sensitivity analysis: drop listed molecules from CAL (ids rebuilt from the source dataset prefix)
        # and from TEST, then recompute every threshold
        src = test_ids[0].split(":")[0] if "__to__" not in str(path) else "BBBP"
        cal_ids = np.array([f"{src}:{i}" for i in z["idx_cal"]])
        kc = ~np.isin(cal_ids, list(exclude)); kt = ~np.isin(test_ids, list(exclude))
        y_cal, p_cal, sim_cal = y_cal[kc], p_cal[kc], sim_cal[kc]
        y_test, p_test, sim_test = y_test[kt], p_test[kt], sim_test[kt]
        test_ids, test_scaf, idx_test = test_ids[kt], test_scaf[kt], idx_test[kt]
    s_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]          # LAC, true-label column
    S_test = 1.0 - p_test                                        # LAC, every candidate label
    ref = z["sim_val"] if len(z["sim_val"]) else z["sim_cal"]
    b_cal, edges = sim_bins(ref, sim_cal)
    b_test, _ = sim_bins(ref, sim_test)
    sets, T = {}, {}
    for m in ("marginal", "mondrian", "mondrian_x"):
        T[m] = C.CALIBRATORS[m](s_cal, y_cal, S_test, alpha, bin_cal=b_cal, bin_test=b_test)
        sets[m] = C.predict_sets(S_test, T[m])
    rows = np.arange(len(y_test))
    hit = {m: sets[m][rows, y_test] for m in sets}
    hit["argmax"] = p_test.argmax(1) == y_test
    mm = y_test == minority
    check = {"n_test": len(y_test), "n_test_minority": int(mm.sum()),
             "bin_edges": ";".join(f"{e:.3f}" for e in edges)}
    for m in ("marginal", "mondrian", "mondrian_x"):
        check[f"cov_minority__{m}"] = float(hit[m][mm].mean()) if mm.any() else np.nan
        m0 = mm & (b_test == 0)
        check[f"cov_min_bin0__{m}"] = float(hit[m][m0].mean()) if m0.any() else np.nan
    ms = sets["mondrian"]
    set_type = np.where(ms.sum(1) == 2, "both", np.where(ms.sum(1) == 0, "empty",
                        np.where(ms[:, minority], "minority_only", "majority_only")))
    mol = pd.DataFrame({
        "mol_id": test_ids, "scaffold": test_scaf,
        "idx_target": idx_test, "y": y_test, "minority": minority,
        "p_minority": p_test[:, minority], "sim_train": sim_test, "support_bin": b_test,
        "mondrian_set": set_type,
        # LAC Mondrian includes BBB- iff p_minority >= 1 - q_minority; keep the cut-off
        "mondrian_minority_cutoff": 1.0 - T["mondrian"][:, minority],
        **{f"miss_{k}": ~hit[k] for k in MISS_KINDS}})
    return mol, check


def collect(settings, loss, seeds, alpha, exclude=frozenset()):
    mols, checks, missing = [], [], []
    for s in settings:
        sub = SETTINGS[s][0]
        for b in BACKBONES:
            for seed in seeds:
                p = RAW / sub / f"{b}_{loss}" / f"seed{seed}.npz"
                if not p.exists():
                    missing.append(str(p.relative_to(REPO)))
                    continue
                m, c = split_records(p, alpha, exclude)
                m["setting"], m["backbone"], m["seed"] = s, b, seed
                c.update(setting=s, backbone=b, seed=seed)
                mols.append(m); checks.append(c)
    if missing:
        print(f"WARNING: {len(missing)} cache files missing, e.g. {missing[:3]}", file=sys.stderr)
    return pd.concat(mols, ignore_index=True), pd.DataFrame(checks), missing


# ------------------------------------------------------------------------------ checks
def check_alignment(M):
    """Same seed must give the same test molecules for every backbone."""
    bad = []
    for (s, seed), g in M.groupby(["setting", "seed"]):
        ref = None
        for b, gb in g.groupby("backbone"):
            ids = tuple(sorted(gb.mol_id))
            if ref is None:
                ref = ids
            elif ids != ref:
                bad.append((s, seed, b))
    return bad


def check_against_stageb(checks, csv_path, loss, alpha):
    """Every reconstructed per-split coverage must equal Stage B (LAC, alpha, loss)."""
    want = ["dataset", "external", "regime", "model_name", "loss", "score", "method", "alpha", "seed",
            "cov_minority", "cov_min_bin0", "n_test_minority"]
    it = pd.read_csv(csv_path, usecols=lambda c: c in want, chunksize=200_000, low_memory=False)
    parts = []
    for ch in it:
        ch = ch[(ch.score == "lac") & np.isclose(ch.alpha, alpha) & (ch.loss == loss)
                & ch.method.isin(["marginal", "mondrian", "mondrian_x"])]
        parts.append(ch)
    B = pd.concat(parts, ignore_index=True)
    B["external"] = B["external"].fillna("").astype(str).replace({"nan": ""})
    rows = []
    for _, c in checks.iterrows():
        _, _, _, ds, ext, reg = SETTINGS[c.setting]
        sel = B[(B.dataset == ds) & (B.external == (ext or "")) & (B.regime == reg)
                & (B.model_name == c.backbone) & (B.seed == c.seed)]
        for m in ("marginal", "mondrian", "mondrian_x"):
            r = sel[sel.method == m]
            for metric in ("cov_minority", "cov_min_bin0"):
                ours = c[f"{metric}__{m}"]
                theirs = float(r[metric].iloc[0]) if len(r) == 1 else np.nan
                rows.append({"setting": c.setting, "backbone": c.backbone, "seed": c.seed, "method": m,
                             "metric": metric, "reconstructed": ours, "stage_b": theirs,
                             "n_stage_b_rows": len(r),
                             "abs_diff": abs(ours - theirs) if np.isfinite(theirs) else np.nan})
    V = pd.DataFrame(rows)
    both_nan = V.reconstructed.isna() & V.stage_b.isna()
    V["ok"] = both_nan | (V.n_stage_b_rows == 1) & (V.abs_diff < 1e-9)
    return V


# ------------------------------------------------------------------------------ per molecule
def per_molecule(M, min_appear):
    """One row per (setting, unique BBB- molecule)."""
    Mm = M[M.y == M.minority]
    agg = {"n_appear": ("seed", "nunique"), "mean_p_minority": ("p_minority", "mean"),
           **{f"n_miss_{k}": (f"miss_{k}", "sum") for k in MISS_KINDS}}
    G = Mm.groupby(["setting", "mol_id", "backbone"]).agg(**agg).reset_index()
    W = G.pivot_table(index=["setting", "mol_id"], columns="backbone",
                      values=[c for c in G.columns if c not in ("setting", "mol_id", "backbone")])
    W.columns = [f"{a}__{b}" for a, b in W.columns]
    W = W.reset_index()
    n_app = W[[f"n_appear__{b}" for b in BACKBONES if f"n_appear__{b}" in W]]
    W["n_appear"] = n_app.max(axis=1)
    W["n_appear_consistent"] = n_app.nunique(axis=1) == 1
    first = Mm.groupby(["setting", "mol_id"]).agg(
        scaffold=("scaffold", "first"), idx_target=("idx_target", "first"),
        mean_sim_train=("sim_train", "mean"), min_sim_train=("sim_train", "min"),
        frac_low_support=("support_bin", lambda v: float(np.mean(np.asarray(v) == 0)))).reset_index()
    W = W.merge(first, on=["setting", "mol_id"], how="left")
    W["eligible"] = W.n_appear >= min_appear
    for k in MISS_KINDS:
        for b in BACKBONES:
            W[f"rate_{k}__{b}"] = W[f"n_miss_{k}__{b}"] / W[f"n_appear__{b}"]
            W[f"persist_{k}__{b}"] = W.eligible & (W[f"n_miss_{k}__{b}"] == W[f"n_appear__{b}"])
        W[f"n_backbones_persist_{k}"] = sum(W[f"persist_{k}__{b}"].astype(int) for b in BACKBONES)
        W[f"cross_arch_{k}"] = W[f"n_backbones_persist_{k}"] == len(BACKBONES)
        W[f"majority_all3_{k}"] = W.eligible & np.all(
            [W[f"rate_{k}__{b}"] >= 0.5 for b in BACKBONES], axis=0)
    k = PRIMARY_MISS
    any_miss = sum(W[f"n_miss_{k}__{b}"] for b in BACKBONES)
    W["category"] = np.select(
        [~W.eligible, W[f"cross_arch_{k}"], W[f"n_backbones_persist_{k}"] > 0, any_miss == 0],
        ["ineligible", "cross_arch_persistent", "persistent_1or2_backbones", "never_missed"],
        "intermittent")
    return W


def attach_chemistry(W):
    """Descriptors (cached features), plus name and other-dataset label via connectivity key."""
    out = []
    ann = annotations()
    for s, g in W.groupby("setting", sort=False):
        tgt = SETTINGS[s][2]
        df = prepare(tgt)["df"]; D = features(tgt)["desc"]
        pos = pd.Series(np.arange(len(df)), index=df.mol_id.values)
        ii = pos.loc[g.mol_id.values].values
        assert np.array_equal(ii, g.idx_target.values.astype(int)), f"mol_id/index mismatch in {s}"
        g = g.copy()
        for j, n in enumerate(DESCRIPTOR_NAMES):
            g[n] = D[ii, j]
        g["smiles"] = df.smiles.values[ii]; g["conn_key"] = df.conn_key.values[ii]
        g["inchikey"] = df.inchikey.values[ii]
        a_self = ann[tgt]; other = "b3db" if tgt == "bbbp" else "bbbp"; a_oth = ann[other]
        g["name"] = g.inchikey.map(a_self["name_by_ik"])
        g["logBB"] = g.inchikey.map(a_self["logbb_by_ik"]) if "logbb_by_ik" in a_self else np.nan
        for col, key in (("b3db_group", "group_by_ik"), ("b3db_n_references", "nref_by_ik"), ("pubchem_cid", "cid_by_ik")):
            g[col] = g.inchikey.map(a_self[key]) if key in a_self else np.nan
        g["other_dataset"] = other
        g["in_other_dataset"] = g.conn_key.isin(a_oth["label_by_ck"].index)
        g["label_in_other"] = g.conn_key.map(a_oth["label_by_ck"])
        lb = a_oth.get("logbb_by_ck")
        g["logBB_in_other"] = g.conn_key.map(lb[~lb.index.duplicated()]) if lb is not None else np.nan
        g["label_disagrees_other_dataset"] = g.label_in_other.notna() & (g.label_in_other != g.minority_label)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def annotations():
    """Names and logBB from the RAW files, keyed by the repo's own standardized identity keys."""
    from bbbcp.chemistry.standardize import standardize_smiles
    from bbbcp.data.datasets import RAW as DRAW
    res = {}
    for name, fn, smi_col, name_col in (("bbbp", "BBBP.csv", "smiles", "name"),
                                        ("b3db", "B3DB_classification.tsv", "SMILES", "compound_name")):
        raw = pd.read_csv(DRAW / fn, sep="\t" if fn.endswith("tsv") else ",", encoding="utf-8-sig")
        recs = [standardize_smiles(s) if isinstance(s, str) else None for s in raw[smi_col]]
        ik = [r.inchikey if r is not None and r.ok else None for r in recs]
        raw = raw.assign(_ik=ik)
        raw = raw[raw._ik.notna()]
        d = {"name_by_ik": raw.groupby("_ik")[name_col].first()}
        if "logBB" in raw:
            d["logbb_by_ik"] = pd.to_numeric(raw.logBB, errors="coerce").groupby(raw._ik).mean()
        if "group" in raw:          # B3DB evidence group (A = numeric logBB ... C/D = categorical only)
            d["group_by_ik"] = raw.groupby("_ik")["group"].agg(lambda v: "".join(sorted(set(map(str, v)))))
            d["nref_by_ik"] = raw.groupby("_ik")["reference"].agg(
                lambda v: max(len([x for x in str(r).split("|") if x]) for r in v))
            d["cid_by_ik"] = raw.groupby("_ik")["CID"].first()
        df = prepare(name)["df"]
        d["label_by_ck"] = df.set_index("conn_key").y        # conn_key is unique after stereo collapse
        if "logbb_by_ik" in d:
            d["logbb_by_ck"] = df.inchikey.map(d["logbb_by_ik"]).set_axis(df.conn_key.values)
        res[name] = d
    return res


# ------------------------------------------------------------------------------ inference
def concordance(W, k, n_perm, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    for s, g in W[W.eligible].groupby("setting", sort=False):
        I = np.column_stack([g[f"persist_{k}__{b}"].values for b in BACKBONES]).astype(bool)
        strata = g.n_appear.values
        obs = int(I.all(1).sum())
        null = np.empty(n_perm, int)
        for r in range(n_perm):
            P = I.copy()
            for st in np.unique(strata):
                ix = np.where(strata == st)[0]
                for j in range(1, P.shape[1]):              # column 0 fixed; permute the others
                    P[ix, j] = I[rng.permutation(ix), j]
            null[r] = P.all(1).sum()
        jac = {}
        for a in range(3):
            for b in range(a + 1, 3):
                u = (I[:, a] | I[:, b]).sum()
                jac[f"jaccard_{BACKBONES[a]}_{BACKBONES[b]}"] = (I[:, a] & I[:, b]).sum() / u if u else np.nan
        rows.append({"setting": s, "miss_definition": k, "n_bbbneg_tested": int((W.setting == s).sum()),
                     "n_eligible": len(g), "median_appearances": float(np.median(strata)),
                     **{f"n_persistent_{b}": int(I[:, j].sum()) for j, b in enumerate(BACKBONES)},
                     "n_cross_arch": obs, "frac_cross_arch_of_eligible": obs / len(g) if len(g) else np.nan,
                     "n_majority_all3": int(g[f"majority_all3_{k}"].sum()),
                     "null_mean": float(null.mean()), "null_q975": float(np.quantile(null, 0.975)),
                     "excess_ratio": obs / null.mean() if null.mean() > 0 else np.inf,
                     "perm_p_one_sided": (1 + (null >= obs).sum()) / (1 + n_perm), **jac})
    return pd.DataFrame(rows)


def cluster_boot_median_diff(x, gx, y, gy, n_boot, seed=0):
    """Median(x) - median(y), resampling Murcko scaffolds (acyclic = own cluster) jointly."""
    lab = np.r_[np.ones(len(x), bool), np.zeros(len(y), bool)]
    v = np.r_[x, y]; g = np.r_[gx, gy]
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == u)[0] for u in range(len(uniq))]
    rng = np.random.default_rng(seed); est = []
    for _ in range(n_boot):
        ix = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        a, b = v[ix][lab[ix]], v[ix][~lab[ix]]
        if len(a) and len(b):
            est.append(np.median(a) - np.median(b))
    return np.percentile(est, [2.5, 97.5]) if est else (np.nan, np.nan)


def characterise(W, k, n_boot, min_group=5):
    rows = []
    for s, g in W[W.eligible].groupby("setting", sort=False):
        pers = g[g[f"cross_arch_{k}"]]; rest = g[~g[f"cross_arch_{k}"]]
        never = g[g.category == "never_missed"]
        cl = lambda d: np.where(d.scaffold.values == "", "acyc:" + d.mol_id.values, d.scaffold.values)
        for dn in PRIMARY_DESC + ["mean_sim_train"] + OTHER_DESC:
            x, y = pers[dn].values.astype(float), rest[dn].values.astype(float)
            r = {"setting": s, "descriptor": dn, "role": "primary" if dn in PRIMARY_DESC else "descriptive",
                 "n_persistent": len(x), "n_rest": len(y), "n_never_missed": len(never),
                 "median_persistent": np.median(x) if len(x) else np.nan,
                 "median_rest": np.median(y) if len(y) else np.nan,
                 "median_never_missed": np.median(never[dn]) if len(never) else np.nan,
                 "q1_persistent": np.quantile(x, .25) if len(x) else np.nan,
                 "q3_persistent": np.quantile(x, .75) if len(x) else np.nan}
            if len(x) >= min_group and len(y) >= min_group:
                u = mannwhitneyu(x, y, alternative="two-sided")
                r["auc_persistent_gt_rest"] = u.statistic / (len(x) * len(y))
                r["mw_p"] = u.pvalue
                r["median_diff"] = r["median_persistent"] - r["median_rest"]
                r["median_diff_lo"], r["median_diff_hi"] = cluster_boot_median_diff(
                    x, cl(pers), y, cl(rest), n_boot)
            rows.append(r)
    R = pd.DataFrame(rows)
    prim = (R.role == "primary") & R.get("mw_p", pd.Series(np.nan, index=R.index)).notna()
    R["holm_p_primary_family"] = np.nan
    if prim.any():
        R.loc[prim, "holm_p_primary_family"] = holm(R.loc[prim, "mw_p"].values)
    return R


def scaffolds(W, k):
    rows = []
    for s, g in W[W.eligible].groupby("setting", sort=False):
        g = g.assign(sc=np.where(g.scaffold == "", "(acyclic)", g.scaffold))
        tot_p = int(g[f"cross_arch_{k}"].sum()); tot = len(g)
        for sc, h in g.groupby("sc"):
            n_p = int(h[f"cross_arch_{k}"].sum())
            if n_p == 0:
                continue
            tab = [[n_p, len(h) - n_p], [tot_p - n_p, tot - len(h) - (tot_p - n_p)]]
            rows.append({"setting": s, "scaffold": sc, "n_bbbneg_eligible": len(h), "n_cross_arch": n_p,
                         "frac_of_scaffold": n_p / len(h), "share_of_all_cross_arch": n_p / tot_p,
                         "fisher_p_enrichment": fisher_exact(tab, alternative="greater")[1] if len(h) >= 3 else np.nan,
                         "example_names": "; ".join(h[h[f"cross_arch_{k}"]].name.dropna().astype(str).head(4))})
    S = pd.DataFrame(rows)
    return S.sort_values(["setting", "n_cross_arch"], ascending=[True, False]) if len(S) else S


# ------------------------------------------------------------------------------ figure
def figure(W, conc, k, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bbbcp.utils.figio import save
    ORANGE, GREY, INK, MUTED = "#eb6834", "#9a9a94", "#222222", "#6b6b66"
    sets_ = [s for s in SETTINGS if s in set(conc.setting)]
    panels = ["TPSA", "HBD", "MW", "cLogP", "mean_sim_train"]
    titles = {"TPSA": "TPSA (Å²)", "HBD": "H-bond donors", "MW": "MW (Da)", "cLogP": "cLogP",
              "mean_sim_train": "Support: max Tanimoto to train"}
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.1), gridspec_kw={"width_ratios": [1.3, 1, 1, 1, 1, 1]})
    ax = axes[0]; yy = np.arange(len(sets_))[::-1]
    c = conc.set_index("setting").loc[sets_]
    ax.barh(yy, c.n_cross_arch, color=ORANGE, height=0.55, label="observed")
    ax.errorbar(c.null_mean, yy, xerr=[c.null_mean * 0, (c.null_q975 - c.null_mean).clip(lower=0)], fmt="|", color=INK,
                ms=10, capsize=3, lw=1, label="null (independent backbones)\nmean, 97.5%")
    for yi, (n, e) in zip(yy, zip(c.n_cross_arch, c.n_eligible)):
        ax.text(n, yi, f"  {n}/{e}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(yy); ax.set_yticklabels(sets_, fontsize=7)
    ax.set_xlabel("BBB− molecules missed by all 3\nbackbones at every appearance", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title("A  Cross-architecture persistent failures", fontsize=8, loc="left")
    E = W[W.eligible]
    for i, (axp, dn) in enumerate(zip(axes[1:], panels)):
        for j, s in enumerate(sets_):
            g = E[E.setting == s]
            for off, grp, col in ((-0.17, g[g[f"cross_arch_{k}"]], ORANGE), (0.17, g[~g[f"cross_arch_{k}"]], GREY)):
                v = grp[dn].dropna().values
                if not len(v):
                    continue
                pos = yy[j] + off
                axp.boxplot(v, positions=[pos], vert=False, widths=0.28, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor="none", edgecolor=col, lw=1.2),
                            medianprops=dict(color=col, lw=2), whiskerprops=dict(color=col, lw=1),
                            capprops=dict(color=col, lw=1))
                if col == ORANGE:
                    jit = np.random.default_rng(j).uniform(-0.09, 0.09, len(v))
                    axp.scatter(v, pos + jit, s=8, color=ORANGE, alpha=0.7, lw=0, zorder=3)
        axp.set_yticks(yy); axp.set_yticklabels([] if i else [], fontsize=7)
        axp.set_title(("BCDEF"[i]) + "  " + titles[dn], fontsize=8, loc="left"); axp.tick_params(labelsize=7)
        if dn == "TPSA":
            axp.axvline(90, ls=":", c=MUTED, lw=0.7)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=ORANGE), Line2D([], [], color=INK, marker="|", ls="", ms=10),
                Line2D([], [], color=ORANGE, lw=2), Line2D([], [], color=GREY, lw=2)],
               ["A: observed count (n shown as persistent / eligible)",
                "A: null if backbones failed independently (mean, 97.5th pct.)",
                "B–F: cross-architecture persistent BBB−", "B–F: other eligible BBB−"],
               fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); save(fig, out); plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="ce")
    ap.add_argument("--seeds", nargs=2, type=int, default=[100, 119], metavar=("FIRST", "LAST"))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--min-appear", type=int, default=2)
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--stageb", default=str(REPO / "results/processed/full_v13_conformal.csv"),
                    help="Stage B CSV to verify against, or 'none'")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude-ids", nargs="*", default=[], help="sensitivity: mol_ids removed from CAL and TEST")
    a = ap.parse_args()
    tag = f"{a.loss}{a.tag}"
    seeds = list(range(a.seeds[0], a.seeds[1] + 1))
    T = REPO / "tables"; F = REPO / "figures"; T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)

    M, checks, missing = collect(a.settings, a.loss, seeds, a.alpha, frozenset(a.exclude_ids))
    # tidy event-level export: one row per (setting, backbone, seed, test molecule)
    M.to_csv(T / f"persistent_failure_events_{tag}.csv.gz", index=False)
    print(f"loaded {len(checks)} split files ({len(missing)} missing), {len(M)} test-molecule rows")
    bad = check_alignment(M)
    if bad:
        sys.exit(f"STOP: test sets differ across backbones for {len(bad)} (setting, seed): {bad[:5]}")
    print("split alignment across backbones: OK")

    if a.stageb.lower() != "none":
        sb = Path(a.stageb)
        if not sb.exists() and Path(str(sb) + ".gz").exists():
            sb = Path(str(sb) + ".gz")               # the public repository ships the Stage B rows gzipped
        V = check_against_stageb(checks, str(sb), a.loss, a.alpha)
        V.to_csv(T / f"persistent_failure_verification_{tag}.csv", index=False)
        n_bad = int((~V.ok).sum())
        print(f"Stage B reconstruction check: {len(V) - n_bad}/{len(V)} values match exactly "
              f"(max |diff| {V.abs_diff.max():.2e})")
        if n_bad:
            print(V[~V.ok].head(12).to_string(), file=sys.stderr)
            sys.exit("STOP: reconstruction does not match Stage B; nothing downstream is trustworthy")
    else:
        print("Stage B reconstruction check SKIPPED (--stageb none)")

    W = per_molecule(M, a.min_appear)
    W["minority_label"] = W.mol_id.map(M.drop_duplicates("mol_id").set_index("mol_id").minority)
    if not W.n_appear_consistent.all():
        sys.exit("STOP: a molecule has different appearance counts across backbones")
    W = attach_chemistry(W)
    conc = pd.concat([concordance(W, k, a.n_perm if k == PRIMARY_MISS else max(a.n_perm // 5, 200))
                      for k in MISS_KINDS], ignore_index=True)
    R = characterise(W, PRIMARY_MISS, a.n_boot)
    # sensitivity definition (pre-stated in the docstring): missed in >= half of appearances by all three
    Wm = W.copy(); Wm[f"cross_arch_{PRIMARY_MISS}"] = Wm[f"majority_all3_{PRIMARY_MISS}"]
    R_maj = characterise(Wm, PRIMARY_MISS, a.n_boot)
    R_maj.to_csv(T / f"persistent_failure_descriptors_majority_{tag}.csv", index=False)
    S = scaffolds(W, PRIMARY_MISS)

    W.sort_values(["setting", "category", "n_appear"], ascending=[True, True, False]).to_csv(
        T / f"persistent_failure_molecules_{tag}.csv", index=False)
    conc.to_csv(T / f"persistent_failure_concordance_{tag}.csv", index=False)
    R.to_csv(T / f"persistent_failure_descriptors_{tag}.csv", index=False)
    S.to_csv(T / f"persistent_failure_scaffolds_{tag}.csv", index=False)
    lab = (W[W.eligible].groupby(["setting", "category"])
           .agg(n=("mol_id", "size"),
                n_in_other_dataset=("in_other_dataset", "sum"),
                n_label_disagrees_other=("label_disagrees_other_dataset", "sum")).reset_index())
    lab.to_csv(T / f"persistent_failure_label_check_{tag}.csv", index=False)
    ev = []
    for s_, g in W[W.eligible & W.b3db_group.notna()].groupby("setting", sort=False):
        weak = ~g.b3db_group.astype(str).str.contains("A")       # no numeric-logBB record
        pers = g[f"cross_arch_{PRIMARY_MISS}"].astype(bool)
        tab = [[int((pers & weak).sum()), int((pers & ~weak).sum())],
               [int((~pers & weak).sum()), int((~pers & ~weak).sum())]]
        ev.append({"setting": s_, "n_persistent": int(pers.sum()),
                   "persistent_no_logBB_record": tab[0][0], "rest_no_logBB_record": tab[1][0], "n_rest": int((~pers).sum()),
                   "frac_persistent_no_logBB": tab[0][0] / max(pers.sum(), 1),
                   "frac_rest_no_logBB": tab[1][0] / max((~pers).sum(), 1),
                   "median_nref_persistent": float(g[pers].b3db_n_references.median()),
                   "median_nref_rest": float(g[~pers].b3db_n_references.median()),
                   "fisher_p_two_sided": fisher_exact(tab)[1] if pers.any() else np.nan})
    pd.DataFrame(ev).to_csv(T / f"persistent_failure_evidence_group_{tag}.csv", index=False)
    figure(W, conc[conc.miss_definition == PRIMARY_MISS], PRIMARY_MISS, F / f"paper_fig3_persistent_failures_{tag}")

    pd.set_option("display.width", 200)
    print("\n== concordance (primary miss = Mondrian LAC exclusion) ==")
    print(conc[conc.miss_definition == PRIMARY_MISS].drop(columns="miss_definition").round(3).to_string(index=False))
    print("\n== concordance under the secondary miss definitions ==")
    print(conc[conc.miss_definition != PRIMARY_MISS][["setting", "miss_definition", "n_eligible", "n_cross_arch",
                                                      "null_mean", "perm_p_one_sided"]].round(3).to_string(index=False))
    print("\n== descriptors: cross-arch persistent vs other eligible BBB- (primary four, Holm) ==")
    cols = ["setting", "descriptor", "n_persistent", "n_rest", "median_persistent", "median_rest",
            "median_diff", "median_diff_lo", "median_diff_hi", "auc_persistent_gt_rest", "mw_p", "holm_p_primary_family"]
    print(R[R.role == "primary"][[c for c in cols if c in R]].round(3).to_string(index=False))
    print("\n== SENSITIVITY (majority definition): primary four, Holm within this table ==")
    print(R_maj[R_maj.role == "primary"][[c for c in cols if c in R_maj]].round(3).to_string(index=False))
    print(R_maj[R_maj.descriptor == "mean_sim_train"][[c for c in cols if c in R_maj]].round(3).to_string(index=False))
    print("\n== support and other descriptors (descriptive only) ==")
    print(R[R.role != "primary"][[c for c in cols[:7] if c in R]].round(3).to_string(index=False))
    print("\n== label check against the other dataset ==")
    print(lab.to_string(index=False))
    print("\n== B3DB evidence group: persistent vs rest (descriptive) ==")
    print(pd.read_csv(T / f"persistent_failure_evidence_group_{tag}.csv").round(3).to_string(index=False))
    print("\n== top scaffolds among cross-arch persistent failures ==")
    print(S.groupby("setting").head(5).round(3).to_string(index=False) if len(S) else "(none)")
    print(f"\nwrote tables/persistent_failure_*_{tag}.csv and figures/paper_fig3_persistent_failures_{tag}.(png,pdf)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", message=".*vert.*")
    main()
