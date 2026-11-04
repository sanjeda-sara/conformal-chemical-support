"""Independent check of scripts/run_persistent_failures.py.  Shares NO analysis code with it.

Re-derives from the raw .npz caches, with its own conformal quantile and its own LAC score:
  1. per (setting, BBB- molecule, backbone): appearances and Mondrian-LAC misses
  2. the cross-architecture persistent set and its size per setting
  3. the four primary descriptors, recomputed from the stored SMILES with RDKit
     (not from the features cache)
and compares each with tables/persistent_failure_{molecules,concordance,descriptors}_<tag>.csv.
Exit code 0 only if everything matches.

usage: python scripts/verify_persistent_failures.py [--loss ce] [--seeds 100 119] [--min-appear 2]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAWDIR = {"bbbp/random": "bbbp/random", "bbbp/scaffold": "bbbp/scaffold", "b3db/random": "b3db/random",
          "b3db/scaffold": "b3db/scaffold", "bbbp→b3db/scaffold": "bbbp__to__b3db/scaffold"}
BB = ("rf", "gin", "chemprop")


def q_hat(scores, alpha):
    s = np.sort(scores); n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return math.inf if k > n else s[k - 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--loss", default="ce")
    ap.add_argument("--seeds", nargs=2, type=int, default=[100, 119]); ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min-appear", type=int, default=2); ap.add_argument("--tag", default="")
    ap.add_argument("--exclude-ids", nargs="*", default=[])
    a = ap.parse_args(); tag = a.loss + a.tag
    W = pd.read_csv(REPO / f"tables/persistent_failure_molecules_{tag}.csv")
    conc = pd.read_csv(REPO / f"tables/persistent_failure_concordance_{tag}.csv")
    R = pd.read_csv(REPO / f"tables/persistent_failure_descriptors_{tag}.csv")
    problems = []

    recs = []
    for setting, sub in RAWDIR.items():
        if setting not in set(W.setting):
            continue
        for b in BB:
            for seed in range(a.seeds[0], a.seeds[1] + 1):
                f = REPO / "results/raw" / sub / f"{b}_{a.loss}" / f"seed{seed}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)          # object id arrays under pandas 3
                mino = int(z["minority"]); yc = z["y_cal"].astype(int); yt = z["y_test"].astype(int)
                sc = np.array([1 - z["p_cal"][i, yc[i]] for i in range(len(yc))])
                ids_t = z["test_ids"].astype(str)
                if a.exclude_ids:
                    pre = "BBBP" if "__to__" in sub else ids_t[0].split(":")[0]
                    kc = ~np.isin(np.array([f"{pre}:{i}" for i in z["idx_cal"]]), a.exclude_ids)
                    sc, yc = sc[kc], yc[kc]
                q_min = q_hat(sc[yc == mino], a.alpha)
                excluded = (1 - z["p_test"][:, mino]) > q_min       # BBB- not in its Mondrian set
                for mid, y, ex in zip(ids_t, yt, excluded):
                    if mid in a.exclude_ids:
                        continue
                    if y == mino:
                        recs.append((setting, mid, b, int(ex)))
    D = pd.DataFrame(recs, columns=["setting", "mol_id", "backbone", "miss"])
    G = D.groupby(["setting", "mol_id", "backbone"]).miss.agg(["size", "sum"]).unstack("backbone")

    # 1. per-molecule counts
    n_checked = 0
    for (setting, mid), row in G.iterrows():
        w = W[(W.setting == setting) & (W.mol_id == mid)]
        if len(w) != 1:
            problems.append(f"{setting} {mid}: {len(w)} rows in molecules table"); continue
        w = w.iloc[0]
        for b in BB:
            if ("size", b) not in row.index or pd.isna(row[("size", b)]):
                continue
            if int(row[("size", b)]) != int(w[f"n_appear__{b}"]) or int(row[("sum", b)]) != int(w[f"n_miss_mondrian__{b}"]):
                problems.append(f"{setting} {mid} {b}: verifier {int(row[('size', b)])}/{int(row[('sum', b)])} "
                                f"vs table {w[f'n_appear__{b}']}/{w[f'n_miss_mondrian__{b}']}")
            n_checked += 1
    if len(G) != len(W):
        problems.append(f"molecule count: verifier {len(G)} vs table {len(W)}")

    # 2. cross-architecture persistent set per setting
    n_app = G["size"].max(axis=1)
    pers = (n_app >= a.min_appear) & np.all([G[("sum", b)] == G[("size", b)] for b in BB if ("sum", b) in G], 0)
    mine = pers.groupby(level=0).sum()
    c = conc[conc.miss_definition == "mondrian"].set_index("setting")
    for s, n in mine.items():
        if int(c.loc[s, "n_cross_arch"]) != int(n):
            problems.append(f"{s}: cross-arch persistent verifier {n} vs table {c.loc[s, 'n_cross_arch']}")
        if int(c.loc[s, "n_eligible"]) != int((n_app.loc[s] >= a.min_appear).sum()):
            problems.append(f"{s}: eligible count differs")
    pset = set(pers[pers].index)
    tset = set(map(tuple, W[W.cross_arch_mondrian.astype(bool)][["setting", "mol_id"]].values))
    if pset != tset:
        problems.append(f"persistent molecule SETS differ: {len(pset ^ tset)} molecules")

    # 3. descriptors from SMILES, then group medians
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    fns = {"TPSA": rdMolDescriptors.CalcTPSA, "HBD": Lipinski.NumHDonors, "MW": Descriptors.MolWt,
           "cLogP": Crippen.MolLogP}
    E = W[W.n_appear >= a.min_appear].copy()
    mols = [Chem.MolFromSmiles(s) for s in E.smiles]
    for dn, fn in fns.items():
        v = np.array([fn(m) for m in mols], float)
        if not np.allclose(v, E[dn].values, atol=1e-6):
            problems.append(f"{dn}: table values differ from RDKit-from-SMILES (max {np.abs(v - E[dn].values).max():.3g})")
        E[dn + "_v"] = v
    for s, g in E.groupby("setting"):
        p = g[g.cross_arch_mondrian.astype(bool)]; r = g[~g.cross_arch_mondrian.astype(bool)]
        for dn in fns:
            t = R[(R.setting == s) & (R.descriptor == dn)].iloc[0]
            for grp, col in ((p, "median_persistent"), (r, "median_rest")):
                mv = np.median(grp[dn + "_v"]) if len(grp) else np.nan
                if not (np.isnan(mv) and np.isnan(t[col])) and not np.isclose(mv, t[col], atol=1e-6):
                    problems.append(f"{s} {dn} {col}: verifier {mv:.4f} vs table {t[col]:.4f}")

    # 4. sensitivity (majority) definition: missed in >= half of appearances by all three backbones
    Rm = pd.read_csv(REPO / f"tables/persistent_failure_descriptors_majority_{tag}.csv")
    maj = (n_app >= a.min_appear) & np.all([G[("sum", b)] / G[("size", b)] >= 0.5 for b in BB if ("sum", b) in G], 0)
    mset = set(maj[maj].index)
    for s in mine.index:
        if int(c.loc[s, "n_majority_all3"]) != int(maj.loc[s].sum()):
            problems.append(f"{s}: majority count verifier {int(maj.loc[s].sum())} vs table {c.loc[s, 'n_majority_all3']}")
    E["maj_v"] = [(s, m) in mset for s, m in zip(E.setting, E.mol_id)]
    for s, g in E.groupby("setting"):
        p = g[g.maj_v]; r = g[~g.maj_v]
        for dn in fns:
            t = Rm[(Rm.setting == s) & (Rm.descriptor == dn)].iloc[0]
            for grp, col in ((p, "median_persistent"), (r, "median_rest")):
                mv = np.median(grp[dn + "_v"]) if len(grp) else np.nan
                if not (np.isnan(mv) and np.isnan(t[col])) and not np.isclose(mv, t[col], atol=1e-6):
                    problems.append(f"MAJORITY {s} {dn} {col}: verifier {mv:.4f} vs table {t[col]:.4f}")
    # 5. per-backbone persistent counts, permutation null (own draws), majority-rule TPSA Holm, evidence group
    from scipy.stats import fisher_exact, mannwhitneyu
    checked = {}
    elig = n_app >= a.min_appear
    rng = np.random.default_rng(2027)
    for s in mine.index:
        tag_s = s.replace("→", "to")
        Is = []
        for b in BB:
            pb = elig.loc[s] & (G.loc[s][("sum", b)] == G.loc[s][("size", b)])
            Is.append(pb.values); checked[f"pf.{tag_s}.{b}"] = int(pb.sum())
            if int(pb.sum()) != int(c.loc[s, f"n_persistent_{b}"]):
                problems.append(f"{s} {b}: persistent count verifier {int(pb.sum())} vs table {c.loc[s, f'n_persistent_{b}']}")
        I = np.column_stack(Is)[elig.loc[s].values]; kk = n_app.loc[s][elig.loc[s]].values
        null = []
        for _ in range(2000):
            P = I.copy()
            for st_ in np.unique(kk):
                ix = np.where(kk == st_)[0]
                for j in (1, 2):
                    P[ix, j] = I[rng.permutation(ix), j]
            null.append(P.all(1).sum())
        checked[f"pf.{tag_s}.null"] = float(np.mean(null))
        if abs(np.mean(null) - c.loc[s, "null_mean"]) > max(0.03, 0.15 * c.loc[s, "null_mean"]):
            problems.append(f"{s}: null mean verifier {np.mean(null):.3f} vs table {c.loc[s, 'null_mean']:.3f}")
        for k_ in ("n_cross_arch", "n_eligible", "n_majority_all3"):
            checked[f"pf.{tag_s}.{k_}"] = int(c.loc[s, k_])          # equality with own counts asserted above
    # majority-rule descriptor tests: own Mann-Whitney + own Holm over the tested family
    pv, keys_ = [], []
    for s, g in E.groupby("setting"):
        p = g[g.maj_v]; r = g[~g.maj_v]
        if len(p) >= 5 and len(r) >= 5:
            for dn in fns:
                pv.append(mannwhitneyu(p[dn + "_v"], r[dn + "_v"], alternative="two-sided").pvalue); keys_.append((s, dn))
    pv = np.array(pv); order = np.argsort(pv); adj = np.empty_like(pv); run_ = 0
    for rnk, i in enumerate(order):
        run_ = max(run_, (len(pv) - rnk) * pv[i]); adj[i] = min(1, run_)
    tp = [adj[i] for i, (s, dn) in enumerate(keys_) if dn == "TPSA"]
    checked["pd.maj.ntests"] = len(pv); checked["pd.maj.tpsa.maxholm"] = float(max(tp))
    tbl = Rm.set_index(["setting", "descriptor"])
    for i, (s, dn) in enumerate(keys_):
        if not np.isclose(adj[i], tbl.loc[(s, dn), "holm_p_primary_family"], atol=1e-9):
            problems.append(f"MAJORITY Holm {s} {dn}: verifier {adj[i]:.4g} vs table {tbl.loc[(s, dn), 'holm_p_primary_family']:.4g}")
    for s, g in E.groupby("setting"):
        tag_s = s.replace("→", "to")
        for dn in ("TPSA",):
            for nm_, mask in (("str", g.cross_arch_mondrian.astype(bool)), ("maj", g.maj_v)):
                checked[f"pd.{nm_}.{tag_s}.{dn}.med"] = float(np.median(g[mask][dn + "_v"])) if mask.any() else np.nan
                checked[f"pd.{nm_}.{tag_s}.{dn}.rest"] = float(np.median(g[~mask][dn + "_v"]))
    # evidence group (cross-check: group labels come from the annotated table, test recomputed here)
    for s, g in E[E.b3db_group.notna()].groupby("setting"):
        pers = g.cross_arch_mondrian.astype(bool); weak = ~g.b3db_group.astype(str).str.contains("A")
        if pers.any():
            tag_s = s.replace("→", "to")
            checked[f"ev.{tag_s}.fp"] = float(weak[pers].mean()); checked[f"ev.{tag_s}.fr"] = float(weak[~pers].mean())
            checked[f"ev.{tag_s}.p"] = float(fisher_exact([[int((pers & weak).sum()), int((pers & ~weak).sum())],
                                                           [int((~pers & weak).sum()), int((~pers & ~weak).sum())]])[1])
    pref = "pfx" if a.exclude_ids else "pf"
    if a.exclude_ids:
        checked = {k.replace("pf.", "pfx.", 1): v for k, v in checked.items() if k.startswith("pf.")}
    pd.Series(checked, name="value").rename_axis("key").to_csv(REPO / f"tables/verify_persistent_checked_{tag}.csv")
    print(f"checked {n_checked} (molecule, backbone) counts, {len(mine)} settings, 4 descriptors, strict and majority definitions")
    if problems:
        print(f"FAIL: {len(problems)} discrepancies"); print("\n".join(problems[:40])); sys.exit(1)
    print("PASS: molecules, persistent sets, counts and descriptor medians all reproduce independently")


if __name__ == "__main__":
    main()
