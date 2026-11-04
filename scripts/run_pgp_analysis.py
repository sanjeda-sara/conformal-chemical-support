"""Item D27: are the hard, CNS-looking BBB- molecules (low TPSA, under-covered by Mondrian)
enriched for P-glycoprotein interaction?  Association only.

IMPORTANT: TDC 'Pgp_Broccatelli' is P-gp INHIBITION (1 = inhibitor), not SUBSTRATE status.
BBB efflux is about substrates.  Inhibition is a correlated but different property, so a null
or positive result here does not test the efflux explanation directly.  If you have a P-gp
SUBSTRATE set (SMILES, label), pass it with --substrate-csv; it is analysed the same way.

Procedure
  1. Fit an ECFP4 random forest on the P-gp set (5-fold cross-fitted predictions are used for
     BBB molecules that also occur in the P-gp set; measured labels are reported separately).
  2. For every cached Stage A run (default: CE runs of the confirmatory seeds), build Mondrian
     LAC sets at alpha = 0.1 and mark each BBB- TEST molecule as covered or missed.
  3. Pool over seeds; report miss rate by P-gp score tertile overall and within TPSA < 60,
     and a logistic regression miss ~ pgp + TPSA + support (standardized), with a bootstrap
     over seeds for the coefficients.

usage: python scripts/run_pgp_analysis.py [--runs "results/raw/*/*/rf_ce/seed1[0-1]?.npz"]
                                          [--substrate-csv my_pgp_substrates.csv]
Output: tables/pgp_association.csv, tables/pgp_logit.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from bbbcp.chemistry.fingerprints import bitvects_to_numpy, descriptors, ecfp4_bitvects
from bbbcp.chemistry.standardize import standardize_smiles
from bbbcp.conformal import calibrators as C
from bbbcp.data import datasets
from bbbcp.pipeline import REPO, prepare


def pgp_model(df):
    X = bitvects_to_numpy(ecfp4_bitvects(df.smiles)); y = df.y.values
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0).fit(X, y)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return rf, dict(zip(df.conn_key.values, oof)), dict(zip(df.conn_key.values, y))


def load_substrates(path):
    raw = pd.read_csv(path)
    recs = [standardize_smiles(s) for s in raw.iloc[:, 0]]
    df = pd.DataFrame({"smiles": [r.smiles for r in recs], "conn_key": [r.conn_key for r in recs],
                       "ok": [r.ok for r in recs], "y": raw.iloc[:, 1].astype(int).values})
    return df[df.ok].drop_duplicates("conn_key")


def analyse(pgp_df, label, run_glob, alpha=0.1):
    rf, oof, measured = pgp_model(pgp_df)
    rows = []
    for p in sorted(glob.glob(str(REPO / run_glob))):
        z = np.load(p)
        parts = p.split("/raw/")[1].split("/")
        tag, regime = parts[0], parts[1]
        if tag.split("__to__")[0] not in ("bbbp", "b3db"):
            continue                                  # BBB runs only
        src = tag.split("__to__")
        dtest = prepare(src[-1])["df"]
        test = dtest.iloc[z["idx_test"]]
        y_cal, y_test = z["y_cal"].astype(int), z["y_test"].astype(int)
        minority = int(z["minority"]) if "minority" in z.files else 0
        s_cal = 1 - z["p_cal"][np.arange(len(y_cal)), y_cal]
        S_t = 1 - z["p_test"]
        sets = C.predict_sets(S_t, C.mondrian(s_cal, y_cal, S_t, alpha))
        hit = sets[np.arange(len(y_test)), y_test]
        X = bitvects_to_numpy(ecfp4_bitvects(test.smiles))
        score = rf.predict_proba(X)[:, 1]
        ck = test.conn_key.values
        in_set = np.array([k in oof for k in ck])
        score[in_set] = [oof[k] for k in ck[in_set]]
        meas = np.array([measured.get(k, np.nan) for k in ck], float)
        desc = descriptors(test.smiles)
        m = y_test == minority
        for i in np.where(m)[0]:
            rows.append({"set": label, "run": p.split("/raw/")[1], "setting": f"{tag}/{regime}",
                         "seed": int(p.split("seed")[-1].split(".")[0]), "miss": int(~hit[i]),
                         "pgp_score": score[i], "pgp_measured": meas[i], "in_pgp_set": bool(in_set[i]),
                         "tpsa": desc[i, 2], "hbd": desc[i, 3], "support": z["sim_test"][i]})
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (setname, setting), g in df.groupby(["set", "setting"]):
        g = g.copy()
        g["pgp_tertile"] = pd.qcut(g.pgp_score.rank(method="first"), 3, labels=["low", "mid", "high"])
        for sub, h in (("all BBB-", g), ("BBB- with TPSA<60", g[g.tpsa < 60])):
            for t, hh in h.groupby("pgp_tertile", observed=True):
                out.append({"set": setname, "setting": setting, "subset": sub, "pgp_tertile": t,
                            "n": len(hh), "miss_rate": hh.miss.mean()})
            mm = h[h.pgp_measured.notna()]
            for lab in (0, 1):
                q = mm[mm.pgp_measured == lab]
                out.append({"set": setname, "setting": setting, "subset": sub + " (measured label)",
                            "pgp_tertile": f"measured={lab}", "n": len(q),
                            "miss_rate": q.miss.mean() if len(q) else np.nan})
    return pd.DataFrame(out)


def logit_boot(df, B=500, seed=0):
    rng = np.random.default_rng(seed)
    res = []
    for (setname, setting), g in df.groupby(["set", "setting"]):
        cols = ["pgp_score", "tpsa", "support"]
        seeds = g.seed.unique()
        coefs = []
        for b in range(B):
            pick = rng.choice(seeds, len(seeds), replace=True)
            h = pd.concat([g[g.seed == s] for s in pick])
            if h.miss.nunique() < 2:
                continue
            Xh = (h[cols] - g[cols].mean()) / g[cols].std()
            coefs.append(LogisticRegression(C=1e4, max_iter=2000).fit(Xh, h.miss).coef_[0])
        coefs = np.array(coefs)
        for j, c in enumerate(cols):
            res.append({"set": setname, "setting": setting, "term": c, "coef_mean": coefs[:, j].mean(),
                        "ci_lo": np.quantile(coefs[:, j], 0.025), "ci_hi": np.quantile(coefs[:, j], 0.975),
                        "n_boot": len(coefs)})
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/raw/*/*/rf_ce/seed1[01][0-9].npz")
    ap.add_argument("--substrate-csv", default=None)
    a = ap.parse_args()
    frames = []
    pgp, _ = datasets.load("tdc_pgp")
    frames.append(analyse(pgp, "Pgp_Broccatelli (INHIBITION)", a.runs))
    if a.substrate_csv:
        frames.append(analyse(load_substrates(a.substrate_csv), "user P-gp SUBSTRATE set", a.runs))
    df = pd.concat(frames, ignore_index=True)
    s = summarize(df); lb = logit_boot(df)
    s.to_csv(REPO / "tables/pgp_association.csv", index=False)
    lb.to_csv(REPO / "tables/pgp_logit.csv", index=False)
    print(s.to_string()); print(lb.round(3).to_string())


if __name__ == "__main__":
    main()
