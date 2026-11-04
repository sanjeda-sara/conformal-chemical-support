"""Writes results/dataset_audit.csv and results/dataset_overlap_audit.csv."""
import itertools
import json

import numpy as np
import pandas as pd

from bbbcp.chemistry.fingerprints import max_tanimoto
from bbbcp.pipeline import REPO, prepare

names = ["bbbp", "b3db", "b3db_external"]
rows = []
for n in names:
    a = dict(prepare(n)["audit"]); a["dataset"] = n; rows.append(a)
pd.DataFrame(rows).to_csv(REPO / "results/dataset_audit.csv", index=False)

ov = []
for a, b in itertools.permutations(names, 2):
    A, B = prepare(a), prepare(b)
    da, db = A["df"], B["df"]
    m_ik = da.inchikey.isin(set(db.inchikey)); m_ck = da.conn_key.isin(set(db.conn_key))
    m_sc = da.scaffold.isin(set(db.scaffold) - {""})
    merged = da[m_ck].merge(db[["conn_key", "y"]], on="conn_key", suffixes=("_a", "_b"))
    agree = (merged.y_a == merged.y_b).mean() if len(merged) else np.nan
    sim = max_tanimoto(A["fps"], B["fps"])
    ov.append({"dataset_a": a, "dataset_b": b, "n_a": len(da),
               "n_a_exact_inchikey_in_b": int(m_ik.sum()), "n_a_connectivity_in_b": int(m_ck.sum()),
               "frac_a_connectivity_in_b": float(m_ck.mean()),
               "label_agreement_on_connectivity_overlap": float(agree),
               "n_label_disagreements": int((merged.y_a != merged.y_b).sum()),
               "n_a_scaffold_in_b": int(m_sc.sum()),
               "n_a_not_in_b": int((~m_ck).sum()),
               "n_a_not_in_b_minority": int(((~m_ck) & (da.y == 0)).sum()),
               "median_max_tanimoto_a_to_b": float(np.median(sim)),
               "frac_a_max_tanimoto_ge_0.8": float((sim >= 0.8).mean())})
pd.DataFrame(ov).to_csv(REPO / "results/dataset_overlap_audit.csv", index=False)
print(pd.DataFrame(ov).to_string())

# stereo-family audit on raw B3DB records (before collapse)
from rdkit import Chem
from bbbcp.chemistry.standardize import standardize_smiles
raw = pd.read_csv(REPO / "data/raw/B3DB_classification.tsv", sep="\t", encoding="utf-8-sig")
recs = [standardize_smiles(s) for s in raw.SMILES]
raw["ck"] = [r.conn_key for r in recs]
raw["flat"] = [Chem.MolToSmiles(Chem.MolFromSmiles(r.smiles), isomericSmiles=False) if r.ok else None for r in recs]
fam = raw[raw.ck.notna()].groupby("ck").filter(lambda g: len(g) > 1)
fa = {"rows_in_multi_record_families": int(len(fam)), "n_families": int(fam.ck.nunique()),
      "families_all_same_2D_structure": int(fam.groupby("ck").flat.nunique().eq(1).sum())}
pd.DataFrame([fa]).to_csv(REPO / "results/b3db_stereo_family_audit.csv", index=False); print(fa)
