"""Dataset loaders with standardization, de-duplication and an audit trail.

Label convention used everywhere in this repo (binary):
    y = 1  BBB+ / permeable   (majority in BBBP and B3DB)
    y = 0  BBB- / non-permeable (minority)
Other datasets map their positive/active class to y = 1 and the audit records
which class is the minority.

Every loader returns a pandas DataFrame with columns
    mol_id, smiles (standardized), inchikey, conn_key, scaffold, y, source
and an audit dict (counts dropped for each reason).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bbbcp.chemistry.standardize import standardize_smiles

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw"


def _standardize_frame(df: pd.DataFrame, smiles_col: str, y_col: str, source: str):
    audit = {"source": source, "n_raw": int(len(df))}
    recs = [standardize_smiles(s) for s in df[smiles_col].tolist()]
    out = pd.DataFrame({
        "input_smiles": [r.input_smiles for r in recs],
        "smiles": [r.smiles for r in recs],
        "inchikey": [r.inchikey for r in recs],
        "conn_key": [r.conn_key for r in recs],
        "scaffold": [r.scaffold for r in recs],
        "salt_stripped": [r.note == "salt_stripped" for r in recs],
        "ok": [r.ok for r in recs],
        "y": df[y_col].astype(int).values,
    })
    audit["n_unparseable"] = int((~out.ok).sum())
    audit["n_salt_stripped"] = int(out.salt_stripped.sum())
    out = out[out.ok].drop(columns="ok")

    # exact duplicates after standardization (same full InChIKey)
    g = out.groupby("inchikey")["y"]
    nlab = g.transform("nunique")
    conflict = nlab > 1
    audit["n_rows_in_conflicting_duplicate_groups"] = int(conflict.sum())
    audit["n_conflicting_duplicate_groups"] = int(out[conflict].inchikey.nunique())
    out = out[~conflict]
    dup = out.duplicated("inchikey", keep="first")
    audit["n_consistent_duplicates_removed"] = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    # stereo-isomer families (same connectivity key = first InChIKey block, different
    # InChIKey).  B3DB lists many E/Z and enantiomer variants as separate records; kept
    # apart they put near-identical molecules on both sides of a random split.  We collapse
    # each family to ONE record when all labels agree and drop the family when they disagree.
    fam = out.groupby("conn_key")["inchikey"].transform("nunique")
    audit["n_rows_in_stereo_families"] = int((fam > 1).sum())
    fam_lab = out.groupby("conn_key")["y"].transform("nunique")
    audit["n_rows_dropped_stereo_label_disagreement"] = int((fam_lab > 1).sum())
    out = out[fam_lab == 1]
    dup2 = out.duplicated("conn_key", keep="first")
    audit["n_stereo_variants_collapsed"] = int(dup2.sum())
    out = out[~dup2].reset_index(drop=True)
    out["source"] = source
    out.insert(0, "mol_id", [f"{source}:{i}" for i in range(len(out))])
    audit["n_final"] = int(len(out))
    audit["n_pos_y1"] = int((out.y == 1).sum())
    audit["n_neg_y0"] = int((out.y == 0).sum())
    audit["minority_label"] = int(out.y.value_counts().idxmin())
    audit["minority_frac"] = float(out.y.value_counts(normalize=True).min())
    audit["n_scaffolds"] = int(out.scaffold.nunique())
    audit["n_acyclic"] = int((out.scaffold == "").sum())
    return out, audit


def load_bbbp():
    """MoleculeNet BBBP (Martins et al. 2012). Raw file copied from the MoML project."""
    df = pd.read_csv(RAW / "BBBP.csv")
    return _standardize_frame(df, "smiles", "p_np", "BBBP")


def load_b3db(which: str = "classification"):
    """B3DB (Meng et al. 2021, CC0). which in {classification, external}."""
    fn = {"classification": "B3DB_classification.tsv",
          "external": "B3DB_classification_external.tsv"}[which]
    df = pd.read_csv(RAW / fn, sep="\t", encoding="utf-8-sig")
    df["y"] = (df["BBB+/BBB-"].str.strip() == "BBB+").astype(int)
    src = "B3DB" if which == "classification" else "B3DB_ext"
    out, audit = _standardize_frame(df, "SMILES", "y", src)
    return out, audit


LOADERS = {"bbbp": load_bbbp, "b3db": load_b3db,
           "b3db_external": lambda: load_b3db("external")}


def load(name: str):
    return LOADERS[name]()


# ----------------------------------------------------------------------------- added v1.3
# Other imbalanced tasks used by Tursunbadalov & Tursunbadalov (2026): BACE, ClinTox CT_TOX,
# Tox21 SR-ARE (request item D26).  Raw CSVs are the MoleculeNet originals, mirrored at
# github.com/GLambard/Molecules_Dataset_Collection (originals/); scripts/download_data.py
# fetches them.  Convention: y = 1 = active / toxic / inhibitor, which is the MINORITY for
# these tasks (the audit records it, and Stage B reads it as `minority`).
MIRROR = "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/originals/"


def _read_or_fetch(fname: str, url: str, **kw) -> pd.DataFrame:
    path = RAW / fname
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path, **kw)


def load_bace():
    df = _read_or_fetch("bace.csv", MIRROR + "bace.csv")
    return _standardize_frame(df, "smiles", "Class", "BACE")


def load_clintox():
    df = _read_or_fetch("clintox.csv", MIRROR + "clintox.csv")
    return _standardize_frame(df, "smiles", "CT_TOX", "ClinTox")


def load_tox21_sr_are():
    df = _read_or_fetch("tox21.csv", MIRROR + "tox21.csv")
    df = df[df["SR-ARE"].notna()].copy()
    return _standardize_frame(df, "smiles", "SR-ARE", "Tox21_SR-ARE")


def load_bbbp_raw():
    """BBBP as used by Tursunbadalov & Tursunbadalov (2026): every RDKit-parseable record
    (2,039), WITHOUT de-duplication or stereo collapse.  Only for the head-to-head
    reproduction (item F42); never for headline results."""
    from rdkit import Chem
    df = pd.read_csv(RAW / "BBBP.csv")
    recs = []
    for s, y in zip(df.smiles, df.p_np):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            r = standardize_smiles(s)
            recs.append((Chem.MolToSmiles(m), r.inchikey, r.conn_key, r.scaffold, int(y)))
    out = pd.DataFrame(recs, columns=["smiles", "inchikey", "conn_key", "scaffold", "y"])
    out["source"] = "BBBP_raw"; out.insert(0, "mol_id", [f"BBBP_raw:{i}" for i in range(len(out))])
    audit = {"source": "BBBP_raw", "n_raw": int(len(df)), "n_final": int(len(out)),
             "n_pos_y1": int(out.y.sum()), "n_neg_y0": int((out.y == 0).sum()),
             "minority_label": int(out.y.value_counts().idxmin()),
             "minority_frac": float(out.y.value_counts(normalize=True).min()),
             "n_scaffolds": int(out.scaffold.nunique()), "note": "no de-duplication (reproduction only)"}
    return out, audit


TDC_DATAVERSE = {"BBB_Martins": 4259566, "Pgp_Broccatelli": 4259597}   # ids from PyTDC 1.1.15 metadata


def _tdc_frame(name: str):
    """Download a TDC single-prediction file straight from Harvard Dataverse (the same file
    PyTDC downloads; tab-separated, columns Drug_ID, Drug, Y).  PyTDC itself is NOT needed:
    it pins numpy < 2 and rdkit < 2024.3, which conflict with this environment.
    Cached to data/raw/tdc_<name>.tsv."""
    path = RAW / f"tdc_{name}.tsv"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://dataverse.harvard.edu/api/access/datafile/{TDC_DATAVERSE[name]}", path)
    return pd.read_csv(path, sep="\t")


def load_tdc_bbb_martins():
    """TDC BBB_Martins (item D24).  y = 1 = BBB+.  Use with regime 'tdc_scaffold'."""
    df = _tdc_frame("BBB_Martins")
    return _standardize_frame(df, "Drug", "Y", "TDC_BBB_Martins")


def load_tdc_pgp():
    """TDC Pgp_Broccatelli (item D27).  NOTE: this is P-gp INHIBITION (y = 1 = inhibitor),
    not P-gp SUBSTRATE status.  Efflux at the BBB is about substrates; inhibition is at best
    a correlated proxy.  Used only by scripts/run_pgp_analysis.py."""
    df = _tdc_frame("Pgp_Broccatelli")
    return _standardize_frame(df, "Drug", "Y", "TDC_Pgp_Broccatelli")


LOADERS.update({"bace": load_bace, "clintox": load_clintox, "tox21_sr_are": load_tox21_sr_are,
                "bbbp_raw": load_bbbp_raw, "tdc_bbb_martins": load_tdc_bbb_martins,
                "tdc_pgp": load_tdc_pgp})
