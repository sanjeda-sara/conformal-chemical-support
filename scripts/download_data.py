"""Fetch raw data (run on your own machine; the build sandbox could only reach GitHub).

BBBP.csv and B3DB TSVs are already in data/raw/ (copied from the MoML project and
github.com/theochem/B3DB @ 75dab1cc, CC0-1.0).  This script re-downloads them and,
optionally, MoleculeNet tasks for secondary experiments (Tier 2).
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SOURCES = {
    "BBBP.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "B3DB_classification.tsv": "https://raw.githubusercontent.com/theochem/B3DB/75dab1cc607bb7a03f3de5c576cffd223e767844/B3DB/B3DB_classification.tsv",
    "B3DB_classification_external.tsv": "https://raw.githubusercontent.com/theochem/B3DB/75dab1cc607bb7a03f3de5c576cffd223e767844/B3DB/B3DB_classification_external.tsv",
    "B3DB_regression.tsv": "https://raw.githubusercontent.com/theochem/B3DB/75dab1cc607bb7a03f3de5c576cffd223e767844/B3DB/B3DB_regression.tsv",
}
OPTIONAL = {  # Tier 2, MoleculeNet (deepchem S3)
    "HIV.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
}
_M = "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/originals/"
EXTRA = {  # v1.3: other imbalanced tasks (MoleculeNet originals, GitHub mirror) and TDC files
    "bace.csv": _M + "bace.csv",                       # sha1 aa9789b217c8
    "clintox.csv": _M + "clintox.csv",                 # sha1 ef7929dad518
    "tox21.csv": _M + "tox21.csv",                     # sha1 d3cfa45ff93e
    "tdc_BBB_Martins.tsv": "https://dataverse.harvard.edu/api/access/datafile/4259566",
    "tdc_Pgp_Broccatelli.tsv": "https://dataverse.harvard.edu/api/access/datafile/4259597",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--optional", action="store_true")
    ap.add_argument("--extra", action="store_true", help="BACE, ClinTox, Tox21, TDC BBB_Martins and Pgp")
    ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in {**SOURCES, **(OPTIONAL if a.optional else {}), **(EXTRA if a.extra else {})}.items():
        dest = RAW / name
        if dest.exists() and not a.force:
            print("exists", name, hashlib.sha1(dest.read_bytes()).hexdigest()[:12]); continue
        print("downloading", url); urllib.request.urlretrieve(url, dest)
        print("  sha1", hashlib.sha1(dest.read_bytes()).hexdigest()[:12])
