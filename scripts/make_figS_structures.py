"""Supplementary figure: structures of every cross-architecture persistent BBB- failure (strict
definition, seeds 100-119), with dataset label flags.  Source: tables/persistent_failure_molecules_ce.csv."""
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

REPO = Path(__file__).resolve().parents[1]
W = pd.read_csv(REPO / "tables/persistent_failure_molecules_ce.csv", low_memory=False)
P = W[W.cross_arch_mondrian.astype(bool)]
CURATION = {"B3DB:2544": "name/structure mismatch", "B3DB:3330": "mis-drawn (propyl)", "B3DB:3001": "mis-drawn (BBB+ record exists)"}
rows = []
for mid, g in P.groupby("mol_id", sort=False):
    r = g.iloc[0]
    name = str(r["name"]) if pd.notna(r["name"]) else mid
    name = name.replace(" hydrochloride", " HCl").replace(" citrate", "")
    if name[:1].isdigit():
        name = mid
    flag = ""
    if bool(g.label_disagrees_other_dataset.any()):
        flag = "B3DB label: BBB+"
    if mid in CURATION:
        flag = CURATION[mid]
    sets = ", ".join(sorted(s.replace("bbbp", "BBBP").replace("b3db", "B3DB") for s in g.setting))
    rows.append((Chem.MolFromSmiles(r.smiles), f"{name}\n{sets}" + (f"\n[{flag}]" if flag else "")))
rows.sort(key=lambda t: t[1])
# Printed at full text width: 4 structures per row, large atom labels and legends (about 8 pt in the PDF).
opts = Draw.MolDrawOptions()
opts.legendFontSize = 40; opts.legendFraction = 0.30; opts.bondLineWidth = 2.6
opts.minFontSize = 30; opts.maxFontSize = 40; opts.padding = 0.06
img = Draw.MolsToGridImage([m for m, _ in rows], molsPerRow=4, subImgSize=(600, 520),
                           legends=[l for _, l in rows], returnPNG=False, drawOptions=opts)
out = REPO / "figures/supp_persistent_structures.png"
img.save(out); print(f"{len(rows)} unique molecules -> {out}")
