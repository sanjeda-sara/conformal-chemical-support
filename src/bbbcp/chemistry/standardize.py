"""Molecular standardization, identity keys and scaffolds.

Rules (documented in docs/DATA_AUDIT.md):
  1. Parse SMILES with RDKit; unparseable -> dropped and logged.
  2. Keep the largest organic fragment (removes salts / counter-ions such as [Cl-], Na+).
  3. Neutralize charges where a neutral form exists (rdMolStandardize.Uncharger).
  4. Canonical isomeric SMILES is the record key; the InChIKey is computed on the
     standardized parent.  `conn_key` = first InChIKey block (connectivity only,
     ignores stereo and isotopes) and is used for conservative overlap auditing.
  5. Tautomers are NOT canonicalized (slow, and can change scaffolds); this is a
     stated limitation.
  6. Bemis-Murcko scaffold without chirality; acyclic molecules get scaffold "".
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()


@dataclass(frozen=True)
class StdRecord:
    input_smiles: str
    smiles: str | None
    inchikey: str | None
    conn_key: str | None
    scaffold: str | None
    n_fragments_in: int
    ok: bool
    note: str = ""


def standardize_smiles(smi: str) -> StdRecord:
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        # second chance without sanitization problems (e.g. valence), then give up
        return StdRecord(str(smi), None, None, None, None, 0, False, "unparseable")
    nfrag = len(Chem.GetMolFrags(mol))
    try:
        parent = _chooser.choose(mol)
        parent = _uncharger.uncharge(parent)
        Chem.SanitizeMol(parent)
    except Exception as exc:  # pragma: no cover - rare RDKit failures
        return StdRecord(smi, None, None, None, None, nfrag, False, f"standardize:{exc}")
    can = Chem.MolToSmiles(parent, isomericSmiles=True)
    ik = Chem.MolToInchiKey(parent) or None
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    return StdRecord(smi, can, ik, ik.split("-")[0] if ik else None, scaf, nfrag, True,
                     "salt_stripped" if nfrag > 1 else "")
