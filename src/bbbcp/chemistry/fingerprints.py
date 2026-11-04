"""ECFP4 fingerprints, Tanimoto support measures and RDKit descriptors."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4_bitvects(smiles):
    return [_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def bitvects_to_numpy(fps) -> np.ndarray:
    X = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def max_tanimoto(query_fps, ref_fps) -> np.ndarray:
    """Max Tanimoto similarity of each query to the reference set (nearest-neighbour support).
    The reference set must not contain the query molecules themselves."""
    out = np.empty(len(query_fps))
    for i, fp in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        out[i] = max(sims) if len(sims) else 0.0
    return out


def mean_topk_tanimoto(query_fps, ref_fps, k: int = 5) -> np.ndarray:
    out = np.empty(len(query_fps))
    for i, fp in enumerate(query_fps):
        sims = np.sort(np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)))[::-1]
        out[i] = sims[:k].mean()
    return out


DESCRIPTOR_NAMES = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "FracCSP3",
                    "AromRings", "Rings", "HeavyAtoms", "FormalChargeAbs", "AromFrac"]


def descriptors(smiles) -> np.ndarray:
    """Reliably computable BBB-relevant descriptors. NOT a CNS MPO score
    (CNS MPO needs pKa and logD, which RDKit does not compute defensibly)."""
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        ha = m.GetNumHeavyAtoms()
        arom = sum(a.GetIsAromatic() for a in m.GetAtoms())
        rows.append([
            Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumRings(m),
            ha, sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()),
            arom / ha if ha else 0.0,
        ])
    return np.asarray(rows, dtype=float)


# ----------------------------------------------------------------------------- added v1.3
def maccs_numpy(smiles) -> np.ndarray:
    from rdkit.Chem import MACCSkeys
    out = np.zeros((len(smiles), 167), dtype=np.uint8)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)), out[i])
    return out


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between binary fingerprint matrices A (n, d) and B (m, d)."""
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    na = A.sum(1)[:, None]; nb = B.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def support_measures(X_query: np.ndarray, X_ref: np.ndarray, k: int = 5, chunk: int = 512):
    """(max Tanimoto, mean top-k Tanimoto) of each query row to the reference set."""
    mx, tk = np.empty(len(X_query)), np.empty(len(X_query))
    for s in range(0, len(X_query), chunk):
        T = tanimoto_matrix(X_query[s:s + chunk], X_ref)
        part = np.sort(T, axis=1)[:, ::-1]
        mx[s:s + chunk] = part[:, 0]
        tk[s:s + chunk] = part[:, :k].mean(1)
    return mx, tk
