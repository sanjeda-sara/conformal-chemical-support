"""Additional backbones (request items C19-C22, B18).

maplight     CatBoost on ECFP4 counts (1024) + Avalon counts (1024) + ErG (315) + RDKit 2D
             descriptors, following the MapLight TDC recipe (Notwell & Wood 2023).  CPU.
chemberta    frozen DeepChem/ChemBERTa-77M-MTR, mean-pooled last hidden state (384-d) + logistic
             regression (C chosen on VAL), as in Tursunbadalov & Tursunbadalov (2026).  Needs the
             `transformers` package and a one-time model download from Hugging Face.
molformer    same recipe with ibm/MoLFormer-XL-both-10pct (trust_remote_code).
chemeleon    Chemprop v2 MPNN initialised from the CheMeleon foundation-model message-passing
             weights (Burns et al. 2025), fine-tuned end to end.  Needs chemprop >= 2.2 and a
             one-time download of chemeleon_mp.pt from Zenodo (done automatically).
gin_ens5     deep ensemble of 5 GINs with different seeds (Lakshminarayanan et al. 2017).
chemprop_ens5 deep ensemble of 5 Chemprop models.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "results" / "processed" / "embeddings"


# ----------------------------------------------------------------------------- MapLight
def maplight_features(smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdReducedGraphs
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    names = [n for n, _ in Descriptors._descList]
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        ecfp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        av = np.zeros(1024, np.float32)
        DataStructs.ConvertToNumpyArray(pyAvalonTools.GetAvalonCountFP(m, nBits=1024), av)
        erg = np.asarray(rdReducedGraphs.GetErGFingerprint(m), np.float32)
        desc = []
        for n, f in Descriptors._descList:
            try:
                desc.append(float(f(m)))
            except Exception:
                desc.append(np.nan)
        rows.append(np.concatenate([ecfp, av, erg, np.asarray(desc, np.float32)]))
    X = np.vstack(rows)
    X[~np.isfinite(X)] = np.nan
    return X


def train_maplight(smi_train, y_train, smi_val=None, y_val=None, loss="ce", seed=0):
    from catboost import CatBoostClassifier
    t0 = time.time()
    Xtr = maplight_features(smi_train)
    cw = None
    if loss == "wce":
        cnt = np.bincount(y_train, minlength=2).astype(float)
        cw = list(cnt.sum() / (2 * cnt))
    clf = CatBoostClassifier(random_seed=seed, verbose=0, thread_count=-1, class_weights=cw)
    clf.fit(Xtr, y_train)
    return clf, {"train_seconds": time.time() - t0, "n_features": int(Xtr.shape[1])}


def predict_maplight(clf, smiles):
    return clf.predict_proba(maplight_features(smiles))


# ----------------------------------------------------------------------------- chemical LMs
LM_IDS = {"chemberta": "DeepChem/ChemBERTa-77M-MTR", "molformer": "ibm/MoLFormer-XL-both-10pct"}


def lm_embed(smiles, which="chemberta", batch_size=64, cache_key=None):
    """Mean-pooled last hidden state of a frozen chemical language model (CPU is fine)."""
    if cache_key is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{which}_{cache_key}.npy"
        if f.exists():
            return np.load(f)
    import torch
    from transformers import AutoModel, AutoTokenizer
    mid = LM_IDS[which]
    trc = which == "molformer"
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=trc)
    model = AutoModel.from_pretrained(mid, trust_remote_code=trc, **({"deterministic_eval": True} if trc else {}))
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(smiles), batch_size):
            enc = tok(list(smiles[s:s + batch_size]), padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    E = np.vstack(out).astype(np.float32)
    if cache_key is not None:
        np.save(CACHE / f"{which}_{cache_key}.npy", E)
    return E


def train_lm_head(E_train, y_train, E_val, y_val, loss="ce", seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    best, best_c = -1, 1.0
    cw = "balanced" if loss == "wce" else None
    for c in (0.01, 0.1, 1.0, 10.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight=cw))
        clf.fit(E_train, y_train)
        if len(E_val) and len(np.unique(y_val)) == 2:
            auc = roc_auc_score(y_val, clf.predict_proba(E_val)[:, 1])
            if auc > best:
                best, best_c = auc, c
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000, class_weight=cw))
    clf.fit(E_train, y_train)
    return clf, {"train_seconds": time.time() - t0, "C": best_c, "best_val_auroc": best}


# ----------------------------------------------------------------------------- CheMeleon
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def chemeleon_message_passing():
    """Load the CheMeleon pretrained bond message-passing block (chemprop >= 2.2)."""
    import torch
    from chemprop import nn
    path = REPO / "data" / "models" / "chemeleon_mp.pt"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CHEMELEON_URL, path)
    blob = torch.load(path, weights_only=True)
    mp = nn.BondMessagePassing(**blob["hyper_parameters"])
    mp.load_state_dict(blob["state_dict"])
    return mp
