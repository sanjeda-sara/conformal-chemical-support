"""GIN refactored from legacy_moml2026/train_gin.py.

Kept from MoML: 5 GINConv layers, hidden 128, BatchNorm, dropout 0.5, sum pooling,
Adam lr 1e-3, wd 1e-5, ReduceLROnPlateau, early stopping on validation AUROC.
Changed (see docs/DATA_AUDIT.md):
  * atom features: the 9 integer PyG/OGB atom features are embedded (sum of per-column
    embeddings) instead of being cast to float as raw integers;
  * early stopping uses the VALIDATION split only; calibration molecules are never
    seen by training or model selection;
  * MC dropout keeps BatchNorm in eval mode and activates dropout only (legacy code
    called model.train(), which used test-batch statistics AND updated the BN running
    stats with test data);
  * loss: 'ce' or 'wce' (inverse class frequency, normalized to mean weight 1).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import from_smiles

ATOM_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]  # torch_geometric.utils.smiles x_map sizes


def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def featurize(smiles, y=None):
    data = []
    for i, s in enumerate(smiles):
        d = from_smiles(s)
        d.x = d.x.long()
        d.y = torch.tensor([int(y[i]) if y is not None else -1])
        d.idx = torch.tensor([i])
        data.append(d)
    return data


class AtomEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n + 1, dim) for n in ATOM_DIMS])

    def forward(self, x):
        x = x.clamp(min=0)
        return sum(e(x[:, j].clamp(max=e.num_embeddings - 1)) for j, e in enumerate(self.emb))


class GIN(nn.Module):
    def __init__(self, hidden=128, layers=5, dropout=0.5, n_classes=2):
        super().__init__()
        self.dropout = dropout
        self.mc = False
        self.enc = AtomEncoder(hidden)
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True)); self.bns.append(nn.BatchNorm1d(hidden))
        self.head_fc = nn.Linear(hidden, hidden)
        self.head_out = nn.Linear(hidden, n_classes)

    def _drop(self, x):
        return F.dropout(x, p=self.dropout, training=self.training or self.mc)

    def embed(self, data):
        x = self.enc(data.x)
        for conv, bn in zip(self.convs, self.bns):
            x = self._drop(F.relu(bn(conv(x, data.edge_index))))
        return global_add_pool(x, data.batch)

    def forward(self, data):
        h = F.relu(self.head_fc(self.embed(data)))
        return self.head_out(self._drop(h))


def class_weights(y, kind):
    if kind != "wce":
        return None
    cnt = np.bincount(y, minlength=2).astype(float)
    w = cnt.sum() / (2 * cnt)
    return torch.tensor(w / w.mean(), dtype=torch.float)


@torch.no_grad()
def predict_proba(model, data, dev, mc_passes: int = 0, batch_size=256):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    model.eval()
    if mc_passes:
        model.mc = True
    runs = []
    for _ in range(max(1, mc_passes)):
        out = [F.softmax(model(b.to(dev)), -1).cpu().numpy() for b in loader]
        runs.append(np.concatenate(out))
    model.mc = False
    return np.mean(runs, 0)


def focal_loss(logits, target, gamma=2.0):
    """Focal loss (Lin et al. 2017), gamma = 2, no alpha balancing."""
    logp = F.log_softmax(logits, -1).gather(1, target.view(-1, 1)).view(-1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()


def train_gin(train_data, val_data, loss="ce", seed=0, epochs=60, patience=15, lr=1e-3,
              batch_size=64, dev=None, verbose=False, model_fn=None):
    """loss: 'ce' | 'wce' (inverse-frequency weights) | 'focal' (gamma 2) | 'os' (random
    oversampling of the minority with a weighted sampler).  model_fn builds the network
    (default GIN; GCN for the Tursunbadalov comparison).  With an empty VAL set the model
    trains for `epochs` epochs and keeps the last state (used by the 50/25/25 reproduction)."""
    dev = dev or device_auto()
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    model = (model_fn or GIN)().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    ytr = np.array([int(d.y) for d in train_data])
    w = class_weights(ytr, loss)
    w = w.to(dev) if w is not None else None
    if loss == "os":
        cnt = np.bincount(ytr, minlength=2).astype(float)
        sw = torch.tensor((1.0 / cnt)[ytr], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(sw, num_samples=len(ytr), replacement=True, generator=g)
        loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
    yval = np.array([int(d.y) for d in val_data])
    best, best_state, since, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b in loader:
            b = b.to(dev)
            if b.num_graphs < 2:  # BatchNorm needs >1 graph
                continue
            opt.zero_grad()
            out = model(b)
            if loss == "focal":
                l = focal_loss(out, b.y.view(-1))
            else:
                l = F.cross_entropy(out, b.y.view(-1), weight=w)
            l.backward(); opt.step()
        if len(val_data) == 0:
            best_state = copy.deepcopy(model.state_dict()); best = float("nan")
            continue
        pv = predict_proba(model, val_data, dev)
        metric = roc_auc_score(yval, pv[:, 1]) if len(np.unique(yval)) == 2 else -F.nll_loss(
            torch.log(torch.tensor(pv) + 1e-12), torch.tensor(yval)).item()
        sched.step(metric); hist.append(metric)
        if metric > best + 1e-4:
            best, best_state, since = metric, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        if verbose:
            print(ep, round(metric, 4))
        if since >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_val_auroc": best, "epochs_run": ep + 1, "train_seconds": time.time() - t0,
                   "device": str(dev)}


# ----------------------------------------------------------------------------- GCN (v1.3)
_ELEMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
_HYB = None


def featurize_onehot(smiles, y=None):
    """32-d one-hot atom features in the style of Tursunbadalov & Tursunbadalov (2026):
    element (12 + other = 13), degree 0-5 (6), total H 0-4 (5), hybridization sp/sp2/sp3/other (4),
    aromatic (1), formal charge -1/0/+1 (3) = 32."""
    from rdkit import Chem
    from torch_geometric.data import Data
    hyb = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
           Chem.rdchem.HybridizationType.SP3]
    data = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        feats = []
        for a in m.GetAtoms():
            f = np.zeros(32, np.float32)
            sym = a.GetSymbol(); f[_ELEMS.index(sym) if sym in _ELEMS else 12] = 1
            f[13 + min(a.GetDegree(), 5)] = 1
            f[19 + min(a.GetTotalNumHs(), 4)] = 1
            h = a.GetHybridization(); f[24 + (hyb.index(h) if h in hyb else 3)] = 1
            f[28] = float(a.GetIsAromatic())
            f[29 + int(np.clip(a.GetFormalCharge(), -1, 1)) + 1] = 1
            feats.append(f)
        ei = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()]
        ei = ei + [[j, i2] for i2, j in ei]
        edge = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros((2, 0), dtype=torch.long)
        d = Data(x=torch.tensor(np.array(feats)), edge_index=edge)
        d.y = torch.tensor([int(y[i]) if y is not None else -1]); d.idx = torch.tensor([i])
        data.append(d)
    return data


class GCN(nn.Module):
    """Two GCNConv layers 32 -> 64 -> 64, mean pooling, linear read-out (Tursunbadalov 2026)."""
    def __init__(self, in_dim=32, hidden=64, n_classes=2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1 = GCNConv(in_dim, hidden); self.c2 = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, n_classes)
        self.mc = False

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        return self.out(global_mean_pool(x, data.batch))
