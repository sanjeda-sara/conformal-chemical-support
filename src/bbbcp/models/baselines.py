"""ECFP4 + Random Forest and Chemprop v2 D-MPNN wrappers.

loss='ce'  -> unweighted training
loss='wce' -> inverse-class-frequency weighting (RF: class_weight='balanced_subsample';
              Chemprop: per-datapoint weights in the BCE loss, mean weight 1)
"""
from __future__ import annotations

import time

import numpy as np


def train_rf(X_train, y_train, loss="ce", seed=0, n_estimators=500, n_jobs=-1):
    """loss 'os' = random oversampling of the minority to balance classes before fitting."""
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    if loss == "os":
        rng = np.random.default_rng(seed)
        cnt = np.bincount(y_train, minlength=2)
        mi = int(cnt.argmin())
        extra = rng.choice(np.where(y_train == mi)[0], cnt.max() - cnt.min(), replace=True)
        X_train = np.vstack([X_train, X_train[extra]]); y_train = np.concatenate([y_train, y_train[extra]])
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=1, max_features="sqrt",
                                class_weight=("balanced_subsample" if loss == "wce" else None),
                                n_jobs=n_jobs, random_state=seed)
    rf.fit(X_train, y_train)
    return rf, {"train_seconds": time.time() - t0}


def train_chemprop(smi_train, y_train, smi_val, y_val, loss="ce", seed=0, epochs=40, patience=10,
                   batch_size=64, accelerator="auto", foundation=None):
    """Chemprop v2 MPNN (bond message passing, mean aggregation, binary FFN head).
    Early stopping on validation loss; calibration data are never used.
    foundation='chemeleon' initialises the message passing from CheMeleon (chemprop >= 2.2)."""
    import lightning.pytorch as pl
    import torch
    from chemprop import data, featurizers, models, nn
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    pl.seed_everything(seed, workers=True)
    t0 = time.time()
    cnt = np.bincount(y_train, minlength=2).astype(float)
    cw = cnt.sum() / (2 * cnt); cw = cw / cw.mean()

    def dps(smis, ys, weighted):
        return [data.MoleculeDatapoint.from_smi(s, np.array([float(y)]),
                                                weight=float(cw[int(y)]) if weighted else 1.0)
                for s, y in zip(smis, ys)]

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dtr = data.MoleculeDataset(dps(smi_train, y_train, loss == "wce"), feat)
    dva = data.MoleculeDataset(dps(smi_val, y_val, False), feat)
    ltr = data.build_dataloader(dtr, batch_size=batch_size, num_workers=0, seed=seed)
    lva = data.build_dataloader(dva, batch_size=batch_size, num_workers=0, shuffle=False)
    if foundation == "chemeleon":
        from bbbcp.models.extra import chemeleon_message_passing
        mp = chemeleon_message_passing()
    else:
        mp = nn.BondMessagePassing()
    model = models.MPNN(mp, nn.MeanAggregation(), nn.BinaryClassificationFFN(input_dim=mp.output_dim),
                        batch_norm=True)
    import tempfile
    ckpt = ModelCheckpoint(dirpath=tempfile.mkdtemp(prefix="chemprop_ckpt_"), monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, devices=1, logger=False,
                         enable_progress_bar=False, enable_model_summary=False, deterministic=False,
                         callbacks=[EarlyStopping("val_loss", patience=patience, mode="min"), ckpt])
    trainer.fit(model, ltr, lva)
    best = models.MPNN.load_from_checkpoint(ckpt.best_model_path)
    best.eval()
    return best, {"train_seconds": time.time() - t0, "epochs_run": trainer.current_epoch}


def predict_chemprop(model, smiles):
    import torch
    from chemprop import data, featurizers
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles], feat)
    loader = data.build_dataloader(ds, batch_size=256, num_workers=0, shuffle=False)
    out = []
    model = model.to("cpu").eval()
    with torch.no_grad():
        for b in loader:
            out.append(model(b.bmg, b.V_d, b.X_d).cpu().numpy().reshape(-1))
    p1 = np.clip(np.concatenate(out), 1e-7, 1 - 1e-7)
    return np.stack([1 - p1, p1], 1)
