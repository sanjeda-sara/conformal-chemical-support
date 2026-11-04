"""Two-stage pipeline.

Stage A (train.py)  : split -> fit model on TRAIN, select on VAL -> cache probabilities for
                      val / cal / test (+ MC-dropout for GIN) and chemical-support features
                      to results/raw/<dataset>/<regime>/<model>_<loss>/seed<k>.npz, and append
                      a provenance record to results/manifests/registry.jsonl.
Stage B (evaluate.py): read cached predictions only -> every conformal method x score x alpha
                      -> long-format results CSV.  No model is refit; no test label is used
                      for anything except computing the reported metrics.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bbbcp.chemistry.fingerprints import bitvects_to_numpy, ecfp4_bitvects, max_tanimoto
from bbbcp.data import datasets
from bbbcp.data.splits import fingerprint_clusters, make_split

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "results" / "processed"
RAW = REPO / "results" / "raw"
MANIFEST = REPO / "results" / "manifests" / "registry.jsonl"


def code_version():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        h = hashlib.sha1()
        for p in sorted((REPO / "src").rglob("*.py")):
            h.update(p.read_bytes())
        return "srchash-" + h.hexdigest()[:10]


def file_sha(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


_CACHE = {}


def _load_cache(path: Path):
    """Read a derived-cache pickle, or return None so the caller rebuilds it.

    Pickles are a SPEED cache only: they are written by whichever pandas/numpy produced them
    and are not portable across major versions (a pandas 3.x pickle raises inside pandas 2.x,
    and vice versa).  Any failure therefore means "rebuild", never "crash" -- rebuilding costs
    ~1 min per dataset.  Delete results/processed/*.pkl at any time; `make rebuild-caches`
    regenerates them in the current environment.
    """
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception as e:                                   # noqa: BLE001 - any failure -> rebuild
        import warnings
        warnings.warn(f"rebuilding unreadable cache {path.name} ({type(e).__name__}: {e})", stacklevel=2)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _load_cache(path):
    """Read a derived cache, returning None when it is absent OR unreadable.

    Pickles are NOT portable across pandas versions (a file written by pandas 3.x raises
    StringDtype.__init__ errors under pandas 2.2).  Caches here are derived data that can
    always be rebuilt from data/raw/, so an unreadable cache is a rebuild, never an error.
    Set BBBCP_REBUILD_CACHE=1 to force a rebuild."""
    import os
    if os.environ.get("BBBCP_REBUILD_CACHE") or not Path(path).exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception as e:                                    # noqa: BLE001
        import warnings
        warnings.warn(f"rebuilding unreadable cache {Path(path).name} ({type(e).__name__}: {e})")
        return None


def prepare(name: str):
    """Standardized dataset + ECFP4 + fixed chemical clusters (cached)."""
    if name in _CACHE:
        return _CACHE[name]
    PROC.mkdir(parents=True, exist_ok=True)
    fp_path = PROC / f"{name}.pkl"
    obj = _load_cache(fp_path)
    if obj is None:
        df, audit = datasets.load(name)
        fps = ecfp4_bitvects(df.smiles)
        X = bitvects_to_numpy(fps)
        clusters = fingerprint_clusters(X, k=20, seed=0)
        obj = {"df": df, "audit": audit, "X": X, "clusters": clusters}
        pd.to_pickle(obj, fp_path)
    obj["fps"] = ecfp4_bitvects(obj["df"].smiles)
    _CACHE[name] = obj
    return obj


def features(name: str):
    """RDKit descriptors (12, see chemistry.fingerprints.DESCRIPTOR_NAMES) and MACCS keys,
    cached separately so the v1.2 dataset pickles stay valid."""
    key = f"feat::{name}"
    if key in _CACHE:
        return _CACHE[key]
    path = PROC / f"{name}_features.pkl"
    obj = _load_cache(path)
    if obj is None:
        from bbbcp.chemistry.fingerprints import descriptors, maccs_numpy
        smi = prepare(name)["df"].smiles.values
        obj = {"desc": descriptors(smi), "maccs": maccs_numpy(smi)}
        pd.to_pickle(obj, path)
    _CACHE[key] = obj
    return obj


def butina_labels(name: str, threshold: float):
    path = PROC / f"{name}_butina{threshold}.npy"
    if path.exists():
        return np.load(path)
    from bbbcp.data.splits import butina_clusters
    lab = butina_clusters(prepare(name)["fps"], threshold)
    np.save(path, lab)
    return lab


def tdc_scaffold_split(name: str, seed: int, frac=(0.7, 0.1, 0.2)):
    """Port of TDC's create_scaffold_split (PyTDC 1.1.15, tdc/utils/split.py, MIT licence),
    applied to our curated molecules: Bemis-Murcko scaffold sets, 'big' sets (> half of the
    valid or test size) shuffled first, then small sets, with Python's random.Random(seed);
    filled into train / valid / test.  CAL = 1/8 of the TDC-train scaffold sets (seeded),
    VAL = TDC valid, TEST = TDC test.  NOTE: the TDC ADMET leaderboard uses a fixed,
    pre-computed test set; this reproduces the split FUNCTION, not that exact test set."""
    from random import Random
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    df = prepare(name)["df"]
    rnd = Random(seed)
    scaffolds = {}
    for i, smi in enumerate(df.smiles.values):
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), includeChirality=False)
        scaffolds.setdefault(sc, set()).add(i)
    n = len(df); tr_n = int(n * frac[0]); va_n = int(n * frac[1]); te_n = n - tr_n - va_n
    sets = list(scaffolds.values())
    big = [s_ for s_ in sets if len(s_) > va_n / 2 or len(s_) > te_n / 2]
    small = [s_ for s_ in sets if not (len(s_) > va_n / 2 or len(s_) > te_n / 2)]
    rnd.seed(seed); rnd.shuffle(big); rnd.shuffle(small)
    train, val, test = [], [], []
    for s_ in big + small:
        if len(train) + len(s_) <= tr_n:
            train += sorted(s_)
        elif len(val) + len(s_) <= va_n:
            val += sorted(s_)
        else:
            test += sorted(s_)
    rng = np.random.default_rng(seed)
    tr = np.array(train, int); scaf = df.scaffold.values[tr]
    groups = {}
    for i, sc in zip(tr, scaf):
        groups.setdefault(sc, []).append(i)
    cal, target = [], len(tr) / 8
    for g in rng.permutation(list(groups)):
        if len(cal) >= target:
            break
        cal.extend(groups[g])
    cal = np.sort(np.array(cal, int))
    return {"train": np.setdiff1d(tr, cal), "val": np.sort(np.array(val, int)), "cal": cal,
            "test": np.sort(np.array(test, int))}


def split_for(name, regime, seed):
    d = prepare(name)
    kw = {}
    if regime.startswith("butina"):
        kw["butina"] = butina_labels(name, float(regime.replace("butina", "")))
    if regime == "tdc_scaffold":
        kw["tdc_split"] = tdc_scaffold_split(name, seed)
    sp = make_split(regime, seed, d["df"].scaffold.values, d["clusters"], **kw)
    return sp


def _empty_safe(fn, smi, Xe):
    return np.zeros((0, 2)) if len(smi) == 0 else fn(smi, Xe)


def _fit_predict(model_name, loss, seed, d, sp, eval_sets, dataset=None):
    """Returns dict part -> probs, plus info.  eval_sets: dict part -> (smiles, X)."""
    df, X = d["df"], d["X"]
    y = df.y.values
    tr, va = sp["train"], sp["val"]
    info = {}
    out = {}
    t_pred = [0.0]

    def predict_all(fn):
        t0 = time.time()
        for k, (smi, Xe) in eval_sets.items():
            out[k] = _empty_safe(fn, smi, Xe)
        t_pred[0] = time.time() - t0

    if model_name == "rf":
        from bbbcp.models.baselines import train_rf
        m, info = train_rf(X[tr], y[tr], loss, seed)
        predict_all(lambda smi, Xe: m.predict_proba(Xe))
    elif model_name in ("gin", "gcn", "gin_ens5"):
        from bbbcp.models.gin import GCN, device_auto, featurize, featurize_onehot, predict_proba, train_gin
        dev = device_auto()
        feat = featurize_onehot if model_name == "gcn" else featurize
        trd = feat(df.smiles.values[tr], y[tr]); vad = feat(df.smiles.values[va], y[va])
        n_members = 5 if model_name == "gin_ens5" else 1
        members = []
        for k in range(n_members):
            mk, info_k = train_gin(trd, vad, loss=loss, seed=seed * 100 + k if n_members > 1 else seed, dev=dev,
                                   model_fn=GCN if model_name == "gcn" else None,
                                   epochs=60, patience=60 if model_name == "gcn" else 15)
            members.append(mk); info = info_k
        info["n_members"] = n_members
        t0 = time.time()
        for k, (smi, Xe) in eval_sets.items():
            if len(smi) == 0:
                out[k] = np.zeros((0, 2)); continue
            fd = feat(smi)
            ps = [predict_proba(mm, fd, dev) for mm in members]
            out[k] = np.mean(ps, 0)
            if n_members > 1:
                out[k + "_ensvar"] = np.var([p[:, 1] for p in ps], 0)
            if model_name == "gin" and k in ("cal", "test"):
                out[k + "_mc"] = predict_proba(members[0], fd, dev, mc_passes=30)
        t_pred[0] = time.time() - t0
    elif model_name in ("chemprop", "chemeleon", "chemprop_ens5"):
        from bbbcp.models.baselines import predict_chemprop, train_chemprop
        n_members = 5 if model_name == "chemprop_ens5" else 1
        members = []
        for k in range(n_members):
            mk, info = train_chemprop(df.smiles.values[tr], y[tr], df.smiles.values[va], y[va], loss,
                                      seed * 100 + k if n_members > 1 else seed,
                                      foundation="chemeleon" if model_name == "chemeleon" else None)
            members.append(mk)
        info["n_members"] = n_members
        predict_all(lambda smi, Xe: np.mean([predict_chemprop(mm, smi) for mm in members], 0))
    elif model_name == "maplight":
        from bbbcp.models.extra import predict_maplight, train_maplight
        m, info = train_maplight(df.smiles.values[tr], y[tr], loss=loss, seed=seed)
        predict_all(lambda smi, Xe: predict_maplight(m, smi))
    elif model_name in ("chemberta", "molformer"):
        from bbbcp.models.extra import lm_embed, train_lm_head
        E = lm_embed(df.smiles.values, model_name, cache_key=dataset)
        m, info = train_lm_head(E[tr], y[tr], E[va], y[va], loss=loss, seed=seed)
        lookup = {smi: i for i, smi in enumerate(df.smiles.values)}
        for k, (smi, Xe) in eval_sets.items():
            if len(smi) == 0:
                out[k] = np.zeros((0, 2)); continue
            hit = [lookup.get(x_) for x_ in smi]
            if all(h is not None for h in hit):
                Ek = E[np.array(hit)]
            else:                                   # external test set: embed and cache
                Ek = lm_embed(smi, model_name, cache_key=f"ext_{dataset}_{len(smi)}")
            out[k] = m.predict_proba(Ek)
    else:
        raise ValueError(model_name)
    info["predict_seconds"] = t_pred[0]
    return out, info


def run_one(dataset, regime, model_name, loss, seed, external=None, force=False):
    """Train one model on one split and cache predictions.
    external: name of a second dataset used as an external TEST set (e.g. b3db minus
    BBBP overlap); calibration still comes from the source dataset's cal split."""
    tag = f"{dataset}" + (f"__to__{external}" if external else "")
    outdir = RAW / tag / regime / f"{model_name}_{loss}"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"seed{seed}.npz"
    if path.exists() and not force:
        return path
    d = prepare(dataset)
    df, X, fps = d["df"], d["X"], d["fps"]
    sp = split_for(dataset, regime, seed)
    y = df.y.values
    eval_sets = {k: (df.smiles.values[sp[k]], X[sp[k]]) for k in ("val", "cal", "test")}
    ext_keep = None
    if external:
        e = prepare(external)
        # remove every external molecule whose connectivity key appears anywhere in the source
        ext_keep = np.where(~e["df"].conn_key.isin(set(df.conn_key)).values)[0]
        eval_sets["test"] = (e["df"].smiles.values[ext_keep], e["X"][ext_keep])
    t0 = time.time()
    probs, info = _fit_predict(model_name, loss, seed, d, sp, eval_sets, dataset=dataset)
    train_fps = [fps[i] for i in sp["train"]]
    sim = {k: max_tanimoto([fps[i] for i in sp[k]], train_fps) for k in ("val", "cal")}
    if external:
        efps = prepare(external)["fps"]
        sim["test"] = max_tanimoto([efps[i] for i in ext_keep], train_fps)
        y_test = prepare(external)["df"].y.values[ext_keep]
        test_ids = prepare(external)["df"].mol_id.values[ext_keep]
        test_scaf = prepare(external)["df"].scaffold.values[ext_keep]
    else:
        sim["test"] = max_tanimoto([fps[i] for i in sp["test"]], train_fps)
        y_test = y[sp["test"]]; test_ids = df.mol_id.values[sp["test"]]
        test_scaf = df.scaffold.values[sp["test"]]
    arrays = {f"p_{k}": v for k, v in probs.items()}
    arrays.update({f"sim_{k}": v for k, v in sim.items()})
    arrays.update({"y_val": y[sp["val"]], "y_cal": y[sp["cal"]], "y_test": y_test,
                   "idx_train": sp["train"], "idx_val": sp["val"], "idx_cal": sp["cal"],
                   "idx_test": sp["test"] if not external else ext_keep,
                   "test_ids": test_ids.astype(str), "test_scaffold": test_scaf.astype(str),
                   "minority": np.array(int(d["audit"]["minority_label"]))})
    np.savez_compressed(path, **arrays)
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_version": code_version(),
        "dataset": dataset, "external_test": external,
        "dataset_file_sha1": {"bbbp": file_sha(datasets.RAW / "BBBP.csv"),
                              "b3db": file_sha(datasets.RAW / "B3DB_classification.tsv")},
        "standardization": "largest-organic-fragment + uncharge + canonical isomeric SMILES; dedupe by InChIKey; conflicting-label duplicates dropped",
        "regime": regime, "split_seed": seed, "model_seed": seed, "model": model_name, "loss": loss,
        "counts": {k: int(len(sp[k])) for k in ("train", "val", "cal", "test")},
        "class_counts": {"train": np.bincount(y[sp["train"]], minlength=2).tolist(),
                         "val": np.bincount(y[sp["val"]], minlength=2).tolist(),
                         "cal": np.bincount(y[sp["cal"]], minlength=2).tolist(),
                         "test": np.bincount(y_test, minlength=2).tolist()},
        "unique_scaffolds": {k: len(set(df.scaffold.values[sp[k]])) for k in ("train", "val", "cal", "test")},
        "hyperparameters": _hparams(model_name), "train_info": info,
        "wall_seconds": time.time() - t0, "python": platform.python_version(),
        "result_file": str(path.relative_to(REPO)),
    }
    if regime == "cluster":
        rec["held_out_clusters"] = sp["_held_out_clusters"].tolist()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return path


def _hparams(m):
    return {"rf": {"fp": "ECFP4 2048", "n_estimators": 500, "max_features": "sqrt"},
            "gin": {"layers": 5, "hidden": 128, "dropout": 0.5, "lr": 1e-3, "wd": 1e-5,
                    "epochs": 60, "patience": 15, "batch": 64, "select": "val AUROC",
                    "mc_passes": 30},
            "chemprop": {"mp": "bond D-MPNN default", "agg": "mean", "batch_norm": True,
                         "epochs": 40, "patience": 10, "select": "val loss"},
            "chemprop_ens5": {"members": 5, "base": "chemprop"},
            "chemeleon": {"mp": "CheMeleon pretrained (Zenodo 15460715)", "agg": "mean", "epochs": 40,
                          "patience": 10, "select": "val loss"},
            "gin_ens5": {"members": 5, "base": "gin"},
            "gcn": {"layers": "GCNConv 32-64-64", "pool": "mean", "epochs": 60, "lr": 1e-3,
                    "atom_features": "32-d one-hot (Tursunbadalov 2026 style)"},
            "maplight": {"features": "ECFP4 counts 1024 + Avalon counts 1024 + ErG + RDKit 2D",
                         "model": "CatBoost defaults"},
            "chemberta": {"model": "DeepChem/ChemBERTa-77M-MTR frozen, mean pool", "head": "logistic, C on VAL"},
            "molformer": {"model": "ibm/MoLFormer-XL-both-10pct frozen, mean pool", "head": "logistic, C on VAL"}}[m]
