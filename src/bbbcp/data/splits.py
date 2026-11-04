"""Label-free splitting.  NONE of these functions accept labels (enforced by
tests/test_split_integrity.py::test_split_functions_take_no_labels).

Four disjoint parts: train / val / cal / test (default 0.60 / 0.10 / 0.15 / 0.15).
  * train : fit model parameters
  * val   : early stopping / model selection only
  * cal   : conformal calibration only (never seen by the model)
  * test  : final evaluation only

Regimes
  R1  random            molecules permuted uniformly -> cal and test exchangeable.
  R2a scaffold_balanced scaffold groups assigned at random (large groups forced to
                        train, as in Chemprop's balanced scaffold split).  train is
                        scaffold-disjoint from val/cal/test; cal and test are
                        exchangeable *at the scaffold-group level*.
  R2b cluster_holdout   whole chemical clusters (k-means on ECFP4, fixed k, fixed seed)
                        are held out as test; train/val/cal come from the remaining
                        clusters via R2a.  cal and test are NOT exchangeable -> this is
                        an empirical stress test; no finite-sample CP claim is made.
"""
from __future__ import annotations

import numpy as np

FRACS = (0.60, 0.10, 0.15, 0.15)
PARTS = ("train", "val", "cal", "test")


def _check(fracs):
    assert len(fracs) == 4 and abs(sum(fracs) - 1) < 1e-9


def random_split(n: int, seed: int, fracs=FRACS) -> dict[str, np.ndarray]:
    _check(fracs)
    perm = np.random.default_rng(seed).permutation(n)
    cuts = np.round(np.cumsum(fracs)[:-1] * n).astype(int)
    parts = np.split(perm, cuts)
    return {k: np.sort(v) for k, v in zip(PARTS, parts)}


def _groups(keys):
    uniq, inv = np.unique(np.asarray(keys, dtype=object).astype(str), return_inverse=True)
    return [np.where(inv == g)[0] for g in range(len(uniq))]


def _fill(groups, n, fracs, rng, big_to_train=True):
    targets = np.asarray(fracs) * n
    big = [g for g in groups if len(g) > targets[3] / 2] if big_to_train else []
    small = [g for g in groups if len(g) <= targets[3] / 2] if big_to_train else list(groups)
    order = rng.permutation(len(small))
    out = {k: [] for k in PARTS}
    sizes = dict.fromkeys(PARTS, 0)
    for g in big:
        out["train"].append(g); sizes["train"] += len(g)
    for gi in order:
        g = small[gi]
        for k, t in zip(PARTS, targets):
            if sizes[k] + len(g) <= t or k == "test":
                out[k].append(g); sizes[k] += len(g)
                break
    return {k: np.sort(np.concatenate(v)) if v else np.array([], int) for k, v in out.items()}


def scaffold_balanced_split(scaffolds, seed: int, fracs=FRACS) -> dict[str, np.ndarray]:
    _check(fracs)
    groups = _groups(scaffolds)
    # acyclic molecules share scaffold "" -> treat each as its own group so the
    # empty scaffold does not form one giant pseudo-group
    keys = np.asarray(scaffolds, dtype=object)
    groups = [g for g in groups if keys[g[0]] != ""] + \
             [np.array([i]) for i in np.where(keys == "")[0]]
    return _fill(groups, len(scaffolds), fracs, np.random.default_rng(seed))


def fingerprint_clusters(X: np.ndarray, k: int = 20, seed: int = 0) -> np.ndarray:
    """Fixed chemical clusters (k-means on binary ECFP4).  Computed once per dataset."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(X.astype(np.float32))


def cluster_holdout_split(scaffolds, clusters, seed: int, fracs=FRACS) -> dict[str, np.ndarray]:
    _check(fracs)
    rng = np.random.default_rng(seed)
    n = len(scaffolds)
    clusters = np.asarray(clusters)
    # whole scaffold groups follow their majority cluster
    keys = np.asarray(scaffolds, dtype=object).astype(str)
    grp_cluster = {}
    for g in _groups(keys):
        vals, cnt = np.unique(clusters[g], return_counts=True)
        grp_cluster[keys[g[0]]] = vals[cnt.argmax()]
    mol_cluster = np.array([grp_cluster[s] for s in keys])
    mol_cluster[keys == ""] = clusters[keys == ""]
    test, held = [], []
    for c in rng.permutation(np.unique(mol_cluster)):
        if len(test) >= fracs[3] * n:
            break
        idx = np.where(mol_cluster == c)[0]
        if len(test) + len(idx) > 1.6 * fracs[3] * n:  # skip clusters that would overshoot badly
            continue
        test.extend(idx.tolist()); held.append(int(c))
    test = np.sort(np.array(test))
    rest = np.setdiff1d(np.arange(n), test)
    f = np.asarray(fracs[:3]) / sum(fracs[:3])
    sub = scaffold_balanced_split(keys[rest], seed, (f[0], f[1], f[2] / 2, f[2] / 2))
    out = {"train": rest[sub["train"]], "val": rest[sub["val"]],
           "cal": rest[np.concatenate([sub["cal"], sub["test"]])], "test": test}
    out = {k: np.sort(v) for k, v in out.items()}
    out["_held_out_clusters"] = np.array(held)
    return out


def assert_disjoint(split: dict, n: int | None = None):
    parts = [np.asarray(split[k]) for k in PARTS]
    allidx = np.concatenate(parts)
    assert len(allidx) == len(np.unique(allidx)), "split parts overlap"
    if n is not None:
        assert len(allidx) == n, "split does not cover all molecules"


def assert_scaffold_disjoint(split: dict, scaffolds, between=("train", "test"), ignore_acyclic=True):
    s = np.asarray(scaffolds, dtype=object)
    a = set(s[split[between[0]]]); b = set(s[split[between[1]]])
    inter = a & b
    if ignore_acyclic:
        inter.discard("")
    assert not inter, f"{len(inter)} scaffolds shared between {between}"


def butina_clusters(fps, dist_threshold: float) -> np.ndarray:
    """Taylor-Butina clustering on ECFP4 Tanimoto distance (RDKit).  Deterministic."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    cl = Butina.ClusterData(dists, n, dist_threshold, isDistData=True)
    lab = np.empty(n, int)
    for k, members in enumerate(cl):
        lab[list(members)] = k
    return lab


def butina_holdout_split(butina_labels, seed: int, fracs=FRACS) -> dict[str, np.ndarray]:
    """Whole Butina clusters held out as TEST (random cluster order, seeded); TRAIN/VAL/CAL
    are drawn from the remaining clusters by cluster group as in R2a.  CAL and TEST are NOT
    exchangeable (empirical stress test, like R2b)."""
    _check(fracs)
    rng = np.random.default_rng(seed)
    lab = np.asarray(butina_labels)
    n = len(lab)
    test = []
    for c in rng.permutation(np.unique(lab)):
        if len(test) >= fracs[3] * n:
            break
        idx = np.where(lab == c)[0]
        if len(test) + len(idx) > 1.6 * fracs[3] * n:
            continue
        test.extend(idx.tolist())
    test = np.sort(np.array(test, int))
    rest = np.setdiff1d(np.arange(n), test)
    f = np.asarray(fracs[:3]) / sum(fracs[:3])
    groups = {}
    for i in rest:
        groups.setdefault(int(lab[i]), []).append(i)
    order = rng.permutation(list(groups))
    targets = f * len(rest); parts = {"train": [], "val": [], "cal": []}; names = ["train", "val", "cal"]
    big = sorted(order, key=lambda g: -len(groups[g]))
    for g in big:
        sizes = np.array([len(parts[k]) for k in names])
        need = targets - sizes
        k = names[int(np.argmax(need))]
        parts[k].extend(groups[g])
    out = {k: np.sort(np.array(v, int)) for k, v in parts.items()}
    out["test"] = test
    return out


def scaffold_deterministic_split(scaffolds, fracs=(0.70, 0.10, 0.10, 0.10)):
    """MoleculeNet/DeepChem deterministic scaffold split (item D25): scaffold sets sorted by
    size (largest first), filled into TRAIN, then VAL, then CAL, then TEST.  TRAIN+VAL = the
    DeepChem 80% train, CAL ~ DeepChem valid, TEST = DeepChem test (smallest scaffold sets).
    One split only (seed ignored).  CAL and TEST are NOT exchangeable."""
    s = np.asarray(scaffolds, dtype=object).astype(str)
    groups = {}
    for i, k in enumerate(s):
        groups.setdefault(k, []).append(i)
    sets = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(s); cuts = np.cumsum(fracs)[:-1] * n
    parts = {"train": [], "val": [], "cal": [], "test": []}
    for g in sets:
        filled = len(parts["train"]) + len(parts["val"]) + len(parts["cal"])
        if len(parts["train"]) + len(g) <= cuts[0]:
            parts["train"].extend(g)
        elif len(parts["train"]) + len(parts["val"]) + len(g) <= cuts[1]:
            parts["val"].extend(g)
        elif filled + len(g) <= cuts[2]:
            parts["cal"].extend(g)
        else:
            parts["test"].extend(g)
    return {k: np.sort(np.array(v, int)) for k, v in parts.items()}


def make_split(regime: str, seed: int, scaffolds, clusters=None, fracs=FRACS, fps=None,
               butina=None, tdc_split=None):
    """Regimes: random, scaffold, cluster (v1.2); random5025 (Tursunbadalov 50/25/25, no VAL),
    scaffold_det (MoleculeNet deterministic), butina0.4 / butina0.6 (Butina cluster holdout),
    tdc_scaffold (TDC benchmark scaffold split; CAL carved from TDC train by scaffold)."""
    n = len(scaffolds)
    if regime == "random":
        sp = random_split(n, seed, fracs)
    elif regime == "random5025":
        sp = random_split(n, seed, (0.50, 0.0, 0.25, 0.25))
    elif regime == "scaffold":
        sp = scaffold_balanced_split(scaffolds, seed, fracs)
    elif regime == "cluster":
        sp = cluster_holdout_split(scaffolds, clusters, seed, fracs)
    elif regime == "scaffold_det":
        sp = scaffold_deterministic_split(scaffolds)
    elif regime.startswith("butina"):
        if butina is None:
            raise ValueError("butina labels required")
        sp = butina_holdout_split(butina, seed, fracs)
    elif regime == "tdc_scaffold":
        if tdc_split is None:
            raise ValueError("tdc_split required (pipeline supplies it for TDC datasets)")
        sp = tdc_split
    else:
        raise ValueError(regime)
    assert_disjoint(sp, n)
    if regime in ("scaffold", "cluster", "scaffold_det"):
        for other in ("val", "cal", "test"):
            assert_scaffold_disjoint(sp, scaffolds, ("train", other))
    return sp
