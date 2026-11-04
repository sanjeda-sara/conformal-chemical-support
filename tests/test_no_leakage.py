"""Pipeline-level leakage checks on cached prediction files (written by Stage A)."""
import glob
import numpy as np
import pytest
from bbbcp.pipeline import REPO, prepare, run_one

FILES = sorted(glob.glob(str(REPO / "results/raw/**/seed*.npz"), recursive=True))


@pytest.fixture(scope="module")
def fresh_file():
    # tiny RF run, regenerated if missing (fast)
    return run_one("bbbp", "scaffold", "rf", "ce", seed=99)


def _check(path):
    z = np.load(path)
    tr, va, ca = set(z["idx_train"]), set(z["idx_val"]), set(z["idx_cal"])
    assert not (tr & va) and not (tr & ca) and not (va & ca)
    if "__to__" not in str(path):
        te = set(z["idx_test"])
        assert not (te & tr) and not (te & va) and not (te & ca)
    assert len(z["y_cal"]) == len(z["idx_cal"]) and len(z["p_cal"]) == len(z["y_cal"])
    assert len(z["p_test"]) == len(z["y_test"])


def test_fresh_run_is_leak_free(fresh_file):
    _check(fresh_file)


def test_all_cached_runs_leak_free():
    for f in FILES:
        _check(f)


def test_external_test_has_no_source_overlap():
    for f in [f for f in FILES if "__to__" in f][:3]:
        src, ext = f.split("results/raw/")[1].split("/")[0].split("__to__")
        z = np.load(f)
        s = set(prepare(src)["df"].conn_key)
        e = prepare(ext)["df"].conn_key.values[z["idx_test"]]
        assert not (set(e) & s)


def test_deterministic_rf_rerun(tmp_path):
    from bbbcp.models.baselines import train_rf
    d = prepare("bbbp"); X, y = d["X"][:600], d["df"].y.values[:600]
    m1, _ = train_rf(X, y, seed=5, n_estimators=50); m2, _ = train_rf(X, y, seed=5, n_estimators=50)
    assert np.allclose(m1.predict_proba(X[:50]), m2.predict_proba(X[:50]))


def test_evaluation_deterministic(fresh_file):
    from bbbcp.evaluate import evaluate_file
    a = evaluate_file(fresh_file, methods=("marginal", "mondrian", "sabm"))
    b = evaluate_file(fresh_file, methods=("marginal", "mondrian", "sabm"))
    assert a.drop(columns=[]).equals(b)
