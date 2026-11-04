import inspect
import numpy as np
import pytest
from bbbcp.data import splits


def test_split_functions_take_no_labels():
    for fn in (splits.random_split, splits.scaffold_balanced_split, splits.cluster_holdout_split,
               splits.make_split):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"y", "labels", "label", "targets"}, fn.__name__


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_random_split_disjoint_complete_deterministic(seed):
    a = splits.random_split(1000, seed); b = splits.random_split(1000, seed)
    splits.assert_disjoint(a, 1000)
    for k in splits.PARTS:
        assert np.array_equal(a[k], b[k])


def test_scaffold_split_on_synthetic_groups():
    rng = np.random.default_rng(0)
    scaf = np.array([f"s{rng.integers(0, 200)}" for _ in range(2000)], dtype=object)
    sp = splits.make_split("scaffold", 3, scaf)
    for other in ("val", "cal", "test"):
        splits.assert_scaffold_disjoint(sp, scaf, ("train", other))
    splits.assert_scaffold_disjoint(sp, scaf, ("cal", "test"))
