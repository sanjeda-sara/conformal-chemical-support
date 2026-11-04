"""Real-data checks on BBBP and B3DB (standardized)."""
import numpy as np
import pytest
from bbbcp.data.splits import assert_disjoint, assert_scaffold_disjoint, make_split
from bbbcp.pipeline import prepare


@pytest.mark.parametrize("name", ["bbbp", "b3db"])
@pytest.mark.parametrize("regime", ["scaffold", "cluster"])
def test_real_scaffold_disjoint(name, regime):
    d = prepare(name)
    scaf = d["df"].scaffold.values
    for seed in range(3):
        sp = make_split(regime, seed, scaf, d["clusters"])
        assert_disjoint(sp, len(scaf))
        for a, b in [("train", "val"), ("train", "cal"), ("train", "test"), ("cal", "test")]:
            assert_scaffold_disjoint(sp, scaf, (a, b))


@pytest.mark.parametrize("name", ["bbbp", "b3db"])
def test_no_duplicate_molecules_after_standardization(name):
    df = prepare(name)["df"]
    assert not df.inchikey.duplicated().any()
    assert not df.smiles.duplicated().any()
