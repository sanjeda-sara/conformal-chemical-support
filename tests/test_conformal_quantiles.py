import math
import numpy as np
from bbbcp.conformal.calibrators import (conformal_quantile, pac_rank, n_required, marginal,
                                         mondrian, weighted, sabm)


def test_hand_quantile():
    s = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.5, 0.6, 0.7, 0.8])  # n=9
    # alpha=0.2: k = ceil(10*0.8) = 8 -> 8th smallest = 0.8
    assert conformal_quantile(s, 0.2) == 0.8
    # alpha=0.1: k = ceil(10*0.9) = 9 -> max = 0.9
    assert conformal_quantile(s, 0.1) == 0.9


def test_infinite_when_too_few():
    # n=10, alpha=0.05: k = ceil(11*0.95)=11 > 10 -> must be +inf (legacy returned max)
    assert math.isinf(conformal_quantile(np.arange(10.0), 0.05))
    assert math.isinf(conformal_quantile(np.array([]), 0.1))


def test_pac_rank_more_conservative():
    n, a = 50, 0.1
    k_std = math.ceil((n + 1) * (1 - a))
    k_pac = pac_rank(n, a, 0.1)
    assert k_pac is None or k_pac >= k_std


def test_n_required_holds_for_all_larger_n():
    from scipy.stats import beta
    for a in (0.05, 0.1, 0.2):
        for eps in (None, 0.05):
            n0 = n_required(a, eps)
            e = a if eps is None else eps
            for n in range(n0, n0 + 300):
                k = math.ceil((n + 1) * (1 - a) - 1e-12)
                assert beta.cdf(1 - a - e, k, n + 1 - k) <= 0.10 + 1e-12


def test_frozen_n_req_values():
    # frozen 2026-10-09, before confirmatory seeds 100-119
    assert [n_required(a) for a in (0.05, 0.1, 0.2)] == [22, 11, 5]
    assert [n_required(a, 0.05) for a in (0.05, 0.1, 0.2)] == [22, 60, 105]


def test_weighted_equals_marginal_with_unit_weights():
    rng = np.random.default_rng(0)
    s = rng.random(200); y = rng.integers(0, 2, 200); St = rng.random((50, 2))
    Tm = marginal(s, y, St, 0.1)
    Tw = weighted(s, y, St, 0.1, w_cal=np.ones(200), w_test=np.ones(50))
    assert np.allclose(Tm, Tw)


def test_sabm_reduces_to_mondrian_with_one_bin():
    rng = np.random.default_rng(1)
    s = rng.random(300); y = rng.integers(0, 2, 300); St = rng.random((40, 2))
    b0 = np.zeros(300, int); bt = np.zeros(40, int)
    assert np.allclose(sabm(s, y, St, 0.1, bin_cal=b0, bin_test=bt), mondrian(s, y, St, 0.1))
