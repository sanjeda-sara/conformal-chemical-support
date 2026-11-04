"""IID sanity: on exchangeable synthetic data the calibrators meet their guarantees."""
import numpy as np
from bbbcp.conformal import calibrators as C
from bbbcp.conformal.scores import score_matrix


def _draw(rng, n, pi_min=0.15):
    y = (rng.random(n) > pi_min).astype(int)  # class 0 is the minority
    # deliberately miscalibrated model: minority probabilities are under-confident
    logit = np.where(y == 1, rng.normal(1.5, 1.0, n), rng.normal(0.3, 1.0, n))
    p1 = 1 / (1 + np.exp(-logit))
    return np.stack([1 - p1, p1], 1), y


def _run(method, alpha=0.1, reps=400, n_cal=400, n_test=400, kind="lac"):
    rng = np.random.default_rng(123)
    cov, cov0 = [], []
    for _ in range(reps):
        pc, yc = _draw(rng, n_cal); pt, yt = _draw(rng, n_test)
        Sc = score_matrix(pc, kind, rng.random(n_cal))[np.arange(n_cal), yc]
        St = score_matrix(pt, kind, rng.random(n_test))
        bc = np.zeros(n_cal, int); bt = np.zeros(n_test, int)
        sets = C.predict_sets(St, C.CALIBRATORS[method](Sc, yc, St, alpha, bin_cal=bc, bin_test=bt))
        hit = sets[np.arange(n_test), yt]
        cov.append(hit.mean()); cov0.append(hit[yt == 0].mean())
    return np.mean(cov), np.mean(cov0)


def test_marginal_valid_marginally_but_not_for_minority():
    cov, cov0 = _run("marginal")
    assert cov >= 0.89
    assert cov0 < 0.85  # the known failure mode marginal CP has on the minority class


def test_mondrian_valid_per_class():
    cov, cov0 = _run("mondrian")
    assert cov >= 0.89 and cov0 >= 0.89


def test_mondrian_aps_valid_per_class():
    cov, cov0 = _run("mondrian", kind="aps")
    assert cov0 >= 0.89


def test_pac_at_least_mondrian():
    _, c_m = _run("mondrian", reps=100); _, c_p = _run("mondrian_pac", reps=100)
    assert c_p >= c_m - 1e-9


def test_sabm_class_conditional_validity_with_bins():
    """Binned synthetic data where score distributions depend on the support bin;
    SABM keeps per-class coverage >= 1 - alpha."""
    rng = np.random.default_rng(7)
    alpha, reps = 0.1, 300
    covs = {0: [], 1: []}

    def draw(n):
        y = (rng.random(n) > 0.2).astype(int)
        b = rng.integers(0, 3, n)
        shift = np.array([-1.0, 0.0, 1.0])[b]  # low-support bin = worse model
        logit = np.where(y == 1, 1.5 + shift, -0.5 - shift) + rng.normal(0, 1, n)
        p1 = 1 / (1 + np.exp(-logit))
        return np.stack([1 - p1, p1], 1), y, b

    for _ in range(reps):
        pc, yc, bc = draw(300); pt, yt, bt = draw(300)
        Sc = score_matrix(pc, "lac")[np.arange(300), yc]; St = score_matrix(pt, "lac")
        sets = C.predict_sets(St, C.sabm(Sc, yc, St, alpha, bin_cal=bc, bin_test=bt))
        hit = sets[np.arange(300), yt]
        for c in (0, 1):
            covs[c].append(hit[yt == c].mean())
    assert np.mean(covs[0]) >= 0.89 and np.mean(covs[1]) >= 0.89


def test_partition_depends_only_on_counts():
    from bbbcp.conformal.calibrators import _partition_bins
    assert list(_partition_bins(np.array([100, 100, 100]), 60)) == [0, 1, 2]
    assert list(_partition_bins(np.array([10, 10, 100]), 60)) == [0, 0, 0]
    assert list(_partition_bins(np.array([70, 10, 10]), 60)) == [0, 0, 0]
    assert list(_partition_bins(np.array([70, 30, 40]), 60)) == [0, 1, 1]


def test_mondrian_lac_invariant_to_posthoc_logit_adjustment():
    """Proposition (THEORY_NOTES section 7): class-wise monotone transforms leave
    Mondrian-LAC sets unchanged; marginal-LAC sets change."""
    rng = np.random.default_rng(3)
    pc, yc = _draw(rng, 500); pt, _ = _draw(rng, 500)

    def adjust(p, b=1.2):  # logit offset toward the minority class 0
        lo = np.log(p[:, 0] / p[:, 1]) + b
        p0 = 1 / (1 + np.exp(-lo)); return np.stack([p0, 1 - p0], 1)
    out = {}
    for name, (a, b) in {"raw": (pc, pt), "adj": (adjust(pc), adjust(pt))}.items():
        Sc = score_matrix(a, "lac")[np.arange(500), yc]; St = score_matrix(b, "lac")
        out[name] = {m: C.predict_sets(St, C.CALIBRATORS[m](Sc, yc, St, 0.1)) for m in ("marginal", "mondrian")}
    assert np.array_equal(out["raw"]["mondrian"], out["adj"]["mondrian"])
    assert not np.array_equal(out["raw"]["marginal"], out["adj"]["marginal"])
