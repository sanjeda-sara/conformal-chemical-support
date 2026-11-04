"""Tests for the v1.3 additions (SOTA baselines, ablations, metrics, splits, loaders)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bbbcp.conformal import calibrators as C
from bbbcp.conformal import sota as S
from bbbcp.conformal.scores import score_matrix
from bbbcp.metrics.conditional import worst_slab_coverage
from bbbcp.uncertainty.selective import conformal_selection, temperature_scale

REPO = Path(__file__).resolve().parents[1]


def _toy(n, rng, pi0=0.25, shift=1.2):
    y = (rng.uniform(size=n) > pi0).astype(int)
    logit = (2 * y - 1) * shift + np.log((1 - pi0) / pi0) + rng.normal(size=n)
    p1 = 1 / (1 + np.exp(-logit))
    return np.column_stack([1 - p1, p1]), y


def _sets(method, P_cal, y_cal, P_test, kind="lac", alpha=0.1, **ctx):
    Sc = score_matrix(P_cal, kind); St = score_matrix(P_test, kind)
    s = Sc[np.arange(len(y_cal)), y_cal]
    fn = C.CALIBRATORS.get(method) or S.SOTA[method]
    return C.predict_sets(St, fn(s, y_cal, St, alpha, P_cal=P_cal, P_test=P_test, **ctx))


def test_margin_equals_lac_binary():
    rng = np.random.default_rng(1)
    Pc, yc = _toy(300, rng); Pt, _ = _toy(200, rng)
    b_c, b_t = rng.integers(0, 3, 300), rng.integers(0, 3, 200)
    for m in ("marginal", "mondrian", "mondrian_x", "sabm"):
        a = _sets(m, Pc, yc, Pt, "lac", bin_cal=b_c, bin_test=b_t)
        b = _sets(m, Pc, yc, Pt, "margin", bin_cal=b_c, bin_test=b_t)
        assert (a == b).all(), m


def test_posthoc_logit_adjustment_and_temperature_invariance():
    """Prop. 3: Mondrian-LAC and SABM-LAC are invariant to per-label monotone maps
    (logit adjustment, temperature scaling); marginal-LAC is not in general."""
    rng = np.random.default_rng(2)
    Pc, yc = _toy(400, rng); Pt, _ = _toy(300, rng)
    b_c, b_t = rng.integers(0, 3, 400), rng.integers(0, 3, 300)
    prior = np.array([0.25, 0.75])
    la = lambda P: (P / prior) / (P / prior).sum(1, keepdims=True)
    for m in ("mondrian", "sabm", "mondrian_x"):
        assert (_sets(m, Pc, yc, Pt, bin_cal=b_c, bin_test=b_t) ==
                _sets(m, la(Pc), yc, la(Pt), bin_cal=b_c, bin_test=b_t)).all()
    Pv, yv = _toy(200, rng, shift=0.6)
    Pc2, _ = temperature_scale(Pv, yv, Pc); Pt2, _ = temperature_scale(Pv, yv, Pt)
    assert (_sets("mondrian", Pc, yc, Pt) == _sets("mondrian", Pc2, yc, Pt2)).all()
    changed = any((_sets("marginal", Pc, yc, Pt, alpha=a) != _sets("marginal", la(Pc), yc, la(Pt), alpha=a)).any()
                  for a in (0.05, 0.1, 0.2))
    assert changed


def test_rc3p_equals_mondrian_when_top1_error_large():
    rng = np.random.default_rng(3)
    Pc, yc = _toy(300, rng, shift=0.2); Pt, _ = _toy(200, rng, shift=0.2)   # weak model: top-1 error > alpha
    assert (_sets("rc3p", Pc, yc, Pt, alpha=0.05) == _sets("mondrian", Pc, yc, Pt, alpha=0.05)).all()


@pytest.mark.parametrize("method", ["rlcp_sim", "mondrian_ssbc", "rc3p"])
def test_classwise_coverage_methods(method):
    rng = np.random.default_rng(4)
    covs = []
    for r in range(60):
        Pc, yc = _toy(400, rng); Pt, yt = _toy(400, rng)
        sc, st = rng.uniform(size=400), rng.uniform(size=400)
        sets = _sets(method, Pc, yc, Pt, sim_cal=sc, sim_test=st, rng=np.random.default_rng(r))
        covs.append(sets[np.arange(400), yt][yt == 0].mean())
    assert np.mean(covs) >= 0.88


def test_macro_coverage_guarantee():
    rng = np.random.default_rng(5)
    macro = []
    for r in range(80):
        Pc, yc = _toy(300, rng, pi0=0.1); Pt, yt = _toy(2000, rng, pi0=0.1)
        sets = _sets("macro_cov", Pc, yc, Pt)
        hit = sets[np.arange(len(yt)), yt]
        macro.append(0.5 * (hit[yt == 0].mean() + hit[yt == 1].mean()))
    assert np.mean(macro) >= 0.895


def test_pas_marginal_coverage():
    rng = np.random.default_rng(6)
    cov = []
    for r in range(80):
        Pc, yc = _toy(300, rng); Pt, yt = _toy(1000, rng)
        T = S.pas(None, yc, None, 0.1, P_cal=Pc, P_test=Pt, prior=np.array([0.25, 0.75]))
        sets = C.predict_sets(S.pas.S_override, T)
        cov.append(sets[np.arange(len(yt)), yt].mean())
    assert np.mean(cov) >= 0.895


def test_conformal_selection_fdr():
    rng = np.random.default_rng(7)
    fdr = []
    for r in range(100):
        Pc, yc = _toy(400, rng); Pt, yt = _toy(400, rng)
        fdr.append(conformal_selection(Pc, yc, Pt, yt, target=1, q=0.2)["fdr"])
    assert np.mean(fdr) <= 0.2 + 0.02


def test_worst_slab_le_overall():
    rng = np.random.default_rng(8)
    F = rng.normal(size=(500, 5)); hit = rng.uniform(size=500) < 0.9
    assert worst_slab_coverage(F, hit) <= hit.mean() + 1e-12


def test_new_splits_disjoint_and_deterministic():
    from bbbcp.data.splits import assert_disjoint, butina_holdout_split, make_split
    rng = np.random.default_rng(9)
    scaf = np.array([f"s{k}" for k in rng.integers(0, 120, 1000)], dtype=object)
    a = make_split("scaffold_det", 0, scaf); b = make_split("scaffold_det", 5, scaf)
    assert all((a[k] == b[k]).all() for k in ("train", "val", "cal", "test"))
    sp = make_split("random5025", 0, scaf)
    assert len(sp["val"]) == 0 and abs(len(sp["train"]) - 500) <= 1
    lab = rng.integers(0, 80, 1000)
    sp = butina_holdout_split(lab, 0)
    assert_disjoint(sp, 1000)
    assert not (set(lab[sp["test"]]) & set(lab[sp["train"]]))


@pytest.mark.parametrize("name,minority", [("bace", 1), ("clintox", 1), ("tox21_sr_are", 1), ("bbbp_raw", 0)])
def test_extra_loaders(name, minority):
    from bbbcp.data import datasets
    fn = {"bace": "bace.csv", "clintox": "clintox.csv", "tox21_sr_are": "tox21.csv", "bbbp_raw": "BBBP.csv"}[name]
    if not (datasets.RAW / fn).exists():
        pytest.skip("raw file not downloaded")
    df, audit = datasets.load(name)
    assert audit["minority_label"] == minority and len(df) > 1000


def test_v12_stage_b_unchanged_for_untouched_methods():
    """The v1.3 evaluator with the v1.2 profile reproduces the cached v1.2 pilot numbers for
    every method whose definition did not change (all but SABM, whose n_req rule was frozen
    to eps = alpha on 2026-10-09)."""
    from bbbcp.evaluate import evaluate_file
    from bbbcp.pipeline import prepare
    p = REPO / "results/raw/bbbp/random/rf_ce/seed0.npz"
    ref = REPO / "results/processed/pilot_rf_conformal.csv"
    if not (p.exists() and ref.exists()):
        pytest.skip("pilot cache not present")
    z = np.load(p); d = prepare("bbbp")
    new = evaluate_file(p, extra_meta={"X_cal": d["X"][z["idx_cal"]], "X_test": d["X"][z["idx_test"]]})
    old = pd.read_csv(ref)
    old = old[(old.dataset == "bbbp") & (old.regime == "random") & (old.seed == 0) & (old.loss == "ce")]
    for m in ("marginal", "mondrian", "mondrian_pac", "mondrian_x", "weighted"):
        for kind in ("lac", "aps"):
            a = new[(new.method == m) & (new.score == kind) & np.isclose(new.alpha, 0.1)].coverage.values
            b = old[(old.method == m) & (old.score == kind) & np.isclose(old.alpha, 0.1)].coverage.values
            assert len(a) == 1 and len(b) == 1 and math.isclose(a[0], b[0], abs_tol=1e-12), (m, kind)


# ----------------------------------------------------------------- v1.4 (review response)
def test_sabm_equals_fixed_cells_when_no_merging_needed():
    """Honesty test for the paper's central caveat: when every (class, bin) cell already has
    n_req calibration points, SABM IS fixed class x support Mondrian, bit for bit.  If this
    ever fails, the claim in Sec. 'When does adaptive merging matter' must be re-checked."""
    rng = np.random.default_rng(11)
    Pc, yc = _toy(900, rng); Pt, _ = _toy(400, rng)
    b_c, b_t = rng.integers(0, 3, 900), rng.integers(0, 3, 400)
    for c in (0, 1):
        for b in range(3):
            assert ((yc == c) & (b_c == b)).sum() >= 11      # every cell above n_req(0.1)
    a = _sets("sabm", Pc, yc, Pt, bin_cal=b_c, bin_test=b_t)
    b = _sets("mondrian_x", Pc, yc, Pt, bin_cal=b_c, bin_test=b_t)
    assert (a == b).all()


def test_sabm_avoids_degeneracy_when_cells_are_too_small():
    """The converse: with cells below n_req, the FIXED partition returns full sets (infinite
    thresholds) while SABM merges and stays informative.  This is SABM's stated benefit."""
    rng = np.random.default_rng(12)
    Pc, yc = _toy(60, rng); Pt, _ = _toy(400, rng)
    b_c, b_t = rng.integers(0, 8, 60), rng.integers(0, 8, 400)
    fixed = _sets("mondrian_x", Pc, yc, Pt, alpha=0.05, bin_cal=b_c, bin_test=b_t)
    adapt = _sets("sabm", Pc, yc, Pt, alpha=0.05, bin_cal=b_c, bin_test=b_t)
    assert fixed.sum(1).mean() > adapt.sum(1).mean()          # fixed defers more
    assert (fixed.sum(1) == 2).mean() > (adapt.sum(1) == 2).mean()


def test_primary_family_is_small_and_holm_is_applied_once():
    """The primary inferential family must stay small enough to be rejectable at 20 seeds."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from analyze_v13 import PRIMARY_CONTRASTS, paired_primary
    rng = np.random.default_rng(13)
    rows = []
    for st in ("bbbp", "b3db"):
        for rg in ("random", "scaffold"):
            for mdl in ("rf", "gin"):
                for seed in range(20):
                    base = rng.uniform(0.70, 0.80)
                    for meth, bump in (("mondrian", 0.0), ("mondrian_x", 0.15), ("sabm", 0.15)):
                        rows.append({"dataset": st, "regime": rg, "model_name": mdl, "loss": "ce",
                                     "score": "lac", "alpha": 0.10, "method": meth, "seed": seed,
                                     "external": np.nan, "cov_min_bin0": base + bump})
    out = paired_primary(pd.DataFrame(rows))
    assert len(out) == len(PRIMARY_CONTRASTS) * 4 * 2            # contrasts x settings x models
    assert out.family_size.nunique() == 1                        # one family, corrected once
    gain = out[out.comparison == "mondrian_x - mondrian"]
    assert gain["significant_holm_0.05"].all()                   # 20 seeds => detectable
    noop = out[out.comparison == "sabm - mondrian_x"]
    assert np.allclose(noop.mean_diff, 0)                        # and the no-op shows as zero


def test_caches_rebuild_when_unreadable(tmp_path, monkeypatch):
    """A corrupt or version-incompatible derived cache must trigger a rebuild, not an error.
    Regression test for the 9 test failures seen under pandas 2.2.3."""
    from bbbcp import pipeline
    bad = tmp_path / "broken.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.warns(UserWarning, match="rebuilding unreadable cache"):
        assert pipeline._load_cache(bad) is None
    assert pipeline._load_cache(tmp_path / "absent.pkl") is None


# ----------------------------------------------------------------- v1.3.1 (review response)
def test_matched_operating_point_interpolation():
    """metrics/matched.py must (a) recover a method's own metrics when matched to itself,
    and (b) report achievable=False when the requested operating point is out of range."""
    from bbbcp.metrics.matched import compare_at_matched, match_at, operating_curve
    rng = np.random.default_rng(11)
    Pc, yc = _toy(400, rng); Pt, yt = _toy(400, rng)
    bt = rng.integers(0, 3, 400)
    Sc = score_matrix(Pc, "lac"); St = score_matrix(Pt, "lac")
    s = Sc[np.arange(len(yc)), yc]
    grid = np.round(np.arange(0.02, 0.35, 0.02), 3)
    curves = {n: operating_curve(lambda a, f=f: C.predict_sets(St, f(s, yc, St, a)), grid, yt, bt)
              for n, f in (("marginal", C.marginal), ("mondrian", C.mondrian))}
    cmp_ = compare_at_matched(curves, "mondrian", 0.10, by="full_rate")
    self_row = cmp_[cmp_.method == "mondrian"].iloc[0]
    ref = curves["mondrian"]
    i = int(np.argmin(np.abs(ref.alpha.values - 0.10)))
    assert self_row.achievable
    assert abs(self_row.full_rate - ref.iloc[i].full_rate) < 1e-6
    assert abs(self_row.cov_minority - ref.iloc[i].cov_minority) < 1e-6
    _, ok = match_at(curves["mondrian"], 1.5, by="full_rate")       # deferral > 1 is impossible
    assert not ok


def test_sabm_reduces_to_fixed_when_every_cell_is_large():
    """The paper's central honesty claim: with enough calibration points per cell, the
    adaptive rule never merges and SABM IS the fixed class x support partition."""
    rng = np.random.default_rng(12)
    Pc, yc = _toy(4000, rng); Pt, _ = _toy(500, rng)
    b_c, b_t = rng.integers(0, 3, 4000), rng.integers(0, 3, 500)
    a = _sets("sabm", Pc, yc, Pt, bin_cal=b_c, bin_test=b_t)
    b = _sets("mondrian_x", Pc, yc, Pt, bin_cal=b_c, bin_test=b_t)
    assert (a == b).all()
    assert float(C.sabm.last_level.mean()) == 1.0


def test_sabm_never_returns_an_infinite_threshold_when_the_class_has_enough_points():
    """And the claim that earns it a place: a fixed fine partition degenerates to full sets
    on sparse cells, while the adaptive rule merges and stays informative."""
    rng = np.random.default_rng(13)
    Pc, yc = _toy(300, rng, pi0=0.08); Pt, _ = _toy(300, rng, pi0=0.08)
    b_c, b_t = rng.integers(0, 10, 300), rng.integers(0, 10, 300)   # 10 bins, ~24 minority total
    Sc = score_matrix(Pc, "lac"); St = score_matrix(Pt, "lac")
    s = Sc[np.arange(len(yc)), yc]
    T_fixed = C.mondrian_x(s, yc, St, 0.1, bin_cal=b_c, bin_test=b_t)
    T_adapt = C.sabm(s, yc, St, 0.1, bin_cal=b_c, bin_test=b_t)
    assert np.isinf(T_fixed).any(), "fixed partition should degenerate in this regime"
    assert not np.isinf(T_adapt).any(), "adaptive partition should not"
    assert C.predict_sets(St, T_adapt).sum(1).mean() < C.predict_sets(St, T_fixed).sum(1).mean()
