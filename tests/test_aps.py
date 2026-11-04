import numpy as np
from bbbcp.conformal.scores import aps, lac, raps


def test_lac():
    p = np.array([[0.7, 0.3]])
    assert np.allclose(lac(p), [[0.3, 0.7]])


def test_aps_hand_example_shared_u():
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    u = np.array([0.5, 0.25])
    S = aps(p, u)
    # row 0: label0 top -> 0 + .5*.7 ; label1 -> .7 + .5*.3
    # row 1: label1 top -> 0 + .25*.8 ; label0 -> .8 + .25*.2
    assert np.allclose(S, [[0.35, 0.85], [0.85, 0.2]])


def test_aps_deterministic_variant():
    p = np.array([[0.6, 0.3, 0.1]])
    assert np.allclose(aps(p, None), [[0.6, 0.9, 1.0]])


def test_aps_tie_broken_by_label_index():
    p = np.array([[0.5, 0.5]])
    S = aps(p, np.array([1.0]))
    assert np.allclose(S, [[0.5, 1.0]])


def test_raps_penalty():
    p = np.array([[0.6, 0.3, 0.1]])
    S = raps(p, None, lam=0.1, k_reg=1)
    assert np.allclose(S, [[0.6, 0.9 + 0.1, 1.0 + 0.2]])


def test_aps_same_u_for_all_labels():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1], 100); u = rng.random(100)
    S = aps(p, u)
    # with two labels and a shared U, the top label's score is u*p_top and the other is p_top + u*p_other
    top = p.argmax(1)
    assert np.allclose(S[np.arange(100), top], u * p.max(1))
