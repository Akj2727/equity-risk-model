import numpy as np

from erm.regression import restriction_matrix, wls_with_constraint


def _design(rng, n=200, k_ind=4, k_style=3):
    ind = rng.integers(0, k_ind, n)
    D = np.eye(k_ind)[ind]
    S = rng.standard_normal((n, k_style))
    X = np.column_stack([np.ones(n), D, S])
    return X, ind


def test_recovers_constrained_factor_returns_without_noise():
    rng = np.random.default_rng(1)
    X, ind = _design(rng)
    w = np.exp(rng.standard_normal(X.shape[0]))
    caps = np.array([w[ind == k].sum() for k in range(4)])
    g = rng.standard_normal(1 + 3 + 3)
    f_true = restriction_matrix(1, caps, 3) @ g
    r = X @ f_true
    f, u, r2 = wls_with_constraint(r, X, w, 1, 4, caps)
    assert np.allclose(f, f_true)
    assert np.allclose(u, 0)
    assert np.isclose(r2, 1)


def test_industry_constraint_holds_with_noise():
    rng = np.random.default_rng(2)
    X, ind = _design(rng)
    w = np.exp(rng.standard_normal(X.shape[0]))
    caps = np.array([w[ind == k].sum() for k in range(4)])
    r = rng.standard_normal(X.shape[0]) * 0.02
    f, u, _ = wls_with_constraint(r, X, w, 1, 4, caps)
    assert abs(caps @ f[1:5]) < 1e-12


def test_pipeline_regression_tracks_true_factors(sim, pipe):
    _, truth = sim
    fr = pipe.reg.factor_returns
    tf = truth.factor_returns.reindex(fr.index)
    assert np.corrcoef(fr["country"], tf["market"])[0, 1] > 0.9
    assert np.corrcoef(fr["value"], tf["value"])[0, 1] > 0.5
