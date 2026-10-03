import numpy as np
import pandas as pd

from erm.fundamental import FundamentalConfig, ewma_cov_nw, factor_covariance, shrink_specific
from erm.statistical import apca, hybrid_model, statistical_model


def test_ewma_cov_without_lags_is_weighted_covariance():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((300, 4))
    S = ewma_cov_nw(F, 1e9, 0)  # huge half-life = equal weights
    assert np.allclose(S, np.cov(F, rowvar=False, ddof=0), atol=1e-6)


def test_factor_covariance_is_psd():
    rng = np.random.default_rng(1)
    F = rng.standard_normal((400, 12)) @ rng.standard_normal((12, 12)) * 0.01
    S = factor_covariance(F, FundamentalConfig())
    assert np.allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > 0


def test_shrinkage_moves_toward_bucket_mean():
    rng = np.random.default_rng(2)
    cap = pd.Series(np.exp(rng.standard_normal(200)))
    sig = pd.Series(np.abs(rng.standard_normal(200)) * 0.02 + 0.01)
    out = shrink_specific(sig, cap, q=0.5, n_buckets=5)
    assert out.std() < sig.std()
    assert out.notna().all()


def test_apca_recovers_factor_span():
    rng = np.random.default_rng(3)
    T, N = 250, 120
    f = rng.standard_normal((T, 2))
    B = rng.standard_normal((N, 2))
    R = f @ B.T + 0.1 * rng.standard_normal((T, N))
    R -= R.mean(axis=0)
    Fr, Bh, U = apca(R, 2)
    # projection of true factors onto estimated ones should be near perfect
    coef, res, *_ = np.linalg.lstsq(Fr, f - f.mean(axis=0), rcond=None)
    r2 = 1 - res.sum() / ((f - f.mean(axis=0)) ** 2).sum()
    assert r2 > 0.98


def test_hybrid_adds_factors_and_never_raises_specific(pipe):
    d = pipe.uni.month_ends()[-2]
    base = pipe.model("fundamental", d)
    hy = pipe.model("hybrid", d)
    assert hy.X.shape[1] == base.X.shape[1] + pipe.hybrid_factors
    assert (hy.D <= base.D + 1e-15).all()


def test_statistical_model_shapes(pipe):
    d = pipe.uni.month_ends()[-2]
    st = statistical_model(pipe.uni.window(d, 252), k=5)
    assert st.X.shape == (len(pipe.uni.tickers), 5)
    assert (st.D > 0).all()
