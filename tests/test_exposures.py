import numpy as np
import pandas as pd

from erm.data import Universe
from erm.exposures import ewma_weights, standardize, style_exposures


def test_ewma_weights_sum_and_order():
    w = ewma_weights(100, 20)
    assert np.isclose(w.sum(), 1)
    assert np.all(np.diff(w) > 0)          # newest observation weighs most
    assert np.isclose(w[-21] / w[-1], 0.5)  # half-life of 20 steps


def test_standardize_moments():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(500) * 3 + 2)
    cap = pd.Series(np.exp(rng.standard_normal(500)))
    z = standardize(x, cap)
    w = cap / cap.sum()
    assert abs(float((w * z).sum())) < 1e-10
    assert np.isclose(z.std(), 1)


def test_exposure_moments(sim):
    uni, _ = sim
    d = uni.month_ends()[20]
    X = style_exposures(uni, d)
    w = uni.mcap.loc[d] / uni.mcap.loc[d].sum()
    assert np.allclose(X.mul(w, axis=0).sum(), 0, atol=1e-10)
    assert np.allclose(X.std(), 1, atol=1e-10)
    assert X.notna().all().all()


def test_exposures_are_point_in_time(sim):
    uni, _ = sim
    d = uni.month_ends()[20]
    base = style_exposures(uni, d)
    pos = uni.dates.get_loc(d)
    rets = uni.returns.copy()
    rets.iloc[pos + 1:] = rets.iloc[pos + 1:] * 5 + 0.01
    mcap = uni.mcap.copy()
    mcap.iloc[pos + 1:] *= 3
    shocked = Universe(rets, mcap, uni.turnover, uni.industry, uni.fundamentals)
    pd.testing.assert_frame_equal(base, style_exposures(shocked, d))
