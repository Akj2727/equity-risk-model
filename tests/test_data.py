import numpy as np
import pandas as pd

from erm.data.real import universe_from_frames
from erm.evaluation import make_test_portfolios


def _frames():
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(np.cumsum(0.01 * rng.standard_normal((300, 4)), axis=0)),
                         index=dates, columns=tickers)
    close.iloc[:200, 3] = np.nan          # DDD has too little history
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    shares = pd.Series(1e8, index=tickers)
    industry = pd.Series(["Tech", "Tech", "Energy", "Energy"], index=tickers)
    fund = pd.DataFrame({"date": ["2020-03-02", "2020-03-02"], "ticker": ["AAA", "BBB"],
                         "roe": [0.2, 0.1]})
    return close, volume, shares, industry, fund


def test_universe_from_frames_filters_and_aligns():
    close, volume, shares, industry, fund = _frames()
    uni = universe_from_frames(close, volume, shares, industry, fund, report_lag_days=30)
    assert list(uni.tickers) == ["AAA", "BBB", "CCC"]
    assert uni.returns.shape == uni.mcap.shape == uni.turnover.shape
    assert np.allclose(uni.turnover.iloc[5], 0.01)


def test_fundamentals_respect_reporting_lag():
    close, volume, shares, industry, fund = _frames()
    uni = universe_from_frames(close, volume, shares, industry, fund, report_lag_days=30)
    roe = uni.fundamentals["roe"]["AAA"]
    assert roe.loc[:"2020-03-31"].isna().all()
    assert roe.loc["2020-04-01":].eq(0.2).all()


def test_test_portfolio_budgets(sim):
    uni, _ = sim
    P = make_test_portfolios(uni, uni.month_ends()[10], seed=0)
    for c in P.columns:
        s = P[c].sum()
        if c == "benchmark" or c.startswith("long_"):
            assert np.isclose(s, 1)
        else:
            assert np.isclose(s, 0, atol=1e-12)


def test_price_signal_runs_without_fundamentals(sim):
    from erm.alpha import price_signal
    from erm.data import Universe
    uni, _ = sim
    bare = Universe(uni.returns, uni.mcap, uni.turnover, uni.industry)
    s = price_signal(bare, uni.month_ends()[-2])
    assert s.notna().all() and np.isclose(s.mean(), 0, atol=0.1)
