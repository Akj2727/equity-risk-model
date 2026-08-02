"""Synthetic equity universe with a known factor structure.

The point of simulating is that the truth is known, so every model can be
checked against it. The data-generating process has:

- a market factor with stochastic volatility and a calm/stress regime
- industry factors
- style factors driven by observable characteristics (size, value, quality, growth)
- one hidden factor that no fundamental descriptor captures
- heteroskedastic specific returns (small caps are noisier)
- a slowly moving expected residual return per stock, part of which lines up
  with the hidden factor loading. A noisy signal of it is published each month.

The last point is what makes alpha/risk-model misalignment show up later.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import Universe

DAYS_PER_YEAR = 252
DAYS_PER_MONTH = 21


@dataclass
class SimConfig:
    n_assets: int = 300
    n_years: int = 8
    n_industries: int = 10
    seed: int = 7
    # annualised factor premia / vols
    market_drift: float = 0.07
    market_vol: float = 0.16
    industry_vol: float = 0.08
    style_premia: tuple = (-0.015, 0.02, 0.02, 0.0)   # size, value, quality, growth
    style_vols: tuple = (0.04, 0.04, 0.03, 0.03)
    hidden_vol: float = 0.06
    # stress regime
    p_enter_stress: float = 0.004
    p_leave_stress: float = 0.03
    stress_mult: float = 2.2
    # expected residual return
    alpha_dispersion: float = 0.06     # annual cross-sectional std of true alpha
    alpha_hidden_corr: float = 0.5     # how much alpha lines up with the hidden loading
    alpha_persistence: float = 0.8     # monthly AR(1) coefficient
    signal_noise: float = 1.0          # noise std relative to true alpha std


@dataclass
class SimTruth:
    factor_returns: pd.DataFrame
    beta: pd.Series
    hidden_loading: pd.Series
    alpha: pd.DataFrame       # true expected daily residual return, indexed by month end it applies after
    signal: pd.DataFrame      # observable signal, indexed by the month end it is published
    regime: pd.Series


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _ar1_panel(rng, n_months, n_assets, rho, start=None):
    out = np.empty((n_months, n_assets))
    out[0] = rng.standard_normal(n_assets) if start is None else start
    scale = np.sqrt(1 - rho ** 2)
    for m in range(1, n_months):
        out[m] = rho * out[m - 1] + scale * rng.standard_normal(n_assets)
    return out


def simulate_universe(cfg: SimConfig | None = None) -> tuple[Universe, SimTruth]:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, K_ind = cfg.n_assets, cfg.n_industries
    n_months = cfg.n_years * 12
    T = n_months * DAYS_PER_MONTH
    dates = pd.bdate_range("2017-01-02", periods=T)
    tickers = pd.Index([f"S{i:03d}" for i in range(N)])
    month_of_day = np.repeat(np.arange(n_months), DAYS_PER_MONTH)

    industry = pd.Series(rng.integers(0, K_ind, N), index=tickers).map(lambda k: f"IND{k:02d}")
    ind_codes = pd.Categorical(industry).codes

    # stock-level constants
    beta = 1.0 + 0.25 * rng.standard_normal(N)
    hidden = rng.standard_normal(N)
    log_cap0 = np.log(5e9) + 1.2 * rng.standard_normal(N)

    # monthly characteristics (AR(1), standardised scale)
    bp = _ar1_panel(rng, n_months, N, 0.97)
    ep = 0.5 * bp + np.sqrt(0.75) * _ar1_panel(rng, n_months, N, 0.95)
    roe = _ar1_panel(rng, n_months, N, 0.96)
    growth = _ar1_panel(rng, n_months, N, 0.85)

    # true alpha and its published signal
    latent = _ar1_panel(rng, n_months, N, cfg.alpha_persistence)
    c = cfg.alpha_hidden_corr
    alpha_z = c * hidden[None, :] + np.sqrt(1 - c ** 2) * latent
    alpha_ann = cfg.alpha_dispersion * alpha_z
    signal = alpha_z + cfg.signal_noise * rng.standard_normal((n_months, N))

    # regime path and stochastic vol
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        p = cfg.p_leave_stress if regime[t - 1] else cfg.p_enter_stress
        regime[t] = 1 - regime[t - 1] if rng.random() < p else regime[t - 1]
    vol_mult = np.where(regime == 1, cfg.stress_mult, 1.0)
    # mild GARCH-like persistence on top of the regime
    h = np.ones(T)
    shock = rng.standard_normal(T)
    for t in range(1, T):
        h[t] = 0.05 + 0.03 * shock[t - 1] ** 2 + 0.92 * h[t - 1]
    vol_mult = vol_mult * np.sqrt(h / h.mean())

    sd = 1 / np.sqrt(DAYS_PER_YEAR)
    mkt = cfg.market_drift / DAYS_PER_YEAR + cfg.market_vol * sd * vol_mult * rng.standard_normal(T)
    ind_f = cfg.industry_vol * sd * vol_mult[:, None] * rng.standard_normal((T, K_ind))
    ind_f -= ind_f.mean(axis=1, keepdims=True)
    prem = np.array(cfg.style_premia) / DAYS_PER_YEAR
    svol = np.array(cfg.style_vols) * sd
    style_f = prem + svol * vol_mult[:, None] * rng.standard_normal((T, 4))
    hid_f = cfg.hidden_vol * sd * vol_mult * rng.standard_normal(T)

    returns = np.empty((T, N))
    log_cap = log_cap0.copy()
    caps = np.empty((T, N))
    X_style = None
    spec_vol = None
    for t in range(T):
        m = month_of_day[t]
        if t == 0 or month_of_day[t - 1] != m:
            size_z = _zscore(log_cap)
            X_style = np.column_stack([size_z, _zscore(bp[m] + ep[m]), _zscore(roe[m]), _zscore(growth[m])])
            spec_vol = (0.18 + 0.12 * np.exp(-0.5 * np.clip(size_z, -2.5, 2.5))) * sd
        spec = spec_vol * (1 + 0.4 * (vol_mult[t] - 1)) * rng.standard_normal(N)
        r = (beta * mkt[t] + ind_f[t, ind_codes] + X_style @ style_f[t]
             + hidden * hid_f[t] + alpha_ann[m] / DAYS_PER_YEAR + spec)
        r = np.maximum(r, -0.5)
        returns[t] = r
        log_cap = log_cap + np.log1p(r)
        caps[t] = np.exp(log_cap)

    base_turn = np.exp(np.log(0.004) - 0.3 * _zscore(log_cap0) + 0.3 * rng.standard_normal(N))
    turnover = base_turn * np.exp(0.4 * rng.standard_normal((T, N)) + 0.5 * (vol_mult[:, None] - 1))

    ret_df = pd.DataFrame(returns, index=dates, columns=tickers)
    cap_df = pd.DataFrame(caps, index=dates, columns=tickers)
    # market cap known at the close of each day, so shift one day for "as of" use
    turn_df = pd.DataFrame(turnover, index=dates, columns=tickers)

    def daily(panel):
        return pd.DataFrame(panel[month_of_day], index=dates, columns=tickers)

    fundamentals = {
        "book_to_price": daily(np.exp(0.5 * bp) * 0.6),
        "earnings_yield": daily(0.06 + 0.03 * ep),
        "roe": daily(0.14 + 0.08 * roe),
        "sales_growth": daily(0.08 + 0.12 * growth),
    }

    month_ends = pd.DatetimeIndex([dates[(m + 1) * DAYS_PER_MONTH - 1] for m in range(n_months)])
    # signal for month m is published at the end of month m-1
    sig_df = pd.DataFrame(signal[1:], index=month_ends[:-1], columns=tickers)
    alpha_df = pd.DataFrame(alpha_ann[1:] / DAYS_PER_YEAR, index=month_ends[:-1], columns=tickers)

    factor_returns = pd.DataFrame(
        np.column_stack([mkt, style_f, hid_f]),
        index=dates, columns=["market", "size", "value", "quality", "growth", "hidden"],
    )
    truth = SimTruth(
        factor_returns=factor_returns,
        beta=pd.Series(beta, index=tickers),
        hidden_loading=pd.Series(hidden, index=tickers),
        alpha=alpha_df,
        signal=sig_df,
        regime=pd.Series(regime, index=dates),
    )
    uni = Universe(returns=ret_df, mcap=cap_df, turnover=turn_df,
                   industry=industry, fundamentals=fundamentals)
    return uni, truth
