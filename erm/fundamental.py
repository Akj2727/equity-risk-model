"""Barra-style fundamental factor risk model.

Factor covariance
    EWMA with separate half-lives for volatility (84d) and correlation (504d),
    Newey-West correction for autocorrelation (2 lags), then a volatility
    regime adjustment that rescales the whole matrix when recent factor
    returns have been large relative to forecasts.

Specific risk
    EWMA of squared residuals (84d), Bayesian shrinkage toward the cap-weighted
    mean of the stock's size decile, and a separate volatility regime
    adjustment.

All forecasts are daily estimates scaled to the horizon (21 days).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exposures import full_exposure_matrix
from .regression import RegressionResult
from .riskmodel import RiskModel


@dataclass
class FundamentalConfig:
    vol_halflife: int = 84
    corr_halflife: int = 504
    nw_lags: int = 2
    max_history: int = 1008
    vra_halflife: int = 42
    spec_halflife: int = 84
    shrink_q: float = 0.1
    n_size_buckets: int = 10
    horizon: int = 21
    min_spec_obs: int = 42
    use_vra: bool = True


def ewma_cov_nw(F: np.ndarray, halflife: float, lags: int) -> np.ndarray:
    """EWMA covariance of the rows of F (oldest first) with Newey-West lags."""
    T = F.shape[0]
    lam = 0.5 ** (1 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()
    mu = (w[:, None] * F).sum(axis=0)
    Z = F - mu
    S = (w[:, None] * Z).T @ Z
    for l in range(1, lags + 1):
        wl = w[l:] / w[l:].sum()
        G = (wl[:, None] * Z[l:]).T @ Z[:-l]
        S = S + (1 - l / (lags + 1)) * (G + G.T)
    return S


def factor_covariance(F: np.ndarray, cfg: FundamentalConfig) -> np.ndarray:
    """Daily factor covariance: vols from the short half-life, correlations from the long one."""
    S_vol = ewma_cov_nw(F, cfg.vol_halflife, cfg.nw_lags)
    S_corr = ewma_cov_nw(F, cfg.corr_halflife, cfg.nw_lags)
    d_corr = np.sqrt(np.clip(np.diag(S_corr), 1e-16, None))
    C = S_corr / np.outer(d_corr, d_corr)
    v = np.sqrt(np.clip(np.diag(S_vol), 1e-16, None))
    S = C * np.outer(v, v)
    # tidy up tiny negative eigenvalues from the NW terms
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * vals) @ vecs.T


def _recursive_ewma_var(x: np.ndarray, halflife: float, min_obs: int) -> np.ndarray:
    """Running EWMA of x**2 per column, value at row t uses rows <= t. NaNs are skipped."""
    lam = 0.5 ** (1 / halflife)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    num = np.zeros(N)
    den = np.zeros(N)
    cnt = np.zeros(N)
    for t in range(T):
        ok = np.isfinite(x[t])
        num = lam * num
        den = lam * den
        num[ok] += x[t, ok] ** 2
        den[ok] += 1.0
        cnt[ok] += 1
        good = cnt >= min_obs
        out[t, good] = num[good] / den[good]
    return out


def bias_multiplier(standardised: np.ndarray, weights: np.ndarray | None, halflife: float) -> np.ndarray:
    """Volatility regime multiplier lambda_t from cross-sectional bias statistics.

    standardised[t, k] = realised return at t / forecast vol made at t-1.
    lambda_t^2 is the EWMA of the cross-sectional mean of squares.
    """
    sq = standardised ** 2
    if weights is None:
        ok = np.isfinite(sq)
        n = ok.sum(axis=1)
        b2 = np.where(n > 0, np.where(ok, sq, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)
    else:
        w = np.where(np.isfinite(sq), weights, 0.0)
        w = w / np.where(w.sum(axis=1, keepdims=True) > 0, w.sum(axis=1, keepdims=True), np.nan)
        b2 = np.nansum(w * np.where(np.isfinite(sq), sq, 0.0), axis=1)
    lam = 0.5 ** (1 / halflife)
    out = np.full(len(b2), np.nan)
    num = den = 0.0
    for t, b in enumerate(b2):
        num *= lam
        den *= lam
        if np.isfinite(b):
            num += b
            den += 1
        out[t] = np.sqrt(num / den) if den > 0 else np.nan
    return out


def shrink_specific(sigma: pd.Series, cap: pd.Series, q: float, n_buckets: int) -> pd.Series:
    """Bayesian shrinkage of specific vol toward the cap-weighted size-decile mean."""
    ok = cap.notna()
    buckets = pd.qcut(cap[ok].rank(method="first"), n_buckets, labels=False)
    buckets = buckets.reindex(sigma.index)
    out = sigma.copy()
    for b in range(n_buckets):
        m = buckets == b
        s = sigma[m]
        c = cap[m]
        have = s.notna()
        if have.sum() == 0:
            continue
        prior = float((s[have] * c[have]).sum() / c[have].sum())
        spread = float(np.sqrt(((s[have] - prior) ** 2).mean()))
        gap = (s - prior).abs()
        v = q * gap / (spread + q * gap) if spread > 0 else 0.0
        out[m] = (v * prior + (1 - v) * s).fillna(prior)
    return out


class FundamentalRiskModel:
    """Fit once on the full regression history, then query point-in-time."""

    def __init__(self, reg: RegressionResult, exposures: dict, mcap: pd.DataFrame,
                 industry: pd.Series, cfg: FundamentalConfig | None = None):
        self.cfg = cfg or FundamentalConfig()
        self.reg = reg
        self.exposures = exposures
        self.mcap = mcap
        self.industry = industry
        self._exp_dates = pd.DatetimeIndex(sorted(exposures))
        self._precompute()

    def _precompute(self):
        cfg = self.cfg
        f = self.reg.factor_returns.to_numpy()
        u = self.reg.residuals.to_numpy()
        idx = self.reg.factor_returns.index

        # one-step-ahead forecast variances (made at t-1) for the bias statistics
        fvar = _recursive_ewma_var(f, cfg.vol_halflife, cfg.min_spec_obs)
        svar = _recursive_ewma_var(u, cfg.spec_halflife, cfg.min_spec_obs)
        prev_f = np.vstack([np.full((1, f.shape[1]), np.nan), fvar[:-1]])
        prev_s = np.vstack([np.full((1, u.shape[1]), np.nan), svar[:-1]])
        z_f = f / np.sqrt(prev_f)
        z_s = u / np.sqrt(prev_s)
        caps = self.mcap.shift(1).reindex(idx).to_numpy()

        self.lambda_f = pd.Series(bias_multiplier(z_f, None, cfg.vra_halflife), index=idx)
        self.lambda_s = pd.Series(bias_multiplier(z_s, caps, cfg.vra_halflife), index=idx)
        self.spec_var = pd.DataFrame(svar, index=idx, columns=self.reg.residuals.columns)

    def risk_model(self, date) -> RiskModel:
        cfg = self.cfg
        date = pd.Timestamp(date)
        fr = self.reg.factor_returns.loc[:date].iloc[-cfg.max_history:]
        if len(fr) < cfg.vol_halflife:
            raise ValueError(f"not enough factor history at {date.date()}")
        S = factor_covariance(fr.to_numpy(), cfg)
        lam_f = self.lambda_f.loc[:date].iloc[-1]
        lam_f = 1.0 if (not cfg.use_vra or not np.isfinite(lam_f)) else lam_f
        S = S * lam_f ** 2 * cfg.horizon
        names = list(fr.columns)
        F = pd.DataFrame(S, index=names, columns=names)

        X = self.exposures_at(date)
        sig = np.sqrt(self.spec_var.loc[:date].iloc[-1])
        cap = self.mcap.loc[date]
        sig = shrink_specific(sig.reindex(X.index), cap.reindex(X.index), cfg.shrink_q, cfg.n_size_buckets)
        lam_s = self.lambda_s.loc[:date].iloc[-1]
        lam_s = 1.0 if (not cfg.use_vra or not np.isfinite(lam_s)) else lam_s
        D = (sig * lam_s) ** 2 * cfg.horizon
        name = "fundamental" if cfg.use_vra else "fundamental_no_vra"
        return RiskModel(X=X[names], F=F, D=D, name=name)

    def exposures_at(self, date) -> pd.DataFrame:
        i = self._exp_dates.searchsorted(pd.Timestamp(date), side="right") - 1
        if i < 0:
            raise ValueError(f"no exposures on or before {date}")
        styles = self.exposures[self._exp_dates[i]]
        X = full_exposure_matrix(styles, self.industry)
        return X.reindex(columns=["country"] + self.reg.industries + list(styles.columns), fill_value=0.0)
