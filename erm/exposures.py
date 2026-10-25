"""Style factor exposures, built the way Barra's USE4 notes describe.

Each descriptor is computed point-in-time from data up to the exposure date,
winsorised, then standardised to a cap-weighted mean of zero and an
equal-weighted standard deviation of one. Descriptors are averaged into
factors, residual volatility is orthogonalised to beta and size, and missing
values are filled with the industry median (zero after standardising, so the
stock simply gets no tilt on that factor).
"""

import numpy as np
import pandas as pd

from .data.universe import Universe, cap_weighted_market

PRICE_STYLES = ["beta", "momentum", "size", "resvol", "liquidity"]
FUNDAMENTAL_STYLES = {
    "value": ["book_to_price", "earnings_yield"],
    "quality": ["roe"],
    "growth": ["sales_growth"],
}

BETA_WINDOW, BETA_HALFLIFE = 252, 63
MOM_WINDOW, MOM_LAG, MOM_HALFLIFE = 504, 21, 126
MIN_OBS = 126


def ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Weights for n observations, oldest first, summing to one."""
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def winsorize(x: pd.Series, k: float = 3.0, iters: int = 3) -> pd.Series:
    x = x.copy()
    for _ in range(iters):
        mu, sd = x.mean(), x.std()
        if not np.isfinite(sd) or sd == 0:
            break
        x = x.clip(mu - k * sd, mu + k * sd)
    return x


def standardize(x: pd.Series, cap: pd.Series) -> pd.Series:
    """Cap-weighted mean 0, equal-weighted std 1, after winsorising."""
    x = winsorize(x)
    ok = x.notna() & cap.notna()
    w = cap[ok] / cap[ok].sum()
    mu = float((w * x[ok]).sum())
    sd = float(x[ok].std())
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd


def _weighted_beta_resvol(R: np.ndarray, m: np.ndarray, w: np.ndarray):
    """EWMA-weighted regression of each column of R on m, ignoring NaNs per column.

    Returns (beta, residual std, observation count).
    """
    mask = np.isfinite(R) & np.isfinite(m)[:, None]
    Wm = w[:, None] * mask
    sw = Wm.sum(axis=0)
    sw = np.where(sw > 0, sw, np.nan)
    Rz = np.where(mask, R, 0.0)
    mz = np.where(mask, m[:, None], 0.0)
    mr = (Wm * Rz).sum(axis=0) / sw
    mm = (Wm * mz).sum(axis=0) / sw
    cov = (Wm * (Rz - mr) * (mz - mm)).sum(axis=0) / sw
    var = (Wm * (mz - mm) ** 2).sum(axis=0) / sw
    beta = cov / var
    resid = np.where(mask, Rz - mr - beta * (mz - mm), 0.0)
    rvol = np.sqrt((Wm * resid ** 2).sum(axis=0) / sw)
    return beta, rvol, mask.sum(axis=0)


def price_descriptors(uni: Universe, date, market: pd.Series) -> pd.DataFrame:
    pos = uni.dates.get_loc(date)
    tickers = uni.tickers
    out = pd.DataFrame(index=tickers, dtype=float)

    out["size"] = np.log(uni.mcap.iloc[pos])

    lo = max(0, pos - BETA_WINDOW + 1)
    R = uni.returns.iloc[lo:pos + 1].to_numpy()
    m = market.iloc[lo:pos + 1].to_numpy()
    w = ewma_weights(len(m), BETA_HALFLIFE)
    beta, rvol, nobs = _weighted_beta_resvol(R, m, w)
    beta[nobs < MIN_OBS] = np.nan
    rvol[nobs < MIN_OBS] = np.nan
    out["beta"] = beta
    out["resvol"] = rvol

    # relative strength: EWMA of log returns, skipping the most recent month
    end = pos - MOM_LAG
    lo = max(0, end - MOM_WINDOW + 1)
    if end > lo:
        L = np.log1p(uni.returns.iloc[lo:end + 1].to_numpy())
        wm = ewma_weights(L.shape[0], MOM_HALFLIFE)
        ok = np.isfinite(L)
        mom = (wm[:, None] * np.where(ok, L, 0.0)).sum(axis=0) * L.shape[0]
        mom[ok.sum(axis=0) < MIN_OBS] = np.nan
        out["momentum"] = mom
    else:
        out["momentum"] = np.nan

    # liquidity: log share turnover over a month, quarter and year
    turn = uni.turnover.iloc[max(0, pos - 251):pos + 1]
    stom = np.log(turn.iloc[-21:].mean() * 21)
    stoq = np.log(turn.iloc[-63:].mean() * 21)
    stoa = np.log(turn.mean() * 21)
    out["liquidity"] = 0.35 * stom + 0.35 * stoq + 0.30 * stoa
    return out


def _industry_fill(x: pd.Series, industry: pd.Series) -> pd.Series:
    med = x.groupby(industry).transform("median")
    return x.fillna(med).fillna(0.0)


def _orthogonalize(y: pd.Series, X: pd.DataFrame, cap: pd.Series) -> pd.Series:
    """Residual of y on X (plus intercept) under sqrt-cap weights."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    w = np.sqrt(cap.to_numpy())
    # names not yet listed have no cap; fit on the rest, residualise everyone
    ok = np.isfinite(w) & np.isfinite(A).all(axis=1) & np.isfinite(y.to_numpy())
    coef, *_ = np.linalg.lstsq(A[ok] * w[ok, None], y.to_numpy()[ok] * w[ok], rcond=None)
    return y - A @ coef


def style_exposures(uni: Universe, date, market: pd.Series | None = None) -> pd.DataFrame:
    """Standardised style exposures for every ticker as of `date`."""
    if market is None:
        market = cap_weighted_market(uni.returns, uni.mcap)
    cap = uni.mcap.loc[date]
    raw = price_descriptors(uni, date, market)
    ind = uni.industry

    X = pd.DataFrame(index=uni.tickers, dtype=float)
    for name in PRICE_STYLES:
        X[name] = _industry_fill(standardize(raw[name], cap), ind)

    for name, fields in FUNDAMENTAL_STYLES.items():
        fields = [f for f in fields if f in uni.fundamentals]
        if not fields:
            continue
        parts = [standardize(uni.fundamentals[f].loc[date], cap) for f in fields]
        combo = pd.concat(parts, axis=1).mean(axis=1)
        X[name] = _industry_fill(standardize(combo, cap), ind)

    # residual volatility and liquidity are strongly collinear with beta and
    # size; keep only the part they don't explain, as USE4 does
    X["resvol"] = standardize(_orthogonalize(X["resvol"], X[["beta", "size"]], cap), cap)
    X["liquidity"] = standardize(_orthogonalize(X["liquidity"], X[["size"]], cap), cap)

    # final pass so every factor has exactly the target moments
    for c in X.columns:
        X[c] = standardize(X[c], cap)
    return X


def compute_exposures(uni: Universe, dates) -> dict:
    """Style exposures at each of `dates`, keyed by date."""
    market = cap_weighted_market(uni.returns, uni.mcap)
    return {d: style_exposures(uni, d, market) for d in dates}


def industry_dummies(industry: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(industry).astype(float).reindex(industry.index)


def full_exposure_matrix(styles: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """[country | industries | styles], the regression design matrix."""
    country = pd.DataFrame({"country": 1.0}, index=styles.index)
    return pd.concat([country, industry_dummies(industry), styles], axis=1)
