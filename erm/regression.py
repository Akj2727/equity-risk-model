"""Daily cross-sectional regression for factor returns.

r_t = X_{t-1} f_t + u_t, solved by weighted least squares with sqrt(market cap)
weights. Country and industry dummies are collinear (every stock is in exactly
one industry), so the industry returns are constrained to have a cap-weighted
sum of zero. The country factor then reads as the cap-weighted market and each
industry factor as that industry's return relative to it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.universe import Universe
from .exposures import full_exposure_matrix


@dataclass
class RegressionResult:
    factor_returns: pd.DataFrame   # dates x factors
    residuals: pd.DataFrame        # dates x tickers (specific returns)
    r2: pd.Series
    factor_names: list
    industries: list


def restriction_matrix(n_country: int, ind_caps: np.ndarray, n_style: int) -> np.ndarray:
    """R such that f = R g enforces sum_k cap_k * f_ind_k = 0.

    Columns of g are [country, industries except the last, styles].
    """
    k_ind = len(ind_caps)
    K = n_country + k_ind + n_style
    R = np.zeros((K, K - 1))
    R[:n_country, :n_country] = np.eye(n_country)
    for j in range(k_ind - 1):
        R[n_country + j, n_country + j] = 1.0
        R[n_country + k_ind - 1, n_country + j] = -ind_caps[j] / ind_caps[-1]
    R[n_country + k_ind:, n_country + k_ind - 1:] = np.eye(n_style)
    return R


def wls_with_constraint(r, X, w, n_country, n_ind, ind_caps):
    """Solve one cross-section. Returns (factor returns, residuals, r2)."""
    n_style = X.shape[1] - n_country - n_ind
    R = restriction_matrix(n_country, ind_caps, n_style)
    sw = np.sqrt(w)
    A = (X @ R) * sw[:, None]
    g, *_ = np.linalg.lstsq(A, r * sw, rcond=None)
    f = R @ g
    u = r - X @ f
    wr = w / w.sum()
    mu = (wr * r).sum()
    ss_tot = (wr * (r - mu) ** 2).sum()
    r2 = 1 - (wr * u ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return f, u, r2


def _latest_before(sorted_dates: pd.DatetimeIndex, day) -> pd.Timestamp | None:
    i = sorted_dates.searchsorted(day, side="left") - 1
    return sorted_dates[i] if i >= 0 else None


def run_regressions(uni: Universe, exposures: dict, start=None, end=None) -> RegressionResult:
    """Factor returns for every day that has exposures from a strictly earlier date."""
    exp_dates = pd.DatetimeIndex(sorted(exposures))
    industries = sorted(uni.industry.dropna().unique())
    days = uni.dates
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]

    cache = {}
    rows_f, rows_u, rows_r2, used = [], [], [], []
    cap_prev = uni.mcap.shift(1)
    for day in days:
        e = _latest_before(exp_dates, day)
        if e is None:
            continue
        if e not in cache:
            X = full_exposure_matrix(exposures[e], uni.industry)
            X = X.reindex(columns=["country"] + industries + list(exposures[e].columns), fill_value=0.0)
            cache[e] = X
        X = cache[e]
        r = uni.returns.loc[day]
        cap = cap_prev.loc[day]
        ok = r.notna() & cap.notna() & X.notna().all(axis=1)
        if ok.sum() < X.shape[1] + 5:
            continue
        Xo = X[ok].to_numpy()
        w = np.sqrt(cap[ok].to_numpy())
        ind_caps = np.array([cap[ok & (uni.industry == k)].sum() for k in industries])
        if (ind_caps <= 0).any():
            continue
        f, u, r2 = wls_with_constraint(r[ok].to_numpy(), Xo, w, 1, len(industries), ind_caps)
        resid = pd.Series(np.nan, index=uni.tickers)
        resid[ok] = u
        rows_f.append(f)
        rows_u.append(resid.to_numpy())
        rows_r2.append(r2)
        used.append(day)

    names = list(next(iter(cache.values())).columns)
    idx = pd.DatetimeIndex(used)
    return RegressionResult(
        factor_returns=pd.DataFrame(rows_f, index=idx, columns=names),
        residuals=pd.DataFrame(rows_u, index=idx, columns=uni.tickers),
        r2=pd.Series(rows_r2, index=idx),
        factor_names=names,
        industries=industries,
    )
