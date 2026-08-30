"""Benchmark-relative portfolio optimiser on top of any RiskModel.

    maximise   alpha'a  -  lambda * a' Sigma a  -  cost * |w - w_prev|_1
               -  kappa * || Omega^1/2 a ||                  (robust term)
               -  lambda * sigma_y^2 (y'a)^2                 (alpha alignment factor)
    subject to sum(w) = 1, 0 <= w <= max_weight,
               a' Sigma a <= TE^2,
               |X_style' a| <= style_limit, |X_ind' a| <= industry_limit,
               |w - w_prev|_1 <= turnover_limit

where a = w - benchmark. Risk is evaluated in factor form (X F X' + D), so
the problem stays small however many stocks there are.

Robust term (Ceria and Stubbs). Alpha is an estimate. Plain mean-variance
treats it as exact and piles into the stocks whose alpha is most overstated.
Penalising || Omega^1/2 a || is the worst case of alpha'a over an ellipsoid of
plausible alphas, so the optimiser only takes bets the error can't erase.

Alpha alignment factor (Ceria, Saxena and Stubbs). Whatever part of alpha the
risk model's factors can't span looks like free return: the model sees only
specific risk there, so the optimiser over-bets on it. y is that unspanned
part, normalised, and it is added to the risk model as one more factor whose
variance is estimated from how much the residual returns actually move along y.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .riskmodel import RiskModel


@dataclass
class OptConfig:
    risk_aversion: float = 25.0       # on monthly variance
    max_weight: float = 0.05
    te_limit: float | None = 0.05     # annual tracking error cap
    style_limit: float | None = 0.3   # max |active exposure| per style factor
    industry_limit: float | None = 0.05
    turnover_limit: float | None = None  # one-way, as a fraction of the portfolio
    cost_bps: float = 10.0
    kappa: float = 0.0                # robust penalty, 0 = plain mean-variance
    periods_per_year: int = 12


@dataclass
class OptResult:
    weights: pd.Series
    status: str
    predicted_te: float               # annual, under the model passed in
    objective: float


def _sqrt_psd(F: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((F + F.T) / 2)
    vals = np.clip(vals, 0.0, None)
    return vecs * np.sqrt(vals)       # L with L L' = F


def alignment_direction(alpha: pd.Series, X: pd.DataFrame, D: pd.Series) -> pd.Series:
    """Unit-norm part of alpha orthogonal to the factor exposures (GLS, weights 1/D)."""
    Xn, a = X.to_numpy(), alpha.reindex(X.index).fillna(0.0).to_numpy()
    w = 1.0 / D.to_numpy()
    coef = np.linalg.lstsq(Xn * np.sqrt(w)[:, None], a * np.sqrt(w), rcond=None)[0]
    perp = a - Xn @ coef
    nrm = np.linalg.norm(perp)
    if nrm == 0:
        return pd.Series(0.0, index=X.index)
    return pd.Series(perp / nrm, index=X.index)


def alignment_variance(y: pd.Series, residuals: pd.DataFrame, D: pd.Series, horizon: int = 21) -> float:
    """Variance at the horizon of the factor along y, beyond what specific risk explains.

    Residual returns are what the risk model does not explain. Their projection
    on y has variance (factor part) + y'Dy, so the factor part is the excess.
    """
    U = residuals.reindex(columns=y.index).fillna(0.0).to_numpy()
    proj = U @ y.to_numpy()
    total = float(np.var(proj, ddof=1)) * horizon
    spec = float((y ** 2 * D).sum())
    return max(total - spec, 0.05 * spec)


def optimize(rm: RiskModel, alpha: pd.Series, bench: pd.Series, cfg: OptConfig | None = None,
             prev: pd.Series | None = None, style_factors=None, industry_factors=None,
             omega_sqrt: pd.Series | None = None, aaf: tuple[pd.Series, float] | None = None) -> OptResult:
    cfg = cfg or OptConfig()
    tickers = rm.tickers
    n = len(tickers)
    a_vec = alpha.reindex(tickers).fillna(0.0).to_numpy()
    b = bench.reindex(tickers).fillna(0.0).to_numpy()
    w0 = b if prev is None else prev.reindex(tickers).fillna(0.0).to_numpy()

    X = rm.X.to_numpy()
    L = _sqrt_psd(rm.F.to_numpy())
    sqrtD = np.sqrt(rm.D.to_numpy())

    w = cp.Variable(n)
    act = w - b
    y_fac = L.T @ (X.T @ act)
    risk = cp.sum_squares(y_fac) + cp.sum_squares(cp.multiply(sqrtD, act))

    obj = a_vec @ act - cfg.risk_aversion * risk
    if cfg.cost_bps:
        obj -= cfg.cost_bps * 1e-4 * cp.norm1(w - w0)
    if cfg.kappa > 0:
        if omega_sqrt is None:
            raise ValueError("robust optimisation needs omega_sqrt")
        om = omega_sqrt.reindex(tickers).fillna(omega_sqrt.median()).to_numpy()
        obj -= cfg.kappa * cp.norm(cp.multiply(om, act), 2)
    aaf_risk = 0
    if aaf is not None:
        y, var_y = aaf
        aaf_risk = var_y * cp.square(y.reindex(tickers).fillna(0.0).to_numpy() @ act)
        obj -= cfg.risk_aversion * aaf_risk

    cons = [cp.sum(w) == 1, w >= 0, w <= cfg.max_weight]
    if cfg.te_limit is not None:
        cons.append(risk + aaf_risk <= cfg.te_limit ** 2 / cfg.periods_per_year)
    cols = list(rm.X.columns)
    if cfg.style_limit is not None and style_factors:
        idx = [cols.index(s) for s in style_factors if s in cols]
        if idx:
            cons.append(cp.abs(X[:, idx].T @ act) <= cfg.style_limit)
    if cfg.industry_limit is not None and industry_factors:
        idx = [cols.index(s) for s in industry_factors if s in cols]
        if idx:
            cons.append(cp.abs(X[:, idx].T @ act) <= cfg.industry_limit)
    if cfg.turnover_limit is not None and prev is not None:
        cons.append(cp.norm1(w - w0) <= 2 * cfg.turnover_limit)

    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)
    if w.value is None:
        return OptResult(pd.Series(b, index=tickers), prob.status, float("nan"), float("nan"))
    wv = np.clip(w.value, 0.0, None)
    wv = wv / wv.sum()
    ws = pd.Series(wv, index=tickers)
    te = rm.vol(ws - pd.Series(b, index=tickers)) * np.sqrt(cfg.periods_per_year)
    return OptResult(ws, prob.status, te, float(prob.value))


def min_variance(rm: RiskModel, max_weight: float = 0.05) -> pd.Series:
    """Long-only, fully invested minimum-variance portfolio under rm."""
    n = len(rm.tickers)
    X = rm.X.to_numpy()
    L = _sqrt_psd(rm.F.to_numpy())
    sqrtD = np.sqrt(rm.D.to_numpy())
    w = cp.Variable(n)
    risk = cp.sum_squares(L.T @ (X.T @ w)) + cp.sum_squares(cp.multiply(sqrtD, w))
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0, w <= max_weight])
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)
    wv = np.clip(w.value, 0.0, None)
    return pd.Series(wv / wv.sum(), index=rm.tickers)
