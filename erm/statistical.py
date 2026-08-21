"""Axioma-style statistical and hybrid risk models.

Statistical model
    Asymptotic PCA (Connor and Korajczyk): the factors are the top
    eigenvectors of the T x T matrix R R' / N, which works when there are more
    stocks than days in the window. Loadings come from regressing each stock on
    those factor returns. No fundamental data is used, so the model picks up
    whatever is driving co-movement right now, including things no descriptor
    measures. A short window gives a responsive model, a long one a stable model.

Hybrid model
    Start from the fundamental model, run the same PCA on its residuals, and
    add the top few residual components as extra factors. This catches risk the
    fundamental descriptors miss while keeping their interpretability.
"""

import numpy as np
import pandas as pd

from .riskmodel import RiskModel


def _clean_window(R: pd.DataFrame, min_coverage: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Demeaned returns with NaNs set to zero, plus a mask of usable stocks."""
    cover = R.notna().mean()
    usable = (cover >= min_coverage).to_numpy()
    A = R.to_numpy()
    A = A - np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=0)
    A = np.where(np.isfinite(A), A, 0.0)
    return A, usable


def apca(R: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymptotic PCA on a T x N demeaned panel.

    Returns (factor returns T x k with unit variance, loadings N x k, residuals T x N).
    """
    T, N = R.shape
    k = min(k, T - 1)
    M = R @ R.T / N
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1][:k]
    Fr = vecs[:, order] * np.sqrt(T)          # unit variance factor returns
    B = np.linalg.lstsq(Fr, R, rcond=None)[0].T
    U = R - Fr @ B.T
    return Fr, B, U


def statistical_model(window: pd.DataFrame, k: int = 10, horizon: int = 21,
                      name: str = "statistical") -> RiskModel:
    tickers = window.columns
    A, usable = _clean_window(window)
    Fr, B, U = apca(A[:, usable], k)
    T = A.shape[0]
    names = [f"pc{i + 1}" for i in range(Fr.shape[1])]
    F = np.cov(Fr, rowvar=False, ddof=1) * horizon
    X = pd.DataFrame(0.0, index=tickers, columns=names)
    X.loc[tickers[usable]] = B
    spec = (U ** 2).sum(axis=0) / (T - Fr.shape[1])
    D = pd.Series(np.nan, index=tickers)
    D[tickers[usable]] = spec
    # stocks with too little history get the cross-sectional median plus a margin
    D = D.fillna(D.median() * 1.5) * horizon
    return RiskModel(X=X, F=pd.DataFrame(np.atleast_2d(F), index=names, columns=names), D=D, name=name)


def hybrid_model(base: RiskModel, residual_window: pd.DataFrame, k: int = 3,
                 horizon: int = 21) -> RiskModel:
    """Fundamental model plus k statistical factors fitted on its residuals."""
    tickers = base.tickers
    resid = residual_window.reindex(columns=tickers)
    A, usable = _clean_window(resid)
    Fr, B, U = apca(A[:, usable], k)
    T = A.shape[0]
    names = [f"stat{i + 1}" for i in range(Fr.shape[1])]

    Xs = pd.DataFrame(0.0, index=tickers, columns=names)
    Xs.loc[tickers[usable]] = B
    X = pd.concat([base.X, Xs], axis=1)

    Fs = np.cov(Fr, rowvar=False, ddof=1) * horizon
    K0, K1 = base.F.shape[0], len(names)
    F = np.zeros((K0 + K1, K0 + K1))
    F[:K0, :K0] = base.F.to_numpy()
    F[K0:, K0:] = np.atleast_2d(Fs)
    labels = list(base.F.index) + names
    F = pd.DataFrame(F, index=labels, columns=labels)

    # scale the base specific variance by how much of it the new factors explain;
    # this keeps the base model's shrinkage and regime scaling
    before = (A[:, usable] ** 2).sum(axis=0)
    after = (U ** 2).sum(axis=0) * T / max(T - K1, 1)
    ratio = pd.Series(1.0, index=tickers)
    ratio[tickers[usable]] = np.clip(after / np.where(before > 0, before, np.nan), 0.3, 1.0)
    D = base.D * ratio.fillna(1.0)
    return RiskModel(X=X, F=F, D=D, name="hybrid")
