"""Covariance estimators with no factor structure, for comparison.

They are wrapped in the same RiskModel container (X = identity, F = Sigma,
D = 0) so the evaluation and optimiser code treat every model the same way.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .riskmodel import RiskModel


def _as_model(S: np.ndarray, tickers, name: str) -> RiskModel:
    X = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    F = pd.DataFrame(S, index=tickers, columns=tickers)
    D = pd.Series(0.0, index=tickers)
    return RiskModel(X=X, F=F, D=D, name=name)


def _filled(window: pd.DataFrame) -> np.ndarray:
    return window.fillna(window.mean()).fillna(0.0).to_numpy()


def sample_model(window: pd.DataFrame, horizon: int = 21) -> RiskModel:
    S = np.cov(_filled(window), rowvar=False, ddof=1) * horizon
    return _as_model(S, window.columns, "sample")


def ledoit_wolf_model(window: pd.DataFrame, horizon: int = 21) -> RiskModel:
    S = LedoitWolf().fit(_filled(window)).covariance_ * horizon
    return _as_model(S, window.columns, "ledoit_wolf")
