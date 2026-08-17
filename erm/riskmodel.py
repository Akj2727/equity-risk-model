"""The risk model object every estimator returns and every optimiser consumes.

Sigma = X F X' + diag(D), all at the forecast horizon (monthly by default).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskModel:
    X: pd.DataFrame      # tickers x factors
    F: pd.DataFrame      # factors x factors
    D: pd.Series         # tickers, specific variance
    name: str = "model"

    def __post_init__(self):
        if not self.X.columns.equals(self.F.index) or not self.F.index.equals(self.F.columns):
            raise ValueError("factor labels of X and F do not match")
        if not self.X.index.equals(self.D.index):
            raise ValueError("tickers of X and D do not match")

    @property
    def tickers(self):
        return self.X.index

    @property
    def factors(self):
        return list(self.X.columns)

    def cov(self) -> pd.DataFrame:
        X = self.X.to_numpy()
        S = X @ self.F.to_numpy() @ X.T + np.diag(self.D.to_numpy())
        return pd.DataFrame(S, index=self.tickers, columns=self.tickers)

    def variance(self, w: pd.Series) -> float:
        w = w.reindex(self.tickers, fill_value=0.0).to_numpy()
        x = self.X.to_numpy().T @ w
        return float(x @ self.F.to_numpy() @ x + (self.D.to_numpy() * w ** 2).sum())

    def vol(self, w: pd.Series) -> float:
        return float(np.sqrt(max(self.variance(w), 0.0)))

    def exposures(self, w: pd.Series) -> pd.Series:
        w = w.reindex(self.tickers, fill_value=0.0)
        return self.X.T @ w

    def decompose(self, w: pd.Series) -> dict:
        """Factor vs specific variance of portfolio w."""
        w = w.reindex(self.tickers, fill_value=0.0)
        x = self.exposures(w).to_numpy()
        fvar = float(x @ self.F.to_numpy() @ x)
        svar = float((self.D * w ** 2).sum())
        return {"factor_var": fvar, "specific_var": svar, "total_var": fvar + svar}

    def factor_contributions(self, w: pd.Series) -> pd.Series:
        """Each factor's contribution to total variance (x_k * (F x)_k), sums to factor_var."""
        x = self.exposures(w)
        return x * (self.F @ x)

    def subset(self, tickers) -> "RiskModel":
        return RiskModel(self.X.loc[tickers], self.F, self.D.loc[tickers], self.name)
