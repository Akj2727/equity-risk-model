"""Where did the active return and the active risk come from?

Ex post, the active return over a holding period splits exactly into
    sum_k  (active exposure to k) * (factor k return)   +   specific
because the factor returns come from the same cross-sectional regression the
risk model uses, and specific is defined as the remainder.

Ex ante, active variance splits into each factor's contribution
x_k (F x)_k plus specific variance.
"""

import numpy as np
import pandas as pd

from .riskmodel import RiskModel


def return_attribution(active: pd.Series, X: pd.DataFrame, factor_returns: pd.DataFrame,
                       stock_returns: pd.Series, industries: list) -> pd.Series:
    """One period. factor_returns are the daily factor returns over the holding period."""
    active = active.reindex(X.index).fillna(0.0)
    x = X.T @ active
    f = factor_returns.reindex(columns=X.columns).fillna(0.0).sum()
    contrib = x * f
    total = float(active @ stock_returns.reindex(X.index).fillna(0.0))
    out = {"country": contrib.get("country", 0.0),
           "industry": float(contrib[[c for c in industries if c in contrib.index]].sum())}
    for c in X.columns:
        if c != "country" and c not in industries:
            out[c] = float(contrib[c])
    out["specific"] = total - float(contrib.sum())
    out["total"] = total
    return pd.Series(out)


def risk_attribution(rm: RiskModel, active: pd.Series, industries: list,
                     periods_per_year: int = 12) -> pd.Series:
    """Ex-ante active variance by source, as fractions of the total."""
    contrib = rm.factor_contributions(active)
    spec = float((rm.D * active.reindex(rm.tickers, fill_value=0.0) ** 2).sum())
    total = float(contrib.sum()) + spec
    out = {"industry": float(contrib[[c for c in industries if c in contrib.index]].sum())}
    for c in contrib.index:
        if c not in industries:
            out[c] = float(contrib[c])
    out["specific"] = spec
    s = pd.Series(out) / total
    s["te_annual"] = np.sqrt(total * periods_per_year)
    return s
