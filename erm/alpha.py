"""Turning a raw signal into an alpha the optimiser can use.

Grinold's rule: alpha = IC * residual volatility * z-score. A stock gets more
alpha when the signal is stronger and when it is more volatile, so the
optimiser's risk-adjusted view is roughly signal strength alone.
"""

import numpy as np
import pandas as pd

from .data.universe import Universe


def zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    z = (x - x.mean()) / x.std()
    return z.clip(-clip, clip).fillna(0.0)


def grinold_alpha(signal: pd.Series, specific_var: pd.Series, ic: float = 0.05) -> pd.Series:
    """Horizon alpha from a signal and horizon specific variance."""
    z = zscore(signal.reindex(specific_var.index))
    return ic * np.sqrt(specific_var) * z


def alpha_uncertainty(specific_var: pd.Series, ic: float = 0.05) -> pd.Series:
    """Per-stock std of the alpha estimate, used as Omega^1/2 in the robust optimiser."""
    return ic * np.sqrt(specific_var)


def price_signal(uni: Universe, date) -> pd.Series:
    """A simple composite for real data: 12-1 momentum, short-term reversal, quality.

    This is a placeholder so the real-data path runs end to end; swap in your
    own research signal.
    """
    pos = uni.dates.get_loc(date)
    R = uni.returns.iloc[max(0, pos - 251): pos + 1]
    mom = np.log1p(R.iloc[:-21]).sum()
    rev = -np.log1p(R.iloc[-21:]).sum()
    parts = [zscore(mom), zscore(rev)]
    if "roe" in uni.fundamentals:
        parts.append(zscore(uni.fundamentals["roe"].loc[date]))
    return zscore(pd.concat(parts, axis=1).mean(axis=1))
