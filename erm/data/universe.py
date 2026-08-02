from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Universe:
    """Everything the models need, aligned on one date index and one ticker set.

    returns    daily simple returns, dates x tickers
    mcap       market cap, dates x tickers
    turnover   daily traded value / market cap, dates x tickers
    industry   ticker -> industry label
    fundamentals  name -> dates x tickers panel (forward-filled, point-in-time)
    """

    returns: pd.DataFrame
    mcap: pd.DataFrame
    turnover: pd.DataFrame
    industry: pd.Series
    fundamentals: dict = field(default_factory=dict)

    def __post_init__(self):
        idx, cols = self.returns.index, self.returns.columns
        for name in ("mcap", "turnover"):
            panel = getattr(self, name)
            if not (panel.index.equals(idx) and panel.columns.equals(cols)):
                raise ValueError(f"{name} is not aligned with returns")
        for name, panel in self.fundamentals.items():
            if not (panel.index.equals(idx) and panel.columns.equals(cols)):
                raise ValueError(f"fundamental '{name}' is not aligned with returns")
        missing = set(cols) - set(self.industry.index)
        if missing:
            raise ValueError(f"no industry for {sorted(missing)[:5]}")
        self.industry = self.industry.reindex(cols)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index

    @property
    def tickers(self) -> pd.Index:
        return self.returns.columns

    def benchmark_weights(self, date, top: int | None = None) -> pd.Series:
        """Cap-weighted benchmark as of `date`, optionally the largest `top` names."""
        cap = self.mcap.loc[date].dropna()
        if top is not None:
            cap = cap.nlargest(top)
        w = cap / cap.sum()
        return w.reindex(self.tickers, fill_value=0.0)

    def month_ends(self, start=None) -> pd.DatetimeIndex:
        s = pd.Series(self.dates, index=self.dates)
        ends = s.groupby([self.dates.year, self.dates.month]).max()
        ends = pd.DatetimeIndex(ends.values)
        if start is not None:
            ends = ends[ends >= pd.Timestamp(start)]
        return ends

    def window(self, end, length: int) -> pd.DataFrame:
        """Last `length` rows of returns up to and including `end`."""
        pos = self.dates.get_loc(end)
        return self.returns.iloc[max(0, pos - length + 1): pos + 1]


def cap_weighted_market(returns: pd.DataFrame, mcap: pd.DataFrame) -> pd.Series:
    """Daily cap-weighted market return, weights lagged one day."""
    w = mcap.shift(1)
    w = w.div(w.sum(axis=1), axis=0)
    mkt = (w * returns).sum(axis=1)
    mkt.iloc[0] = np.nan
    return mkt
