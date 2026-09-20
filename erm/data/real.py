"""Building a Universe from real market data.

Two layers:
- universe_from_frames: pure pandas, takes prices, volumes, shares, industries
  and optional fundamentals that you already have. Tested offline.
- download_yfinance / sp500_constituents: thin network wrappers. Run them on
  your own machine.

Known limitations, worth saying out loud in an interview:
- Shares outstanding from yfinance are today's count, so historical market cap
  is approximate.
- Today's index constituents carry survivorship bias into the past.
- Fundamentals need a reporting lag; universe_from_frames applies one.
"""

from io import StringIO

import numpy as np
import pandas as pd

from .universe import Universe

SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
FUNDAMENTAL_FIELDS = ["book_to_price", "earnings_yield", "roe", "sales_growth"]


def universe_from_frames(close: pd.DataFrame, volume: pd.DataFrame, shares, industry: pd.Series,
                         fundamentals: pd.DataFrame | None = None, report_lag_days: int = 60,
                         min_history: float = 0.8) -> Universe:
    """close/volume: dates x tickers. shares: Series (constant) or dates x tickers.

    fundamentals, if given, is long format with columns
    date, ticker, and any of book_to_price, earnings_yield, roe, sales_growth.
    Each row becomes usable `report_lag_days` after its date.
    """
    close = close.sort_index()
    keep = close.columns[close.notna().mean() >= min_history]
    keep = keep.intersection(industry.dropna().index)
    close = close[keep]
    volume = volume.reindex(index=close.index, columns=keep)

    if isinstance(shares, pd.Series):
        sh = pd.DataFrame(np.tile(shares.reindex(keep).to_numpy(), (len(close), 1)),
                          index=close.index, columns=keep)
    else:
        sh = shares.reindex(index=close.index, columns=keep).ffill().bfill()
    good = sh.notna().all() & (sh > 0).all()
    keep = keep[good.reindex(keep).fillna(False).to_numpy()]
    close, volume, sh = close[keep], volume[keep], sh[keep]

    returns = close.pct_change(fill_method=None).iloc[1:]
    mcap = (close * sh).ffill().iloc[1:]
    turnover = (volume / sh).iloc[1:].replace([np.inf, -np.inf], np.nan)
    turnover = turnover.where(turnover > 0)

    fund = {}
    if fundamentals is not None and len(fundamentals):
        f = fundamentals.copy()
        f["date"] = pd.to_datetime(f["date"]) + pd.Timedelta(days=report_lag_days)
        for field in FUNDAMENTAL_FIELDS:
            if field not in f.columns:
                continue
            wide = f.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
            wide = wide.reindex(columns=keep)
            panel = wide.reindex(wide.index.union(returns.index)).sort_index().ffill()
            fund[field] = panel.reindex(returns.index)

    return Universe(returns=returns, mcap=mcap, turnover=turnover,
                    industry=industry.reindex(keep), fundamentals=fund)


def sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 list with GICS sector (columns Symbol, Security, GICS Sector)."""
    import urllib.request
    with urllib.request.urlopen(SP500_CSV, timeout=30) as resp:
        text = resp.read().decode()
    df = pd.read_csv(StringIO(text))
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def download_yfinance(tickers, start, end=None, sectors: pd.Series | None = None):
    """Prices, volumes, shares and sectors from Yahoo Finance.

    For NSE stocks pass tickers with the .NS suffix (for example RELIANCE.NS).
    """
    import yfinance as yf

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    close = raw["Close"].reindex(columns=tickers)
    volume = raw["Volume"].reindex(columns=tickers)
    shares, sector = {}, {}
    for t in tickers:
        tk = yf.Ticker(t)
        try:
            shares[t] = float(tk.fast_info["shares"])
        except Exception:
            shares[t] = np.nan
        if sectors is None:
            try:
                sector[t] = tk.info.get("sector")
            except Exception:
                sector[t] = None
    industry = sectors.reindex(tickers) if sectors is not None else pd.Series(sector)
    return close, volume, pd.Series(shares), industry.dropna()
