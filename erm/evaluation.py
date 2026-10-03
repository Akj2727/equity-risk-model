"""How good is each risk model's forecast, out of sample?

Bias statistic
    For a portfolio, z_t = realised return over the next month / forecast vol.
    If the forecast is right, std(z) = 1. Above 1 means risk was
    underpredicted. With T months, a correct model lands inside
    1 +/- sqrt(2/T) about 95% of the time.

Test portfolios
    The cap-weighted benchmark, random long-only portfolios, random
    dollar-neutral long/short portfolios, and random active portfolios
    (long-only minus benchmark). The same portfolios are scored under every model.

Minimum-variance test
    Each model builds its own long-only minimum-variance portfolio. The model
    whose portfolio realises the lowest volatility understands the covariance
    best, and the gap between predicted and realised vol shows how much the
    optimiser exploited the model's errors.
"""

import numpy as np
import pandas as pd

from .data.universe import Universe
from .optimize import min_variance


def forward_returns(uni: Universe, date, horizon: int = 21) -> pd.Series:
    pos = uni.dates.get_loc(date)
    R = uni.returns.iloc[pos + 1: pos + 1 + horizon]
    if len(R) < horizon:
        return None
    return (1 + R.fillna(0.0)).prod() - 1


def make_test_portfolios(uni: Universe, date, seed: int, n_long=20, n_ls=20, n_active=10,
                         bench_top=100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = uni.tickers
    N = len(tickers)
    bench = uni.benchmark_weights(date, bench_top)
    cols = {"benchmark": bench.to_numpy()}
    for i in range(n_long):
        w = np.zeros(N)
        pick = rng.choice(N, 50, replace=False)
        w[pick] = rng.dirichlet(np.ones(50))
        cols[f"long_{i}"] = w
    for i in range(n_ls):
        w = rng.standard_normal(N)
        w -= w.mean()
        cols[f"longshort_{i}"] = w / np.abs(w).sum() * 2
    for i in range(n_active):
        w = np.zeros(N)
        pick = rng.choice(N, 50, replace=False)
        w[pick] = rng.dirichlet(np.ones(50))
        cols[f"active_{i}"] = w - bench.to_numpy()
    return pd.DataFrame(cols, index=tickers)


def _group(name: str) -> str:
    return name.split("_")[0]


def evaluate_models(pipe, dates, models, horizon: int = 21, seed: int = 0) -> dict:
    uni = pipe.uni
    z = {m: [] for m in models}
    mv = {m: [] for m in models}
    used = []
    for i, d in enumerate(dates):
        fwd = forward_returns(uni, d, horizon)
        if fwd is None:
            continue
        P = make_test_portfolios(uni, d, seed + i)
        realised = P.T @ fwd
        for m in models:
            rm = pipe.model(m, d)
            vols = pd.Series({p: rm.vol(P[p]) for p in P.columns})
            z[m].append(realised / vols)
            w = min_variance(rm)
            mv[m].append({"pred_vol": rm.vol(w), "realised": float(w @ fwd)})
        used.append(d)

    T = len(used)
    ci = np.sqrt(2 / T)
    rows = []
    per_portfolio = {}
    for m in models:
        Z = pd.DataFrame(z[m], index=used)
        b = Z.std(ddof=1)
        per_portfolio[m] = b
        for g, bg in b.groupby(_group):
            rows.append({"model": m, "group": g, "mean_bias": bg.mean(),
                         "mrad": (bg - 1).abs().mean(),
                         "in_ci": ((bg - 1).abs() <= ci).mean()})
        rows.append({"model": m, "group": "all", "mean_bias": b.mean(),
                     "mrad": (b - 1).abs().mean(), "in_ci": ((b - 1).abs() <= ci).mean()})
    bias = pd.DataFrame(rows)

    mv_rows = []
    for m in models:
        df = pd.DataFrame(mv[m], index=used)
        mv_rows.append({"model": m,
                        "predicted_vol": df["pred_vol"].mean() * np.sqrt(12),
                        "realised_vol": df["realised"].std(ddof=1) * np.sqrt(12)})
    return {"bias": bias, "min_variance": pd.DataFrame(mv_rows), "months": T,
            "ci_halfwidth": ci, "per_portfolio": pd.DataFrame(per_portfolio)}
