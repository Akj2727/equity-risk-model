"""Monthly benchmark-relative backtest of optimiser + risk model combinations.

Every strategy sees the same alpha. They differ only in which risk model the
optimiser trusts and whether the robust term or the alpha alignment factor is
switched on, so differences in realised tracking error are down to risk
modelling and nothing else.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .alpha import alpha_uncertainty, grinold_alpha
from .attribution import return_attribution
from .evaluation import forward_returns
from .optimize import OptConfig, alignment_direction, alignment_variance, optimize


@dataclass
class Strategy:
    name: str
    model: str = "fundamental"
    kappa: float = 0.0
    aaf: bool = False


DEFAULT_STRATEGIES = [
    Strategy("MVO", "fundamental"),
    Strategy("MVO-Hybrid", "hybrid"),
    Strategy("MVO+AAF", "fundamental", aaf=True),
    Strategy("Robust", "fundamental", kappa=1.0),
    Strategy("Robust+AAF", "fundamental", kappa=1.0, aaf=True),
]


@dataclass
class BacktestResult:
    monthly: pd.DataFrame                 # one row per (strategy, month)
    summary: pd.DataFrame
    attribution: pd.DataFrame             # mean annualised active return by source
    weights: dict = field(default_factory=dict)


def _drift(w: pd.Series, fwd: pd.Series) -> pd.Series:
    g = w * (1 + fwd.reindex(w.index).fillna(0.0))
    return g / g.sum()


def run_backtest(pipe, dates, signal_fn, strategies=None, opt_cfg: OptConfig | None = None,
                 ic: float = 0.05, bench_top: int = 100, horizon: int = 21,
                 hidden_loading: pd.Series | None = None, aaf_window: int = 252) -> BacktestResult:
    uni = pipe.uni
    strategies = strategies or DEFAULT_STRATEGIES
    opt_cfg = opt_cfg or OptConfig()
    rows, attr_rows = [], []
    prev = {s.name: None for s in strategies}
    weights = {s.name: {} for s in strategies}

    for d in dates:
        fwd = forward_returns(uni, d, horizon)
        if fwd is None:
            continue
        bench = uni.benchmark_weights(d, bench_top)
        base = pipe.model("fundamental", d)
        models = {"fundamental": base}
        if any(s.model == "hybrid" for s in strategies):
            models["hybrid"] = pipe.model("hybrid", d)
        alpha = grinold_alpha(signal_fn(d), base.D, ic)
        omega = alpha_uncertainty(base.D, ic)

        pos = pipe.reg.factor_returns.index.searchsorted(d, side="right")
        hold_f = pipe.reg.factor_returns.iloc[pos: pos + horizon]
        resid_hist = pipe.reg.residuals.loc[:d].iloc[-aaf_window:]
        bench_fwd = float(bench @ fwd)

        for s in strategies:
            rm = models[s.model]
            aaf = None
            if s.aaf:
                y = alignment_direction(alpha, rm.X, rm.D)
                aaf = (y, alignment_variance(y, resid_hist, rm.D, horizon))
            cfg = replace(opt_cfg, kappa=s.kappa)
            res = optimize(rm, alpha, bench, cfg, prev=prev[s.name],
                           style_factors=pipe.style_factors, industry_factors=pipe.industry_factors,
                           omega_sqrt=omega, aaf=aaf)
            w = res.weights
            start = bench if prev[s.name] is None else prev[s.name]
            turnover = 0.5 * float((w - start.reindex(w.index).fillna(0.0)).abs().sum())
            active = w - bench
            own_var = rm.variance(active)
            if aaf is not None:
                own_var += aaf[1] * float(aaf[0].reindex(active.index).fillna(0.0) @ active) ** 2
            gross = float(active @ fwd)
            net = gross - opt_cfg.cost_bps * 1e-4 * 2 * turnover
            row = {
                "strategy": s.name, "date": d, "active_return": net, "gross_active": gross,
                "bench_return": bench_fwd,
                "pred_te": np.sqrt(own_var * 12),
                "pred_te_fundamental": base.vol(active) * np.sqrt(12),
                "pred_te_hybrid": models["hybrid"].vol(active) * np.sqrt(12) if "hybrid" in models else np.nan,
                "turnover": turnover, "n_names": int((w > 1e-4).sum()),
                "alpha_exposure": float(alpha @ active), "status": res.status,
            }
            if hidden_loading is not None:
                row["hidden_exposure"] = float(hidden_loading.reindex(w.index).fillna(0.0) @ active)
            rows.append(row)
            att = return_attribution(active, base.X, hold_f, fwd, pipe.industry_factors)
            att["strategy"], att["date"] = s.name, d
            attr_rows.append(att)
            weights[s.name][d] = w
            prev[s.name] = _drift(w, fwd)

    monthly = pd.DataFrame(rows)
    attr = pd.DataFrame(attr_rows)
    return BacktestResult(monthly, summarize(monthly), summarize_attribution(attr), weights)


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in monthly.groupby("strategy", sort=False):
        ann_ret = g["active_return"].mean() * 12
        te = g["active_return"].std(ddof=1) * np.sqrt(12)
        row = {
            "strategy": name,
            "active_return": ann_ret,
            "realised_te": te,
            "predicted_te": g["pred_te"].mean(),
            "te_ratio": te / g["pred_te"].mean(),
            "info_ratio": ann_ret / te if te > 0 else np.nan,
            "turnover": g["turnover"].mean(),
            "names": g["n_names"].mean(),
        }
        if "hidden_exposure" in g:
            row["abs_hidden_exposure"] = g["hidden_exposure"].abs().mean()
        out.append(row)
    return pd.DataFrame(out)


def summarize_attribution(attr: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in attr.columns if c not in ("strategy", "date")]
    return (attr.groupby("strategy", sort=False)[cols].mean() * 12)
