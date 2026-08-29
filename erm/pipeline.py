"""Fits everything once and hands out point-in-time risk models by name."""

from dataclasses import replace

import pandas as pd

from .baselines import ledoit_wolf_model, sample_model
from .data.universe import Universe
from .exposures import compute_exposures, FUNDAMENTAL_STYLES, PRICE_STYLES
from .fundamental import FundamentalConfig, FundamentalRiskModel
from .regression import run_regressions
from .riskmodel import RiskModel
from .statistical import hybrid_model, statistical_model

ALL_MODELS = ["sample", "ledoit_wolf", "statistical", "fundamental_no_vra", "fundamental", "hybrid"]


class Pipeline:
    def __init__(self, uni: Universe, warmup_days: int = 260, cfg: FundamentalConfig | None = None,
                 stat_window: int = 252, stat_factors: int = 10, hybrid_factors: int = 3):
        self.uni = uni
        self.cfg = cfg or FundamentalConfig()
        self.stat_window = stat_window
        self.stat_factors = stat_factors
        self.hybrid_factors = hybrid_factors
        dates = uni.month_ends(start=uni.dates[min(warmup_days, len(uni.dates) - 1)])
        self.exposures = compute_exposures(uni, dates)
        self.reg = run_regressions(uni, self.exposures)
        self.fund = FundamentalRiskModel(self.reg, self.exposures, uni.mcap, uni.industry, self.cfg)
        self.fund_no_vra = FundamentalRiskModel(self.reg, self.exposures, uni.mcap, uni.industry,
                                                replace(self.cfg, use_vra=False))

    @property
    def style_factors(self) -> list:
        styles = PRICE_STYLES + [s for s, f in FUNDAMENTAL_STYLES.items()
                                 if any(x in self.uni.fundamentals for x in f)]
        return styles

    @property
    def industry_factors(self) -> list:
        return list(self.reg.industries)

    def model(self, name: str, date) -> RiskModel:
        date = pd.Timestamp(date)
        h = self.cfg.horizon
        if name == "fundamental":
            return self.fund.risk_model(date)
        if name == "fundamental_no_vra":
            return self.fund_no_vra.risk_model(date)
        if name == "hybrid":
            base = self.fund.risk_model(date)
            resid = self.reg.residuals.loc[:date].iloc[-self.stat_window:]
            return hybrid_model(base, resid, k=self.hybrid_factors, horizon=h)
        window = self.uni.window(date, self.stat_window)
        if name == "statistical":
            return statistical_model(window, k=self.stat_factors, horizon=h)
        if name == "sample":
            return sample_model(window, horizon=h)
        if name == "ledoit_wolf":
            return ledoit_wolf_model(window, horizon=h)
        raise KeyError(name)
