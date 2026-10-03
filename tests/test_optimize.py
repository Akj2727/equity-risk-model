import numpy as np
import pandas as pd

from erm.alpha import grinold_alpha
from erm.attribution import return_attribution
from erm.optimize import OptConfig, alignment_direction, alignment_variance, min_variance, optimize


def _setup(pipe, sim):
    _, truth = sim
    d = pipe.uni.month_ends()[-3]
    rm = pipe.model("fundamental", d)
    bench = pipe.uni.benchmark_weights(d, 40)
    alpha = grinold_alpha(truth.signal.loc[d], rm.D)
    return d, rm, bench, alpha


def test_constraints_hold(pipe, sim):
    d, rm, bench, alpha = _setup(pipe, sim)
    cfg = OptConfig(max_weight=0.06, te_limit=0.03, style_limit=0.2, industry_limit=0.04)
    res = optimize(rm, alpha, bench, cfg, style_factors=pipe.style_factors,
                   industry_factors=pipe.industry_factors)
    w = res.weights
    assert res.status == "optimal"
    assert np.isclose(w.sum(), 1)
    assert (w >= 0).all() and (w <= 0.06 + 1e-6).all()
    assert res.predicted_te <= 0.03 + 1e-4
    x = rm.exposures(w - bench)
    assert (x[pipe.style_factors].abs() <= 0.2 + 1e-5).all()
    assert (x[pipe.industry_factors].abs() <= 0.04 + 1e-5).all()


def test_alignment_direction_is_orthogonal_to_factors(pipe, sim):
    d, rm, bench, alpha = _setup(pipe, sim)
    y = alignment_direction(alpha, rm.X, rm.D)
    assert np.isclose(np.linalg.norm(y), 1)
    gls = rm.X.T @ (y / rm.D)
    assert np.allclose(gls, 0, atol=1e-8)


def test_aaf_cuts_exposure_to_unspanned_alpha(pipe, sim):
    d, rm, bench, alpha = _setup(pipe, sim)
    y = alignment_direction(alpha, rm.X, rm.D)
    var_y = alignment_variance(y, pipe.reg.residuals.loc[:d].iloc[-252:], rm.D)
    plain = optimize(rm, alpha, bench, OptConfig()).weights
    with_aaf = optimize(rm, alpha, bench, OptConfig(), aaf=(y, var_y * 20)).weights
    assert abs(y @ (with_aaf - bench)) < abs(y @ (plain - bench))


def test_robust_penalty_shrinks_active_risk(pipe, sim):
    d, rm, bench, alpha = _setup(pipe, sim)
    omega = 0.05 * np.sqrt(rm.D)
    a0 = optimize(rm, alpha, bench, OptConfig(kappa=0.0)).weights
    a2 = optimize(rm, alpha, bench, OptConfig(kappa=3.0), omega_sqrt=omega).weights
    assert rm.vol(a2 - bench) < rm.vol(a0 - bench)


def test_min_variance_beats_benchmark_ex_ante(pipe):
    d = pipe.uni.month_ends()[-3]
    rm = pipe.model("fundamental", d)
    w = min_variance(rm, max_weight=0.1)
    assert rm.vol(w) < rm.vol(pipe.uni.benchmark_weights(d, 40))


def test_attribution_adds_up(pipe, sim):
    d, rm, bench, alpha = _setup(pipe, sim)
    w = optimize(rm, alpha, bench, OptConfig()).weights
    pos = pipe.reg.factor_returns.index.searchsorted(d, side="right")
    hold = pipe.reg.factor_returns.iloc[pos:pos + 21]
    fwd = (1 + pipe.uni.returns.iloc[pipe.uni.dates.get_loc(d) + 1:][:21]).prod() - 1
    att = return_attribution(w - bench, rm.X, hold, fwd, pipe.industry_factors)
    assert np.isclose(att.drop("total").sum(), att["total"])
