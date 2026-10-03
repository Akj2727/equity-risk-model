import pytest

from erm.data import SimConfig, simulate_universe
from erm.pipeline import Pipeline


@pytest.fixture(scope="session")
def sim():
    return simulate_universe(SimConfig(n_assets=80, n_years=3, n_industries=5, seed=3))


@pytest.fixture(scope="session")
def pipe(sim):
    uni, _ = sim
    return Pipeline(uni)
