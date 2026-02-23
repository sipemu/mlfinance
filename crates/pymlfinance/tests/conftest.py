"""Shared fixtures for pymlfinance tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def price_series(rng):
    """Synthetic price series with geometric Brownian motion."""
    n = 200
    returns = rng.normal(0.0005, 0.02, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def returns_series(rng):
    """Simple return series."""
    return rng.normal(0.001, 0.02, 100)
