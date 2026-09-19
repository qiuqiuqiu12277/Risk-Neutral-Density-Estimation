import numpy as np
import pytest

from rnd_estimator.black_scholes import black_scholes_call, implied_volatility


def test_black_scholes_vectorizes_and_obeys_bounds():
    strikes = np.array([80.0, 100.0, 120.0])
    prices = black_scholes_call(100.0, strikes, 0.5, 0.03, 0.22, 0.01)
    assert prices.shape == strikes.shape
    assert np.all(np.diff(prices) < 0)
    assert np.all(prices >= 0)
    assert np.all(prices <= 100.0)


def test_implied_volatility_round_trip():
    expected = 0.31
    price = black_scholes_call(100.0, 107.0, 0.4, 0.025, expected, 0.01)
    actual = implied_volatility(price, 100.0, 107.0, 0.4, 0.025, 0.01)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_invalid_call_price_is_rejected():
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(120.0, 100.0, 100.0, 0.5, 0.02)
