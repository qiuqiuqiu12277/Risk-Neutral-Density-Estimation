import numpy as np
import pytest
from scipy.integrate import trapezoid

from rnd_estimator.black_scholes import black_scholes_call
from rnd_estimator.distributions import lognormal_mixture_moments, lognormal_mixture_pdf
from rnd_estimator.pricing import lognormal_mixture_call_prices


def test_density_probability_and_forward_moment():
    weights = np.array([0.25, 0.5, 0.25])
    terminal_std = 0.12
    means = np.array([80.0, 100.0, 120.0])
    locations = np.log(means) - 0.5 * terminal_std**2
    grid = np.geomspace(1e-3, 400.0, 200_000)
    density = lognormal_mixture_pdf(grid, weights, locations, terminal_std)
    assert trapezoid(density, grid) == pytest.approx(1.0, abs=2e-6)
    assert trapezoid(grid * density, grid) == pytest.approx(100.0, abs=2e-4)
    assert lognormal_mixture_moments(weights, locations, terminal_std)["mean"] == pytest.approx(
        100.0
    )


def test_single_component_price_matches_black_scholes():
    spot, maturity, rate, dividend, annual_vol = 100.0, 0.4, 0.03, 0.01, 0.25
    terminal_std = annual_vol * np.sqrt(maturity)
    forward = spot * np.exp((rate - dividend) * maturity)
    location = np.log(forward) - 0.5 * terminal_std**2
    strikes = np.linspace(75.0, 130.0, 13)
    mixture_prices = lognormal_mixture_call_prices(
        strikes, [1.0], [location], terminal_std, rate, maturity
    )
    benchmark = black_scholes_call(spot, strikes, maturity, rate, annual_vol, dividend)
    np.testing.assert_allclose(mixture_prices, benchmark, rtol=1e-12, atol=1e-12)


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        lognormal_mixture_pdf([90.0, 100.0], [0.2, 0.2], [4.4, 4.7], 0.1)
