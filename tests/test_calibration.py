import numpy as np
import pytest

from rnd_estimator.calibration import build_component_grid, fit_lognormal_mixture
from rnd_estimator.diagnostics import call_arbitrage_diagnostics
from rnd_estimator.pricing import component_means, lognormal_mixture_call_prices


def _synthetic_case():
    spot, maturity, rate, dividend = 100.0, 0.3, 0.025, 0.01
    terminal_std = 0.2 * np.sqrt(maturity)
    forward = spot * np.exp((rate - dividend) * maturity)
    locations = build_component_grid(forward, terminal_std, 7, 2.5)
    means = component_means(locations, terminal_std)
    lower, upper = 1, 5
    upper_weight = (forward - means[lower]) / (means[upper] - means[lower])
    weights = np.zeros(7)
    weights[lower] = 1.0 - upper_weight
    weights[upper] = upper_weight
    strikes = np.linspace(75.0, 130.0, 19)
    prices = lognormal_mixture_call_prices(
        strikes, weights, locations, terminal_std, rate, maturity
    )
    return spot, maturity, rate, dividend, terminal_std, forward, locations, strikes, prices


def test_calibration_enforces_probability_and_martingale_constraints():
    spot, maturity, rate, dividend, std, forward, locations, strikes, prices = _synthetic_case()
    fit = fit_lognormal_mixture(
        strikes,
        prices,
        spot,
        maturity,
        rate,
        dividend,
        std,
        log_locations=locations,
        smoothness=0.0,
    )
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.all(fit.weights >= 0)
    assert fit.moments["mean"] == pytest.approx(forward, rel=1e-9)
    assert fit.metrics["rmse"] < 1e-5


def test_analytic_mixture_prices_are_static_arbitrage_free():
    spot, maturity, rate, dividend, _, _, _, strikes, prices = _synthetic_case()
    diagnostics = call_arbitrage_diagnostics(
        strikes, prices, spot, maturity, rate, dividend, tolerance=1e-9
    )
    assert diagnostics.is_arbitrage_free
