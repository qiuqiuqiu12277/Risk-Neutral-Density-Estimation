import numpy as np

from rnd_estimator.black_scholes import black_scholes_call
from rnd_estimator.model_selection import compare_models, fit_black_scholes_volatility
from rnd_estimator.pricing import lognormal_mixture_call_prices


def test_black_scholes_baseline_recovers_constant_volatility():
    strikes = np.linspace(80.0, 120.0, 13)
    prices = black_scholes_call(100.0, strikes, 0.5, 0.03, 0.27, 0.01)
    volatility = fit_black_scholes_volatility(
        strikes,
        prices,
        spot=100.0,
        maturity=0.5,
        rate=0.03,
        dividend_yield=0.01,
    )
    assert abs(volatility - 0.27) < 1e-6


def test_model_comparison_is_deterministic_and_auditable():
    spot, maturity, rate, dividend = 100.0, 0.25, 0.03, 0.01
    terminal_std = 0.22 * np.sqrt(maturity)
    forward = spot * np.exp((rate - dividend) * maturity)
    weights = np.array([0.2, 0.6, 0.2])
    means = forward * np.array([0.82, 1.0, 1.18])
    locations = np.log(means) - 0.5 * terminal_std**2
    strikes = np.linspace(75.0, 130.0, 13)
    prices = lognormal_mixture_call_prices(
        strikes, weights, locations, terminal_std, rate, maturity
    )

    first = compare_models(
        strikes,
        prices,
        spot,
        maturity,
        rate,
        dividend,
        terminal_std,
        component_candidates=[5, 7],
        folds=3,
    )
    second = compare_models(
        strikes,
        prices,
        spot,
        maturity,
        rate,
        dividend,
        terminal_std,
        component_candidates=[5, 7],
        folds=3,
    )

    assert first.to_summary() == second.to_summary()
    assert first.selected_components in {5, 7}
    assert first.recommended_model in {"black_scholes", "lognormal_mixture"}
    assert len(first.fold_results) == 9
    assert all(result["failed_folds"] == 0 for result in first.candidate_results)
    assert first.black_scholes["validation_metrics"]["rmse"] >= 0
