"""Constrained calibration of a nonparametric lognormal mixture."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .diagnostics import error_metrics
from .distributions import lognormal_mixture_moments
from .pricing import component_call_price_matrix, component_means


class CalibrationError(RuntimeError):
    """Raised when the constrained optimizer cannot produce a valid fit."""


@dataclass(frozen=True)
class FitResult:
    weights: np.ndarray
    log_locations: np.ndarray
    terminal_log_std: float
    fitted_prices: np.ndarray
    objective: float
    iterations: int
    forward: float
    forward_error: float
    metrics: dict[str, float]
    moments: dict[str, float]

    def to_summary(self) -> dict[str, object]:
        return {
            "weights": self.weights.tolist(),
            "log_locations": self.log_locations.tolist(),
            "terminal_log_std": self.terminal_log_std,
            "objective": self.objective,
            "iterations": self.iterations,
            "forward": self.forward,
            "forward_error": self.forward_error,
            "metrics": self.metrics,
            "moments": self.moments,
        }


def build_component_grid(
    forward: float,
    terminal_log_std: float,
    n_components: int = 9,
    width: float = 3.0,
) -> np.ndarray:
    """Create ordered log-locations whose component means bracket the forward."""

    if forward <= 0:
        raise ValueError("forward must be positive")
    if terminal_log_std <= 0:
        raise ValueError("terminal_log_std must be positive")
    if n_components < 3:
        raise ValueError("n_components must be at least 3")
    if width <= 0:
        raise ValueError("width must be positive")
    centre = np.log(forward) - 0.5 * terminal_log_std**2
    return centre + np.linspace(-width, width, n_components) * terminal_log_std


def _feasible_initial_weights(means: np.ndarray, forward: float) -> np.ndarray:
    if forward < means.min() or forward > means.max():
        raise ValueError("component means must bracket the forward")
    weights = np.zeros(len(means), dtype=float)
    exact = np.flatnonzero(np.isclose(means, forward, rtol=1e-12, atol=1e-12))
    if len(exact):
        weights[int(exact[0])] = 1.0
        return weights
    upper = int(np.searchsorted(means, forward))
    lower = upper - 1
    upper_weight = (forward - means[lower]) / (means[upper] - means[lower])
    weights[lower] = 1.0 - upper_weight
    weights[upper] = upper_weight
    return weights


def _second_difference_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((max(size - 2, 0), size), dtype=float)
    for row in range(size - 2):
        matrix[row, row : row + 3] = (1.0, -2.0, 1.0)
    return matrix


def fit_lognormal_mixture(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    terminal_log_std: float,
    n_components: int = 9,
    width: float = 3.0,
    smoothness: float = 1e-4,
    log_locations: Optional[ArrayLike] = None,
    max_iterations: int = 2_000,
) -> FitResult:
    """Fit non-negative weights under probability and martingale constraints.

    The density is constrained to integrate to one and to satisfy
    ``E_Q[S_T] = S_0 exp((r-q)T)`` exactly. Optimizer failures are raised rather
    than silently replaced with an unconstrained estimate.
    """

    k = np.asarray(strikes, dtype=float)
    observed = np.asarray(call_prices, dtype=float)
    if k.ndim != 1 or observed.ndim != 1 or len(k) != len(observed):
        raise ValueError("strikes and call_prices must be equal-length vectors")
    if len(k) < 3:
        raise ValueError("at least three option prices are required")
    if np.any(k <= 0) or np.any(observed < 0):
        raise ValueError("strikes must be positive and call prices non-negative")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(observed)):
        raise ValueError("input prices must be finite")
    if spot <= 0 or maturity <= 0:
        raise ValueError("spot and maturity must be positive")
    if terminal_log_std <= 0:
        raise ValueError("terminal_log_std must be positive")
    if smoothness < 0:
        raise ValueError("smoothness cannot be negative")

    forward = spot * np.exp((rate - dividend_yield) * maturity)
    if log_locations is None:
        locations = build_component_grid(forward, terminal_log_std, n_components, width)
    else:
        locations = np.asarray(log_locations, dtype=float)
        if locations.ndim != 1 or len(locations) < 3:
            raise ValueError("log_locations must contain at least three values")
        locations = np.sort(locations)

    means = component_means(locations, terminal_log_std)
    initial = _feasible_initial_weights(means, forward)
    design = component_call_price_matrix(k, locations, terminal_log_std, rate, maturity)
    second_difference = _second_difference_matrix(len(locations))

    price_scale = max(float(np.median(np.maximum(observed, 1e-6))), 1e-6)

    def objective(weights: np.ndarray) -> float:
        residual = (design @ weights - observed) / price_scale
        roughness = second_difference @ weights
        return float(np.mean(residual**2) + smoothness * np.dot(roughness, roughness))

    def gradient(weights: np.ndarray) -> np.ndarray:
        residual = (design @ weights - observed) / price_scale
        grad = 2.0 * (design.T @ residual) / (len(observed) * price_scale)
        if len(second_difference):
            grad += 2.0 * smoothness * (second_difference.T @ second_difference @ weights)
        return grad

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        {
            "type": "eq",
            "fun": lambda weights: float(np.dot(weights, means) - forward),
            "jac": lambda weights: means,
        },
    ]
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(locations),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": max_iterations, "disp": False},
    )
    if not result.success or result.x is None:
        raise CalibrationError(f"constrained calibration failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    weights[np.abs(weights) < 1e-12] = 0.0
    weights /= weights.sum()
    fitted = design @ weights
    moments = lognormal_mixture_moments(weights, locations, terminal_log_std)
    forward_error = moments["mean"] - forward
    tolerance = 1e-7 * max(1.0, forward)
    if np.any(weights < -1e-10) or abs(forward_error) > tolerance:
        raise CalibrationError("optimizer returned a solution that violates financial constraints")

    return FitResult(
        weights=weights,
        log_locations=locations,
        terminal_log_std=float(terminal_log_std),
        fitted_prices=fitted,
        objective=float(result.fun),
        iterations=int(result.nit),
        forward=float(forward),
        forward_error=float(forward_error),
        metrics=error_metrics(observed, fitted),
        moments=moments,
    )
