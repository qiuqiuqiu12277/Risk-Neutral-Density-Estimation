"""Analytic option pricing under lognormal mixture distributions."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from .distributions import validate_mixture


def component_means(log_locations: ArrayLike, terminal_log_std: float) -> np.ndarray:
    """Return ``E[S_T]`` for every lognormal component."""

    locations = np.asarray(log_locations, dtype=float)
    if terminal_log_std <= 0:
        raise ValueError("terminal_log_std must be positive")
    return np.exp(locations + 0.5 * terminal_log_std**2)


def component_call_price_matrix(
    strikes: ArrayLike,
    log_locations: ArrayLike,
    terminal_log_std: float,
    rate: float,
    maturity: float,
) -> np.ndarray:
    """Return discounted call prices with shape ``(n_strikes, n_components)``."""

    strike_array = np.asarray(strikes, dtype=float)
    locations = np.asarray(log_locations, dtype=float)
    if strike_array.ndim != 1 or locations.ndim != 1:
        raise ValueError("strikes and log_locations must be one-dimensional")
    if np.any(strike_array <= 0) or not np.all(np.isfinite(strike_array)):
        raise ValueError("strikes must be positive and finite")
    if terminal_log_std <= 0:
        raise ValueError("terminal_log_std must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    means = component_means(locations, terminal_log_std)
    d2 = (locations[None, :] - np.log(strike_array[:, None])) / terminal_log_std
    undiscounted = means[None, :] * ndtr(d2 + terminal_log_std) - strike_array[:, None] * ndtr(d2)
    return np.exp(-rate * maturity) * undiscounted


def lognormal_mixture_call_prices(
    strikes: ArrayLike,
    weights: ArrayLike,
    log_locations: ArrayLike,
    terminal_log_std: float,
    rate: float,
    maturity: float,
) -> np.ndarray:
    """Price European calls analytically under a lognormal mixture RND."""

    validate_mixture(weights, log_locations, terminal_log_std)
    matrix = component_call_price_matrix(
        strikes,
        log_locations,
        terminal_log_std,
        rate,
        maturity,
    )
    return matrix @ np.asarray(weights, dtype=float)
