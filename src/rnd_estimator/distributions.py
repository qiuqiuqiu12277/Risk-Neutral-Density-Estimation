"""Lognormal mixture distributions and analytic moments."""

import numpy as np
from numpy.typing import ArrayLike


def validate_mixture(weights: ArrayLike, log_locations: ArrayLike, terminal_log_std: float) -> None:
    weights_array = np.asarray(weights, dtype=float)
    locations = np.asarray(log_locations, dtype=float)
    if weights_array.ndim != 1 or locations.ndim != 1:
        raise ValueError("weights and log_locations must be one-dimensional")
    if len(weights_array) == 0 or len(weights_array) != len(locations):
        raise ValueError("weights and log_locations must have equal, non-zero length")
    if not np.all(np.isfinite(weights_array)) or not np.all(np.isfinite(locations)):
        raise ValueError("mixture parameters must be finite")
    if np.any(weights_array < -1e-12):
        raise ValueError("mixture weights cannot be negative")
    if not np.isclose(weights_array.sum(), 1.0, atol=1e-8):
        raise ValueError("mixture weights must sum to one")
    if terminal_log_std <= 0 or not np.isfinite(terminal_log_std):
        raise ValueError("terminal_log_std must be positive and finite")


def lognormal_pdf(values: ArrayLike, log_location: float, terminal_log_std: float) -> np.ndarray:
    """Evaluate a lognormal density parameterised on the log-price scale."""

    if terminal_log_std <= 0:
        raise ValueError("terminal_log_std must be positive")
    x = np.asarray(values, dtype=float)
    density = np.zeros_like(x, dtype=float)
    positive = x > 0
    z = (np.log(x[positive]) - log_location) / terminal_log_std
    density[positive] = np.exp(-0.5 * z**2) / (
        x[positive] * terminal_log_std * np.sqrt(2.0 * np.pi)
    )
    return density


def lognormal_mixture_pdf(
    values: ArrayLike,
    weights: ArrayLike,
    log_locations: ArrayLike,
    terminal_log_std: float,
) -> np.ndarray:
    """Evaluate a normalized mixture risk-neutral density."""

    validate_mixture(weights, log_locations, terminal_log_std)
    x = np.asarray(values, dtype=float)
    mixture = np.zeros_like(x, dtype=float)
    for weight, location in zip(weights, log_locations):
        mixture += weight * lognormal_pdf(x, float(location), terminal_log_std)
    return mixture


def raw_moment(
    order: int,
    weights: ArrayLike,
    log_locations: ArrayLike,
    terminal_log_std: float,
) -> float:
    """Return ``E[S_T**order]`` analytically for the mixture."""

    if order < 0:
        raise ValueError("order must be non-negative")
    validate_mixture(weights, log_locations, terminal_log_std)
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(log_locations, dtype=float)
    return float(np.dot(w, np.exp(order * mu + 0.5 * order**2 * terminal_log_std**2)))


def lognormal_mixture_moments(
    weights: ArrayLike,
    log_locations: ArrayLike,
    terminal_log_std: float,
) -> dict[str, float]:
    """Return mean, volatility, skewness and kurtosis of terminal price."""

    m1 = raw_moment(1, weights, log_locations, terminal_log_std)
    m2 = raw_moment(2, weights, log_locations, terminal_log_std)
    m3 = raw_moment(3, weights, log_locations, terminal_log_std)
    m4 = raw_moment(4, weights, log_locations, terminal_log_std)
    variance = max(m2 - m1**2, 0.0)
    standard_deviation = np.sqrt(variance)
    if standard_deviation <= np.finfo(float).eps:
        skewness = 0.0
        kurtosis = 3.0
    else:
        third_central = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
        fourth_central = m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4
        skewness = third_central / standard_deviation**3
        kurtosis = fourth_central / variance**2
    return {
        "mean": float(m1),
        "variance": float(variance),
        "standard_deviation": float(standard_deviation),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }
