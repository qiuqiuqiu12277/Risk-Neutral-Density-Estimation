"""Static-arbitrage and calibration diagnostics."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ArbitrageDiagnostics:
    lower_bound_violations: int
    upper_bound_violations: int
    monotonicity_violations: int
    convexity_violations: int

    @property
    def is_arbitrage_free(self) -> bool:
        return all(value == 0 for value in asdict(self).values())

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = dict(asdict(self))
        result["is_arbitrage_free"] = self.is_arbitrage_free
        return result


def call_arbitrage_diagnostics(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-8,
) -> ArbitrageDiagnostics:
    """Check call bounds, strike monotonicity and convexity on an irregular grid."""

    k = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if k.ndim != 1 or prices.ndim != 1 or len(k) != len(prices) or len(k) < 3:
        raise ValueError(
            "strikes and call_prices must be equal-length vectors with at least 3 rows"
        )
    order = np.argsort(k)
    k = k[order]
    prices = prices[order]
    if np.any(np.diff(k) <= 0):
        raise ValueError("strikes must be unique")

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    lower = np.maximum(discounted_spot - k * np.exp(-rate * maturity), 0.0)
    upper = np.full_like(k, discounted_spot)
    slopes = np.diff(prices) / np.diff(k)
    return ArbitrageDiagnostics(
        lower_bound_violations=int(np.sum(prices < lower - tolerance)),
        upper_bound_violations=int(np.sum(prices > upper + tolerance)),
        monotonicity_violations=int(np.sum(slopes > tolerance)),
        convexity_violations=int(np.sum(np.diff(slopes) < -tolerance)),
    )


def error_metrics(observed: ArrayLike, fitted: ArrayLike) -> dict[str, float]:
    observed_array = np.asarray(observed, dtype=float)
    fitted_array = np.asarray(fitted, dtype=float)
    if observed_array.shape != fitted_array.shape:
        raise ValueError("observed and fitted values must have the same shape")
    residual = fitted_array - observed_array
    scale = np.maximum(np.abs(observed_array), 1e-8)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual) / scale)),
        "max_absolute_error": float(np.max(np.abs(residual))),
    }
