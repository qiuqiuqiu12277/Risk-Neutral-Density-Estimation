"""Black--Scholes reference formulas used for diagnostics and benchmarks."""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import ndtr

Numeric = Union[float, np.ndarray]


def _return_like_input(value: np.ndarray, original: ArrayLike) -> Numeric:
    return float(value) if np.ndim(original) == 0 else value


def black_scholes_call(
    spot: float,
    strike: ArrayLike,
    maturity: float,
    rate: float,
    annual_volatility: float,
    dividend_yield: float = 0.0,
) -> Numeric:
    """Return European call prices under Black--Scholes.

    ``annual_volatility`` is an annualised diffusion volatility. This explicit
    naming prevents it from being confused with a terminal log standard
    deviation, which is ``annual_volatility * sqrt(maturity)``.
    """

    if spot <= 0:
        raise ValueError("spot must be positive")
    if maturity < 0:
        raise ValueError("maturity cannot be negative")
    if annual_volatility < 0:
        raise ValueError("annual_volatility cannot be negative")

    strikes = np.asarray(strike, dtype=float)
    if np.any(strikes <= 0):
        raise ValueError("all strikes must be positive")

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strikes * np.exp(-rate * maturity)
    if maturity == 0 or annual_volatility == 0:
        value = np.maximum(discounted_spot - discounted_strike, 0.0)
        return _return_like_input(value, strike)

    terminal_std = annual_volatility * np.sqrt(maturity)
    d1 = (
        np.log(spot / strikes) + (rate - dividend_yield) * maturity + 0.5 * terminal_std**2
    ) / terminal_std
    d2 = d1 - terminal_std
    value = discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2)
    return _return_like_input(value, strike)


def implied_volatility(
    call_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    """Invert a Black--Scholes call price with a bracketed root solver."""

    if maturity <= 0:
        raise ValueError("maturity must be positive")
    lower_bound = max(
        spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity),
        0.0,
    )
    upper_bound = spot * np.exp(-dividend_yield * maturity)
    tolerance = 1e-10 * max(1.0, spot)
    if call_price < lower_bound - tolerance or call_price > upper_bound + tolerance:
        raise ValueError("call price violates static no-arbitrage bounds")
    if abs(call_price - lower_bound) <= tolerance:
        return 0.0

    def pricing_error(volatility: float) -> float:
        return float(
            black_scholes_call(
                spot,
                strike,
                maturity,
                rate,
                volatility,
                dividend_yield,
            )
            - call_price
        )

    if pricing_error(upper) < 0:
        raise ValueError("implied volatility is above the configured upper bound")
    return float(brentq(pricing_error, lower, upper, xtol=1e-12, rtol=1e-12))
