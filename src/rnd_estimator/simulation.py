"""Reproducible Monte Carlo validation for the density estimator."""

from dataclasses import dataclass

import numpy as np

from .calibration import CalibrationError, fit_lognormal_mixture
from .distributions import lognormal_mixture_pdf
from .pricing import lognormal_mixture_call_prices


@dataclass(frozen=True)
class SimulationConfig:
    spot: float = 100.0
    maturity: float = 30.0 / 365.0
    rate: float = 0.04
    dividend_yield: float = 0.01
    annual_volatility: float = 0.24
    n_strikes: int = 25
    n_components: int = 9
    n_runs: int = 100
    noise_relative: float = 0.01
    noise_floor: float = 0.002
    seed: int = 42

    @property
    def forward(self) -> float:
        return self.spot * np.exp((self.rate - self.dividend_yield) * self.maturity)

    @property
    def terminal_log_std(self) -> float:
        return self.annual_volatility * np.sqrt(self.maturity)

    def validate(self) -> None:
        if self.spot <= 0 or self.maturity <= 0 or self.annual_volatility <= 0:
            raise ValueError("spot, maturity and annual_volatility must be positive")
        if self.n_strikes < 5 or self.n_components < 3 or self.n_runs < 1:
            raise ValueError("use at least 5 strikes, 3 components and 1 run")
        if self.noise_relative < 0 or self.noise_floor < 0:
            raise ValueError("noise parameters cannot be negative")


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    strikes: np.ndarray
    true_prices: np.ndarray
    mean_fitted_prices: np.ndarray
    price_lower: np.ndarray
    price_upper: np.ndarray
    density_grid: np.ndarray
    true_density: np.ndarray
    mean_density: np.ndarray
    density_lower: np.ndarray
    density_upper: np.ndarray
    run_rmse: np.ndarray
    successful_runs: int
    failed_runs: int

    def summary(self) -> dict[str, object]:
        return {
            "seed": self.config.seed,
            "requested_runs": self.config.n_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "mean_rmse": float(np.mean(self.run_rmse)),
            "median_rmse": float(np.median(self.run_rmse)),
            "price_coverage_95": float(
                np.mean(
                    (self.true_prices >= self.price_lower) & (self.true_prices <= self.price_upper)
                )
            ),
        }


def _true_mixture(config: SimulationConfig) -> tuple:
    """Build a skewed, fat-tailed mixture with an exact forward moment."""

    terminal_std = 0.65 * config.terminal_log_std
    mean_ratios = np.asarray([0.72, 0.86, 1.0, 1.18, 1.42])
    base = np.asarray([0.08, 0.19, 0.46, 0.19, 0.08])

    # Exponential tilting preserves positivity while imposing E[S_T]/F = 1.
    from scipy.optimize import brentq

    def tilted_mean(tilt: float) -> float:
        unnormalized = base * np.exp(tilt * mean_ratios)
        weights = unnormalized / unnormalized.sum()
        return float(np.dot(weights, mean_ratios) - 1.0)

    tilt = brentq(tilted_mean, -100.0, 100.0)
    unnormalized = base * np.exp(tilt * mean_ratios)
    weights = unnormalized / unnormalized.sum()
    means = config.forward * mean_ratios
    locations = np.log(means) - 0.5 * terminal_std**2
    return weights, locations, terminal_std


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run independent noisy calibrations using one explicit random seed."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    strikes = np.linspace(0.72 * config.spot, 1.32 * config.spot, config.n_strikes)
    true_weights, true_locations, true_std = _true_mixture(config)
    true_prices = lognormal_mixture_call_prices(
        strikes,
        true_weights,
        true_locations,
        true_std,
        config.rate,
        config.maturity,
    )
    density_grid = np.linspace(0.35 * config.spot, 2.0 * config.spot, 600)
    true_density = lognormal_mixture_pdf(
        density_grid,
        true_weights,
        true_locations,
        true_std,
    )

    fitted_price_samples = []
    density_samples = []
    rmse_samples = []
    failed_runs = 0
    for _ in range(config.n_runs):
        noise_std = (
            config.noise_relative * np.maximum(true_prices, 0.01) + config.noise_floor * config.spot
        )
        observed = np.maximum(true_prices + rng.normal(0.0, noise_std), 0.0)
        try:
            fit = fit_lognormal_mixture(
                strikes,
                observed,
                config.spot,
                config.maturity,
                config.rate,
                config.dividend_yield,
                config.terminal_log_std,
                n_components=config.n_components,
            )
        except CalibrationError:
            failed_runs += 1
            continue
        fitted_price_samples.append(fit.fitted_prices)
        density_samples.append(
            lognormal_mixture_pdf(
                density_grid,
                fit.weights,
                fit.log_locations,
                fit.terminal_log_std,
            )
        )
        rmse_samples.append(float(np.sqrt(np.mean((fit.fitted_prices - true_prices) ** 2))))

    if not fitted_price_samples:
        raise CalibrationError("all simulation calibrations failed")
    prices = np.asarray(fitted_price_samples)
    densities = np.asarray(density_samples)
    return SimulationResult(
        config=config,
        strikes=strikes,
        true_prices=true_prices,
        mean_fitted_prices=np.mean(prices, axis=0),
        price_lower=np.quantile(prices, 0.025, axis=0),
        price_upper=np.quantile(prices, 0.975, axis=0),
        density_grid=density_grid,
        true_density=true_density,
        mean_density=np.mean(densities, axis=0),
        density_lower=np.quantile(densities, 0.025, axis=0),
        density_upper=np.quantile(densities, 0.975, axis=0),
        run_rmse=np.asarray(rmse_samples),
        successful_runs=len(prices),
        failed_runs=failed_runs,
    )
