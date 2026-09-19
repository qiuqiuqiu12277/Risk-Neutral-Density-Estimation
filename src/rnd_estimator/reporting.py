"""Machine-readable outputs and publication-ready diagnostic plots."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np

from .calibration import FitResult
from .distributions import lognormal_mixture_pdf
from .model_selection import ModelComparisonResult
from .simulation import SimulationResult


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_simulation_report(
    result: SimulationResult,
    output_dir: Union[str, Path],
    make_plot: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    prices_path = output / "price_results.csv"
    _write_json(summary_path, {"config": asdict(result.config), "metrics": result.summary()})

    with prices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strike", "true_price", "mean_fitted_price", "lower_95", "upper_95"])
        writer.writerows(
            zip(
                result.strikes,
                result.true_prices,
                result.mean_fitted_prices,
                result.price_lower,
                result.price_upper,
            )
        )

    paths = {"summary": summary_path, "prices": prices_path}
    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_path = output / "simulation_diagnostics.png"
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(result.strikes, result.true_prices, "o-", color="black", label="True")
        axes[0].plot(
            result.strikes,
            result.mean_fitted_prices,
            color="#d62728",
            label="Mean estimate",
        )
        axes[0].fill_between(
            result.strikes,
            result.price_lower,
            result.price_upper,
            color="#d62728",
            alpha=0.18,
            label="95% Monte Carlo interval",
        )
        axes[0].set(xlabel="Strike", ylabel="Call price", title="Option-price recovery")
        axes[0].legend()

        axes[1].plot(
            result.density_grid,
            result.true_density,
            color="black",
            linestyle="--",
            label="True RND",
        )
        axes[1].plot(
            result.density_grid,
            result.mean_density,
            color="#1f77b4",
            label="Mean estimate",
        )
        axes[1].fill_between(
            result.density_grid,
            result.density_lower,
            result.density_upper,
            color="#1f77b4",
            alpha=0.18,
            label="95% Monte Carlo interval",
        )
        axes[1].set(xlabel="Terminal asset price", ylabel="Density", title="Density recovery")
        axes[1].legend()
        figure.suptitle(f"Constrained RND estimation (seed={result.config.seed})")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths["figure"] = figure_path
    return paths


def save_fit_report(
    fit: FitResult,
    strikes: np.ndarray,
    observed_prices: np.ndarray,
    output_dir: Union[str, Path],
    diagnostics: dict[str, object],
    make_plot: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "fit_summary.json"
    prices_path = output / "fitted_prices.csv"
    payload = fit.to_summary()
    payload["input_arbitrage_diagnostics"] = diagnostics
    _write_json(summary_path, payload)

    with prices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strike", "observed_call_price", "fitted_call_price", "residual"])
        writer.writerows(
            zip(strikes, observed_prices, fit.fitted_prices, fit.fitted_prices - observed_prices)
        )

    paths = {"summary": summary_path, "prices": prices_path}
    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_path = output / "fit_diagnostics.png"
        grid = np.linspace(max(1e-6, 0.35 * strikes.min()), 1.8 * strikes.max(), 700)
        density = lognormal_mixture_pdf(
            grid,
            fit.weights,
            fit.log_locations,
            fit.terminal_log_std,
        )
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(strikes, observed_prices, color="black", label="Observed", zorder=3)
        axes[0].plot(strikes, fit.fitted_prices, color="#d62728", label="Mixture fit")
        axes[0].set(xlabel="Strike", ylabel="Call price", title="Observed vs fitted prices")
        axes[0].legend()
        axes[1].plot(grid, density, color="#1f77b4")
        axes[1].axvline(fit.forward, color="black", linestyle="--", label="Forward")
        axes[1].set(xlabel="Terminal asset price", ylabel="Density", title="Estimated RND")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths["figure"] = figure_path
    return paths


def save_model_comparison_report(
    result: ModelComparisonResult,
    output_dir: Union[str, Path],
) -> dict[str, Path]:
    """Write model-selection summary and fold-level audit data."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "model_comparison.json"
    folds_path = output / "cross_validation_folds.csv"
    _write_json(summary_path, result.to_summary())

    fieldnames = [
        "model",
        "components",
        "fold",
        "status",
        "train_records",
        "validation_records",
        "annual_volatility",
        "train_rmse",
        "validation_rmse",
        "validation_mae",
        "error",
    ]
    with folds_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.fold_results:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return {"comparison": summary_path, "folds": folds_path}
