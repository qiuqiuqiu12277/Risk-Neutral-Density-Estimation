"""Deterministic strike-holdout model selection and benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from .black_scholes import black_scholes_call
from .calibration import CalibrationError, fit_lognormal_mixture
from .diagnostics import error_metrics
from .pricing import lognormal_mixture_call_prices


@dataclass(frozen=True)
class ModelComparisonResult:
    """Cross-validation results for mixture candidates and Black--Scholes."""

    selected_components: int
    recommended_model: str
    mixture_rmse_improvement_vs_black_scholes_pct: float | None
    folds: int
    split_strategy: str
    candidate_results: list[dict[str, object]]
    black_scholes: dict[str, object]
    fold_results: list[dict[str, object]]

    def to_summary(self) -> dict[str, object]:
        return {
            "selected_components": self.selected_components,
            "recommended_model": self.recommended_model,
            "mixture_rmse_improvement_vs_black_scholes_pct": (
                self.mixture_rmse_improvement_vs_black_scholes_pct
            ),
            "folds": self.folds,
            "split_strategy": self.split_strategy,
            "selection_metric": "validation_rmse",
            "candidate_results": self.candidate_results,
            "black_scholes_baseline": self.black_scholes,
        }


def fit_black_scholes_volatility(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    lower: float = 1e-4,
    upper: float = 3.0,
) -> float:
    """Fit one constant Black--Scholes volatility by least squares."""

    k = np.asarray(strikes, dtype=float)
    observed = np.asarray(call_prices, dtype=float)
    if k.ndim != 1 or observed.ndim != 1 or len(k) != len(observed) or len(k) < 3:
        raise ValueError("strikes and call_prices must be equal-length vectors with 3+ rows")
    if lower <= 0 or upper <= lower:
        raise ValueError("volatility bounds must satisfy 0 < lower < upper")
    scale = max(float(np.median(np.maximum(observed, 1e-6))), 1e-6)

    def objective(volatility: float) -> float:
        fitted = np.asarray(
            black_scholes_call(
                spot,
                k,
                maturity,
                rate,
                volatility,
                dividend_yield,
            )
        )
        return float(np.mean(((fitted - observed) / scale) ** 2))

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1_000},
    )
    if not result.success or not np.isfinite(result.x):
        raise CalibrationError(f"Black--Scholes calibration failed: {result.message}")
    return float(result.x)


def _fold_indices(size: int, folds: int) -> list[np.ndarray]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if folds > size:
        raise ValueError("folds cannot exceed the number of strikes")
    result = [np.arange(fold, size, folds, dtype=int) for fold in range(folds)]
    if min(size - len(validation) for validation in result) < 3:
        raise ValueError("each training fold must contain at least three strikes")
    return result


def compare_models(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    terminal_log_std: float,
    component_candidates: ArrayLike = (5, 7, 9, 11),
    folds: int = 4,
    width: float = 3.0,
    smoothness: float = 1e-4,
) -> ModelComparisonResult:
    """Select mixture complexity using deterministic interleaved strike folds.

    Every validation fold spans the strike range instead of withholding one
    contiguous tail. This measures interpolation quality while keeping the
    procedure fully deterministic and suitable for small option cross-sections.
    """

    k = np.asarray(strikes, dtype=float)
    observed = np.asarray(call_prices, dtype=float)
    if k.ndim != 1 or observed.ndim != 1 or len(k) != len(observed):
        raise ValueError("strikes and call_prices must be equal-length vectors")
    if len(k) < 5:
        raise ValueError("model comparison requires at least five strikes")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(observed)):
        raise ValueError("strikes and call_prices must be finite")

    candidates = sorted({int(value) for value in np.asarray(component_candidates).tolist()})
    if not candidates or candidates[0] < 3:
        raise ValueError("component candidates must contain integers of at least 3")

    order = np.argsort(k)
    k = k[order]
    observed = observed[order]
    validation_folds = _fold_indices(len(k), folds)
    all_indices = np.arange(len(k))
    fold_results: list[dict[str, object]] = []

    baseline_observed: list[np.ndarray] = []
    baseline_predictions: list[np.ndarray] = []
    baseline_volatilities: list[float] = []
    for fold_number, validation_idx in enumerate(validation_folds, start=1):
        train_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=True)
        volatility = fit_black_scholes_volatility(
            k[train_idx],
            observed[train_idx],
            spot,
            maturity,
            rate,
            dividend_yield,
        )
        train_prediction = np.asarray(
            black_scholes_call(spot, k[train_idx], maturity, rate, volatility, dividend_yield)
        )
        validation_prediction = np.asarray(
            black_scholes_call(spot, k[validation_idx], maturity, rate, volatility, dividend_yield)
        )
        train_metrics = error_metrics(observed[train_idx], train_prediction)
        validation_metrics = error_metrics(observed[validation_idx], validation_prediction)
        baseline_observed.append(observed[validation_idx])
        baseline_predictions.append(validation_prediction)
        baseline_volatilities.append(volatility)
        fold_results.append(
            {
                "model": "black_scholes",
                "components": None,
                "fold": fold_number,
                "status": "ok",
                "train_records": int(len(train_idx)),
                "validation_records": int(len(validation_idx)),
                "annual_volatility": volatility,
                "train_rmse": train_metrics["rmse"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_mae": validation_metrics["mae"],
            }
        )

    baseline_metrics = error_metrics(
        np.concatenate(baseline_observed), np.concatenate(baseline_predictions)
    )
    black_scholes = {
        "successful_folds": folds,
        "failed_folds": 0,
        "validation_metrics": baseline_metrics,
        "mean_fitted_annual_volatility": float(np.mean(baseline_volatilities)),
    }

    candidate_results: list[dict[str, object]] = []
    for components in candidates:
        validation_observed: list[np.ndarray] = []
        validation_predictions: list[np.ndarray] = []
        failed_folds = 0
        for fold_number, validation_idx in enumerate(validation_folds, start=1):
            train_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=True)
            try:
                fit = fit_lognormal_mixture(
                    k[train_idx],
                    observed[train_idx],
                    spot,
                    maturity,
                    rate,
                    dividend_yield,
                    terminal_log_std,
                    n_components=components,
                    width=width,
                    smoothness=smoothness,
                )
                prediction = lognormal_mixture_call_prices(
                    k[validation_idx],
                    fit.weights,
                    fit.log_locations,
                    fit.terminal_log_std,
                    rate,
                    maturity,
                )
                metrics = error_metrics(observed[validation_idx], prediction)
                validation_observed.append(observed[validation_idx])
                validation_predictions.append(prediction)
                fold_results.append(
                    {
                        "model": "lognormal_mixture",
                        "components": components,
                        "fold": fold_number,
                        "status": "ok",
                        "train_records": int(len(train_idx)),
                        "validation_records": int(len(validation_idx)),
                        "annual_volatility": None,
                        "train_rmse": fit.metrics["rmse"],
                        "validation_rmse": metrics["rmse"],
                        "validation_mae": metrics["mae"],
                    }
                )
            except (CalibrationError, ValueError) as exc:
                failed_folds += 1
                fold_results.append(
                    {
                        "model": "lognormal_mixture",
                        "components": components,
                        "fold": fold_number,
                        "status": "failed",
                        "train_records": int(len(train_idx)),
                        "validation_records": int(len(validation_idx)),
                        "annual_volatility": None,
                        "train_rmse": None,
                        "validation_rmse": None,
                        "validation_mae": None,
                        "error": str(exc),
                    }
                )

        metrics = (
            error_metrics(
                np.concatenate(validation_observed), np.concatenate(validation_predictions)
            )
            if validation_observed
            else None
        )
        candidate_results.append(
            {
                "components": components,
                "successful_folds": folds - failed_folds,
                "failed_folds": failed_folds,
                "validation_metrics": metrics,
            }
        )

    eligible = [
        result
        for result in candidate_results
        if result["failed_folds"] == 0 and result["validation_metrics"] is not None
    ]
    if not eligible:
        raise CalibrationError("no component candidate completed every cross-validation fold")
    selected = min(
        eligible,
        key=lambda result: (
            result["validation_metrics"]["rmse"],  # type: ignore[index]
            result["components"],
        ),
    )
    selected_rmse = float(selected["validation_metrics"]["rmse"])  # type: ignore[index]
    baseline_rmse = float(baseline_metrics["rmse"])
    improvement = (
        100.0 * (baseline_rmse - selected_rmse) / baseline_rmse if baseline_rmse > 0 else None
    )
    return ModelComparisonResult(
        selected_components=int(selected["components"]),
        recommended_model=(
            "lognormal_mixture" if selected_rmse < baseline_rmse else "black_scholes"
        ),
        mixture_rmse_improvement_vs_black_scholes_pct=improvement,
        folds=folds,
        split_strategy="deterministic_interleaved_strikes",
        candidate_results=candidate_results,
        black_scholes=black_scholes,
        fold_results=fold_results,
    )
