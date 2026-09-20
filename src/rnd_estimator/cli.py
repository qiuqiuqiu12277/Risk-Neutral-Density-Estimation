"""Command-line interface for offline calibration and reproducible simulation."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration import CalibrationError, fit_lognormal_mixture
from .data import load_option_chain
from .diagnostics import call_arbitrage_diagnostics
from .model_selection import compare_models
from .reporting import (
    save_fit_report,
    save_model_comparison_report,
    save_simulation_report,
)
from .simulation import SimulationConfig, run_simulation


def _add_chain_arguments(parser: argparse.ArgumentParser, default_output: Path) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--maturity", type=float, required=True, help="time to expiry in years")
    parser.add_argument("--rate", type=float, required=True, help="continuously compounded rate")
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--annual-volatility", type=float, required=True)
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--smoothness", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--no-plot", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rnd-estimate",
        description="Estimate arbitrage-aware risk-neutral densities from call option prices.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="run a reproducible Monte Carlo study")
    simulate.add_argument("--runs", type=int, default=100)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--components", type=int, default=9)
    simulate.add_argument("--output-dir", type=Path, default=Path("artifacts/simulation"))
    simulate.add_argument("--no-plot", action="store_true")

    fit = subparsers.add_parser("fit", help="fit a density to an offline option-chain CSV")
    _add_chain_arguments(fit, Path("artifacts/fit"))
    fit.add_argument("--components", type=int, default=9)

    compare = subparsers.add_parser(
        "compare",
        help="cross-validate mixture complexity and compare with Black--Scholes",
    )
    _add_chain_arguments(compare, Path("artifacts/model-comparison"))
    compare.add_argument(
        "--component-candidates",
        type=int,
        nargs="+",
        default=[5, 7, 9, 11],
        metavar="N",
    )
    compare.add_argument("--folds", type=int, default=4)
    return parser


def _run_simulate(args: argparse.Namespace) -> dict:
    config = SimulationConfig(
        n_runs=args.runs,
        seed=args.seed,
        n_components=args.components,
    )
    result = run_simulation(config)
    paths = save_simulation_report(result, args.output_dir, make_plot=not args.no_plot)
    return {
        "metrics": result.summary(),
        "outputs": {name: str(path) for name, path in paths.items()},
    }


def _run_fit(args: argparse.Namespace) -> dict:
    chain = load_option_chain(args.input)
    input_diagnostics = call_arbitrage_diagnostics(
        chain.strikes,
        chain.call_prices,
        args.spot,
        args.maturity,
        args.rate,
        args.dividend_yield,
    ).to_dict()
    fit = fit_lognormal_mixture(
        chain.strikes,
        chain.call_prices,
        args.spot,
        args.maturity,
        args.rate,
        args.dividend_yield,
        args.annual_volatility * np.sqrt(args.maturity),
        n_components=args.components,
        width=args.width,
        smoothness=args.smoothness,
    )
    paths = save_fit_report(
        fit,
        chain.strikes,
        chain.call_prices,
        args.output_dir,
        input_diagnostics,
        make_plot=not args.no_plot,
    )
    return {
        "fit": fit.to_summary(),
        "input_arbitrage_diagnostics": input_diagnostics,
        "outputs": {name: str(path) for name, path in paths.items()},
    }


def _run_compare(args: argparse.Namespace) -> dict:
    chain = load_option_chain(args.input)
    input_diagnostics = call_arbitrage_diagnostics(
        chain.strikes,
        chain.call_prices,
        args.spot,
        args.maturity,
        args.rate,
        args.dividend_yield,
    ).to_dict()
    terminal_log_std = args.annual_volatility * np.sqrt(args.maturity)
    comparison = compare_models(
        chain.strikes,
        chain.call_prices,
        args.spot,
        args.maturity,
        args.rate,
        args.dividend_yield,
        terminal_log_std,
        component_candidates=args.component_candidates,
        folds=args.folds,
        width=args.width,
        smoothness=args.smoothness,
    )
    fit = fit_lognormal_mixture(
        chain.strikes,
        chain.call_prices,
        args.spot,
        args.maturity,
        args.rate,
        args.dividend_yield,
        terminal_log_std,
        n_components=comparison.selected_components,
        width=args.width,
        smoothness=args.smoothness,
    )
    paths = save_model_comparison_report(comparison, args.output_dir)
    paths.update(
        save_fit_report(
            fit,
            chain.strikes,
            chain.call_prices,
            args.output_dir,
            input_diagnostics,
            make_plot=not args.no_plot,
        )
    )
    return {
        "model_comparison": comparison.to_summary(),
        "selected_mixture_fit": fit.to_summary(),
        "input_arbitrage_diagnostics": input_diagnostics,
        "outputs": {name: str(path) for name, path in paths.items()},
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            payload = _run_simulate(args)
        elif args.command == "fit":
            payload = _run_fit(args)
        else:
            payload = _run_compare(args)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
