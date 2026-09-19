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
from .reporting import save_fit_report, save_simulation_report
from .simulation import SimulationConfig, run_simulation


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
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--spot", type=float, required=True)
    fit.add_argument("--maturity", type=float, required=True, help="time to expiry in years")
    fit.add_argument("--rate", type=float, required=True, help="continuously compounded rate")
    fit.add_argument("--dividend-yield", type=float, default=0.0)
    fit.add_argument("--annual-volatility", type=float, required=True)
    fit.add_argument("--components", type=int, default=9)
    fit.add_argument("--width", type=float, default=3.0)
    fit.add_argument("--smoothness", type=float, default=1e-4)
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts/fit"))
    fit.add_argument("--no-plot", action="store_true")
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _run_simulate(args) if args.command == "simulate" else _run_fit(args)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
