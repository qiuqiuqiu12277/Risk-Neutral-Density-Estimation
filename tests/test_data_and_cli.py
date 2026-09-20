import json
from pathlib import Path

import pytest

from rnd_estimator.cli import main
from rnd_estimator.data import load_option_chain

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sample_option_chain_loads_bid_ask_midpoints():
    chain = load_option_chain(PROJECT_ROOT / "data" / "sample_option_chain.csv")
    assert len(chain.strikes) == 13
    assert chain.call_prices[0] == pytest.approx(30.274567)


def test_simulation_cli_is_reproducible(tmp_path, capsys):
    output = tmp_path / "simulation"
    args = ["simulate", "--runs", "3", "--seed", "7", "--output-dir", str(output), "--no-plot"]
    assert main(args) == 0
    first = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    capsys.readouterr()
    assert main(args) == 0
    second = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert first == second
    assert first["metrics"]["successful_runs"] == 3


def test_fit_cli_writes_machine_readable_outputs(tmp_path):
    output = tmp_path / "fit"
    args = [
        "fit",
        "--input",
        str(PROJECT_ROOT / "data" / "sample_option_chain.csv"),
        "--spot",
        "100",
        "--maturity",
        "0.25",
        "--rate",
        "0.03",
        "--dividend-yield",
        "0.01",
        "--annual-volatility",
        "0.22",
        "--output-dir",
        str(output),
        "--no-plot",
    ]
    assert main(args) == 0
    summary = json.loads((output / "fit_summary.json").read_text(encoding="utf-8"))
    assert abs(summary["forward_error"]) < 1e-7
    assert (output / "fitted_prices.csv").exists()


def test_compare_cli_selects_components_and_writes_fold_audit(tmp_path):
    output = tmp_path / "comparison"
    args = [
        "compare",
        "--input",
        str(PROJECT_ROOT / "data" / "sample_option_chain.csv"),
        "--spot",
        "100",
        "--maturity",
        "0.25",
        "--rate",
        "0.03",
        "--dividend-yield",
        "0.01",
        "--annual-volatility",
        "0.22",
        "--component-candidates",
        "5",
        "7",
        "--folds",
        "3",
        "--output-dir",
        str(output),
        "--no-plot",
    ]
    assert main(args) == 0
    comparison = json.loads((output / "model_comparison.json").read_text(encoding="utf-8"))
    assert comparison["selected_components"] in {5, 7}
    assert comparison["black_scholes_baseline"]["validation_metrics"]["rmse"] >= 0
    assert (output / "cross_validation_folds.csv").is_file()
    assert (output / "fit_summary.json").is_file()
