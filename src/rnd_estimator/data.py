"""Validated CSV input for reproducible, offline option-chain analysis."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class OptionChain:
    strikes: np.ndarray
    call_prices: np.ndarray

    def __post_init__(self) -> None:
        if len(self.strikes) != len(self.call_prices) or len(self.strikes) < 3:
            raise ValueError("an option chain needs at least three strike/price rows")
        if np.any(self.strikes <= 0) or np.any(self.call_prices < 0):
            raise ValueError("strikes must be positive and prices non-negative")
        if np.any(np.diff(self.strikes) <= 0):
            raise ValueError("strikes must be strictly increasing after normalization")


def _parse_optional_float(row: dict, name: str) -> Optional[float]:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {name!r} contains a non-numeric value: {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"column {name!r} must contain finite values")
    return parsed


def load_option_chain(path: Union[str, Path]) -> OptionChain:
    """Load ``strike`` plus ``call_price`` or ``bid``/``ask`` from CSV.

    Duplicate strikes are consolidated with the median price. Rows with an
    invalid two-sided quote (negative or ask below bid) fail loudly.
    """

    csv_path = Path(path)
    grouped: dict = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "strike" not in fields:
            raise ValueError("CSV must include a 'strike' column")
        if "call_price" not in fields and not {"bid", "ask"}.issubset(fields):
            raise ValueError("CSV must include 'call_price' or both 'bid' and 'ask'")

        for line_number, row in enumerate(reader, start=2):
            strike = _parse_optional_float(row, "strike")
            direct_price = _parse_optional_float(row, "call_price")
            bid = _parse_optional_float(row, "bid")
            ask = _parse_optional_float(row, "ask")
            if strike is None:
                raise ValueError(f"line {line_number}: strike is missing")
            if direct_price is not None:
                price = direct_price
            elif bid is not None and ask is not None:
                if bid < 0 or ask < bid:
                    raise ValueError(f"line {line_number}: invalid bid/ask quote")
                price = 0.5 * (bid + ask)
            else:
                raise ValueError(f"line {line_number}: no usable call price")
            if strike <= 0 or price < 0:
                raise ValueError(
                    f"line {line_number}: strike must be positive and price non-negative"
                )
            grouped.setdefault(strike, []).append(price)

    if len(grouped) < 3:
        raise ValueError("CSV must contain at least three unique strikes")
    normalized: list[tuple[float, float]] = [
        (strike, float(np.median(prices))) for strike, prices in sorted(grouped.items())
    ]
    return OptionChain(
        strikes=np.asarray([item[0] for item in normalized], dtype=float),
        call_prices=np.asarray([item[1] for item in normalized], dtype=float),
    )
