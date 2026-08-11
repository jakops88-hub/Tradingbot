"""Market-data loading helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from trading_bot.data.models import Bar


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_csv_bars(path: str | Path, symbol: str) -> list[Bar]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {source}: {missing_columns}")

        bars = [
            Bar(
                symbol=symbol,
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reader
        ]

    return sorted(bars, key=lambda bar: bar.timestamp)


def latest_price(bars: Iterable[Bar]) -> float:
    bars_list = list(bars)
    if not bars_list:
        raise ValueError("At least one bar is required")
    return bars_list[-1].close
