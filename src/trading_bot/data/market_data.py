"""Market-data provider abstractions and offline historical CSV support."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_bot.data.models import Candle


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


class MarketDataProvider(ABC):
    @abstractmethod
    def historical_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        """Return historical candles for a symbol in ascending timestamp order."""


class HistoricalDataProvider(MarketDataProvider):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def historical_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        if end < start:
            raise ValueError("end must be greater than or equal to start")

        candles = load_csv_candles(self.path, symbol)
        return [
            candle
            for candle in candles
            if start <= candle.timestamp <= end
        ]


CsvMarketDataProvider = HistoricalDataProvider


def load_csv_candles(path: str | Path, symbol: str) -> list[Candle]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Historical data file not found: {source}")

    with source.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {source}: {missing_columns}")

        candles = []
        for row_number, row in enumerate(reader, start=2):
            candles.append(_parse_candle_row(source, row_number, row, symbol))

    if not candles:
        raise ValueError(f"No candles found in {source}")

    return sorted(candles, key=lambda candle: candle.timestamp)


def _parse_candle_row(
    source: Path,
    row_number: int,
    row: dict[str, str | None],
    symbol: str,
) -> Candle:
    try:
        timestamp_text = _required_text(row, "timestamp")
        return Candle(
            symbol=symbol,
            timestamp=datetime.fromisoformat(timestamp_text),
            open=_required_decimal(row, "open"),
            high=_required_decimal(row, "high"),
            low=_required_decimal(row, "low"),
            close=_required_decimal(row, "close"),
            volume=_required_decimal(row, "volume"),
        )
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Malformed candle in {source} on row {row_number}: {exc}") from exc


def _required_text(row: dict[str, str | None], column: str) -> str:
    value = row.get(column)
    if value is None or value.strip() == "":
        raise ValueError(f"{column} is required")
    return value.strip()


def _required_decimal(row: dict[str, str | None], column: str) -> Decimal:
    return Decimal(_required_text(row, column))
