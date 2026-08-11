"""Market-data provider abstractions and offline CSV support."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
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


class CsvMarketDataProvider(MarketDataProvider):
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


def load_csv_candles(path: str | Path, symbol: str) -> list[Candle]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {source}: {missing_columns}")

        candles = [
            Candle(
                symbol=symbol,
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=Decimal(row["volume"]),
            )
            for row in reader
        ]

    return sorted(candles, key=lambda candle: candle.timestamp)
