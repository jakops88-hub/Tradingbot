"""Yahoo Finance historical data adapter.

This module is the only place that imports yfinance, and it does so lazily.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_bot.data.market_data import MarketDataProvider, load_csv_candles
from trading_bot.data.metadata import DatasetMetadata, save_dataset_metadata
from trading_bot.data.models import Candle


DownloadFn = Callable[[str, datetime, datetime, str], Any]
CurrencyFn = Callable[[str], str | None]


class YahooFinanceDataProvider(MarketDataProvider):
    def __init__(
        self,
        cache_dir: str | Path = "data",
        *,
        downloader: DownloadFn | None = None,
        currency_lookup: CurrencyFn | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.downloader = downloader or _download_with_yfinance
        self.currency_lookup = currency_lookup or _lookup_currency_with_yfinance

    def historical_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[Candle]:
        csv_path = self.download_to_csv(symbol=symbol, start=start, end=end)
        return load_csv_candles(csv_path, symbol)

    def download_to_csv(
        self,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> Path:
        if not symbol.strip():
            raise ValueError("symbol is required")
        if end <= start:
            raise ValueError("end must be greater than start")
        if interval != "1d":
            raise ValueError("only daily interval '1d' is currently supported")

        try:
            raw_data = self.downloader(symbol, start, end, interval)
        except Exception as exc:
            raise RuntimeError(f"Failed to download historical data for {symbol}: {exc}") from exc

        candles = _normalize_yahoo_data(raw_data, symbol)
        quote_currency = self.currency_lookup(symbol)
        if quote_currency is None or quote_currency.strip() == "":
            raise ValueError(f"Unable to determine quote currency for {symbol}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.cache_dir / f"{_safe_filename(symbol)}_daily.csv"
        _write_candles_csv(csv_path, candles)
        save_dataset_metadata(
            csv_path,
            DatasetMetadata(
                symbol=symbol,
                quote_currency=quote_currency.upper(),
                source="yfinance",
                interval="1d",
                start_date=start.date().isoformat(),
                end_date=end.date().isoformat(),
            ),
        )
        return csv_path


def _download_with_yfinance(symbol: str, start: datetime, end: datetime, interval: str) -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `python -m pip install -e .[market-data]`") from exc

    return yf.download(
        symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )


def _lookup_currency_with_yfinance(symbol: str) -> str | None:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `python -m pip install -e .[market-data]`") from exc

    ticker = yf.Ticker(symbol)
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        try:
            currency = fast_info.get("currency")
            if currency:
                return str(currency)
        except AttributeError:
            currency = getattr(fast_info, "currency", None)
            if currency:
                return str(currency)
    info = getattr(ticker, "info", None)
    if isinstance(info, dict):
        currency = info.get("currency")
        if currency:
            return str(currency)
    return None


def _normalize_yahoo_data(raw_data: Any, symbol: str) -> list[Candle]:
    if raw_data is None or bool(getattr(raw_data, "empty", False)):
        raise ValueError(f"No historical data returned for {symbol}")

    candles: list[Candle] = []
    for row_number, (timestamp, row) in enumerate(raw_data.iterrows(), start=1):
        try:
            candle = Candle(
                symbol=symbol,
                timestamp=_normalize_timestamp(timestamp),
                open=_row_decimal(row, "Open"),
                high=_row_decimal(row, "High"),
                low=_row_decimal(row, "Low"),
                close=_row_decimal(row, "Close"),
                volume=_row_decimal(row, "Volume"),
            )
        except (InvalidOperation, KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Malformed yfinance data for {symbol} on row {row_number}: {exc}") from exc
        candles.append(candle)

    if not candles:
        raise ValueError(f"No historical data returned for {symbol}")

    timestamps = [candle.timestamp for candle in candles]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"Duplicate timestamp returned for {symbol}")

    return sorted(candles, key=lambda candle: candle.timestamp)


def _normalize_timestamp(timestamp: Any) -> datetime:
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if isinstance(timestamp, datetime):
        return timestamp.replace(tzinfo=None)
    return datetime.fromisoformat(str(timestamp)).replace(tzinfo=None)


def _row_decimal(row: Any, column: str) -> Decimal:
    value = row[column]
    if hasattr(value, "item"):
        value = value.item()
    if value is None or str(value).strip() == "" or str(value).lower() == "nan":
        raise ValueError(f"{column} is required")
    return Decimal(str(value))


def _write_candles_csv(path: Path, candles: Sequence[Candle]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "timestamp": candle.timestamp.isoformat(),
                    "open": str(candle.open),
                    "high": str(candle.high),
                    "low": str(candle.low),
                    "close": str(candle.close),
                    "volume": str(candle.volume),
                }
            )


def _safe_filename(symbol: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in symbol)
