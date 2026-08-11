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


DownloadFn = Callable[[str, datetime, datetime, str, bool, bool], Any]
CurrencyFn = Callable[[str], str | None]
YAHOO_OHLC_REPAIR_TOLERANCE_PCT = Decimal("0.000001")
YAHOO_OHLC_NORMALIZATION_POLICY = "yahoo_rounding_tolerance"


class YahooFinanceDataProvider(MarketDataProvider):
    def __init__(
        self,
        cache_dir: str | Path = "data",
        *,
        adjustment_policy: str = "adjusted",
        yfinance_repair: bool = False,
        ohlc_repair_tolerance_pct: Decimal = YAHOO_OHLC_REPAIR_TOLERANCE_PCT,
        downloader: DownloadFn | None = None,
        currency_lookup: CurrencyFn | None = None,
    ) -> None:
        if adjustment_policy not in {"adjusted", "unadjusted"}:
            raise ValueError("adjustment_policy must be adjusted or unadjusted")
        self.cache_dir = Path(cache_dir)
        self.adjustment_policy = adjustment_policy
        self.yfinance_repair = yfinance_repair
        self.ohlc_repair_tolerance_pct = ohlc_repair_tolerance_pct
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
            raw_data = self.downloader(
                symbol,
                start,
                end,
                interval,
                self.adjustment_policy == "adjusted",
                self.yfinance_repair,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download historical data for {symbol}: {exc}") from exc

        normalized = _normalize_yahoo_data(
            raw_data,
            symbol,
            ohlc_repair_tolerance_pct=self.ohlc_repair_tolerance_pct,
        )
        quote_currency = self.currency_lookup(symbol)
        if quote_currency is None or quote_currency.strip() == "":
            raise ValueError(f"Unable to determine quote currency for {symbol}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.cache_dir / f"{_safe_filename(symbol)}_daily.csv"
        _write_candles_csv(csv_path, normalized.candles)
        save_dataset_metadata(
            csv_path,
            DatasetMetadata(
                symbol=symbol,
                quote_currency=quote_currency.upper(),
                source="yfinance",
                interval="1d",
                start_date=start.date().isoformat(),
                end_date=end.date().isoformat(),
                adjustment_policy=self.adjustment_policy,
                auto_adjust=self.adjustment_policy == "adjusted",
                yfinance_repair=self.yfinance_repair,
                ohlc_normalization_policy=YAHOO_OHLC_NORMALIZATION_POLICY,
                repaired_ohlc_rows=normalized.repaired_rows,
                largest_repaired_ohlc_violation_pct=str(normalized.largest_repaired_violation_pct),
            ),
        )
        return csv_path


def _download_with_yfinance(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    auto_adjust: bool,
    repair: bool,
) -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance with `python -m pip install -e .[market-data]`") from exc

    return yf.download(
        symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval=interval,
        auto_adjust=auto_adjust,
        repair=repair,
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


class NormalizedYahooData:
    def __init__(
        self,
        candles: list[Candle],
        repaired_rows: int,
        largest_repaired_violation_pct: Decimal,
    ) -> None:
        self.candles = candles
        self.repaired_rows = repaired_rows
        self.largest_repaired_violation_pct = largest_repaired_violation_pct


class NormalizedOHLC:
    def __init__(
        self,
        high: Decimal,
        low: Decimal,
        repaired: bool,
        violation_pct: Decimal,
    ) -> None:
        self.high = high
        self.low = low
        self.repaired = repaired
        self.violation_pct = violation_pct


def _normalize_yahoo_data(
    raw_data: Any,
    symbol: str,
    *,
    ohlc_repair_tolerance_pct: Decimal = YAHOO_OHLC_REPAIR_TOLERANCE_PCT,
) -> NormalizedYahooData:
    if raw_data is None or bool(getattr(raw_data, "empty", False)):
        raise ValueError(f"No historical data returned for {symbol}")

    candles: list[Candle] = []
    repaired_rows = 0
    largest_repaired_violation_pct = Decimal("0")
    for row_number, (timestamp, row) in enumerate(raw_data.iterrows(), start=1):
        try:
            timestamp_value = _normalize_timestamp(timestamp)
            open_price = _row_decimal(row, "Open")
            high_price = _row_decimal(row, "High")
            low_price = _row_decimal(row, "Low")
            close_price = _row_decimal(row, "Close")
            normalized_ohlc = _normalize_yahoo_ohlc(
                symbol=symbol,
                timestamp=timestamp_value,
                row_number=row_number,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                tolerance_pct=ohlc_repair_tolerance_pct,
            )
            if normalized_ohlc.repaired:
                repaired_rows += 1
                largest_repaired_violation_pct = max(
                    largest_repaired_violation_pct,
                    normalized_ohlc.violation_pct,
                )
            candle = Candle(
                symbol=symbol,
                timestamp=timestamp_value,
                open=open_price,
                high=normalized_ohlc.high,
                low=normalized_ohlc.low,
                close=close_price,
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

    return NormalizedYahooData(
        candles=sorted(candles, key=lambda candle: candle.timestamp),
        repaired_rows=repaired_rows,
        largest_repaired_violation_pct=largest_repaired_violation_pct,
    )


def _normalize_yahoo_ohlc(
    *,
    symbol: str,
    timestamp: datetime,
    row_number: int,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    tolerance_pct: Decimal,
) -> NormalizedOHLC:
    normalized_high = max(high_price, open_price, close_price, low_price)
    normalized_low = min(low_price, open_price, close_price, high_price)
    high_violation = normalized_high - high_price
    low_violation = low_price - normalized_low
    violation = max(high_violation, low_violation, Decimal("0"))
    if violation <= 0:
        return NormalizedOHLC(high=high_price, low=low_price, repaired=False, violation_pct=Decimal("0"))

    reference_price = max(abs(open_price), abs(close_price), Decimal("0.00000001"))
    violation_pct = (violation / reference_price) * Decimal("100")
    if violation_pct > tolerance_pct:
        raise ValueError(
            "Yahoo OHLC violation exceeds repair tolerance "
            f"for {symbol} on row {row_number} ({timestamp.isoformat()}): "
            f"open={open_price}, high={high_price}, low={low_price}, close={close_price}, "
            f"violation={violation}, violation_pct={violation_pct}, tolerance_pct={tolerance_pct}"
        )

    return NormalizedOHLC(
        high=normalized_high,
        low=normalized_low,
        repaired=True,
        violation_pct=violation_pct,
    )


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
