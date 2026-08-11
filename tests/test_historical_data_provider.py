from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.data.market_data import HistoricalDataProvider, load_csv_candles


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_csv_candles_orders_rows_chronologically() -> None:
    candles = load_csv_candles(FIXTURES / "sample_ohlcv.csv", "ABC")

    assert [candle.timestamp for candle in candles] == sorted(candle.timestamp for candle in candles)
    assert candles[0].close == Decimal("100")
    assert candles[-1].close == Decimal("90")


def test_historical_data_provider_filters_date_range() -> None:
    provider = HistoricalDataProvider(FIXTURES / "sample_ohlcv.csv")

    candles = provider.historical_candles(
        "ABC",
        start=datetime(2024, 1, 2),
        end=datetime(2024, 1, 4),
    )

    assert [candle.close for candle in candles] == [
        Decimal("105"),
        Decimal("110"),
        Decimal("100"),
    ]


def test_load_csv_candles_reports_malformed_rows() -> None:
    with pytest.raises(ValueError, match="row 2"):
        load_csv_candles(FIXTURES / "malformed_ohlcv.csv", "ABC")


def test_load_csv_candles_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing_columns.csv"
    path.write_text("timestamp,open,high,low,close\n2024-01-01T00:00:00,1,1,1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing columns"):
        load_csv_candles(path, "ABC")


def test_load_csv_candles_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="Duplicate timestamp"):
        load_csv_candles(FIXTURES / "duplicate_ohlcv.csv", "ABC")
