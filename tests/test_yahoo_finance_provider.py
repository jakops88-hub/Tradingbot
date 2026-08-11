from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.data.market_data import load_csv_candles
from trading_bot.data.metadata import (
    DatasetMetadata,
    load_dataset_metadata,
    require_matching_currency,
    save_dataset_metadata,
)
from trading_bot.data.yahoo_finance import YahooFinanceDataProvider
from trading_bot.app import main


class FakeFrame:
    def __init__(self, rows: list[tuple[datetime, dict[str, object]]]) -> None:
        self.rows = rows
        self.empty = len(rows) == 0

    def iterrows(self):
        return iter(self.rows)


def fake_rows() -> list[tuple[datetime, dict[str, object]]]:
    return [
        (
            datetime(2024, 1, 2),
            {"Open": "101", "High": "102", "Low": "100", "Close": "101.5", "Volume": "2000"},
        ),
        (
            datetime(2024, 1, 1),
            {"Open": "100", "High": "101", "Low": "99", "Close": "100.5", "Volume": "1000"},
        ),
    ]


def test_downloaded_data_is_normalized_and_cached(tmp_path: Path) -> None:
    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        downloader=lambda symbol, start, end, interval, auto_adjust: FakeFrame(fake_rows()),
        currency_lookup=lambda symbol: "SEK",
    )

    csv_path = provider.download_to_csv(
        symbol="VOLV-B.ST",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
    )

    assert csv_path == tmp_path / "VOLV-B.ST_daily.csv"
    candles = load_csv_candles(csv_path, "VOLV-B.ST")
    assert [candle.timestamp for candle in candles] == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    assert candles[0].close == Decimal("100.5")
    assert candles[1].volume == Decimal("2000")


def test_downloaded_dataset_metadata_is_written(tmp_path: Path) -> None:
    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        downloader=lambda symbol, start, end, interval, auto_adjust: FakeFrame(fake_rows()),
        currency_lookup=lambda symbol: "sek",
    )

    csv_path = provider.download_to_csv(
        symbol="VOLV-B.ST",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
    )
    metadata = load_dataset_metadata(csv_path)

    assert metadata is not None
    assert metadata.symbol == "VOLV-B.ST"
    assert metadata.quote_currency == "SEK"
    assert metadata.source == "yfinance"
    assert metadata.interval == "1d"
    assert metadata.start_date == "2024-01-01"
    assert metadata.end_date == "2024-01-03"
    assert metadata.adjustment_policy == "adjusted"


def test_adjusted_price_policy_is_passed_to_downloader(tmp_path: Path) -> None:
    observed_auto_adjust: list[bool] = []

    def fake_download(symbol: str, start: datetime, end: datetime, interval: str, auto_adjust: bool) -> FakeFrame:
        observed_auto_adjust.append(auto_adjust)
        return FakeFrame(fake_rows())

    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        adjustment_policy="adjusted",
        downloader=fake_download,
        currency_lookup=lambda symbol: "SEK",
    )

    provider.download_to_csv(symbol="VOLV-B.ST", start=datetime(2024, 1, 1), end=datetime(2024, 1, 3))

    assert observed_auto_adjust == [True]


def test_empty_download_response_is_rejected(tmp_path: Path) -> None:
    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        downloader=lambda symbol, start, end, interval, auto_adjust: FakeFrame([]),
        currency_lookup=lambda symbol: "SEK",
    )

    with pytest.raises(ValueError, match="No historical data"):
        provider.download_to_csv(symbol="INVALID", start=datetime(2024, 1, 1), end=datetime(2024, 1, 2))


def test_missing_ohlcv_value_is_rejected(tmp_path: Path) -> None:
    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        downloader=lambda symbol, start, end, interval, auto_adjust: FakeFrame(
            [(datetime(2024, 1, 1), {"Open": "100", "High": "101", "Low": "99", "Close": None, "Volume": "1000"})]
        ),
        currency_lookup=lambda symbol: "SEK",
    )

    with pytest.raises(ValueError, match="Malformed yfinance data"):
        provider.download_to_csv(symbol="ABC", start=datetime(2024, 1, 1), end=datetime(2024, 1, 2))


def test_duplicate_download_timestamps_are_rejected(tmp_path: Path) -> None:
    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        downloader=lambda symbol, start, end, interval, auto_adjust: FakeFrame(
            [
                (datetime(2024, 1, 1), {"Open": "100", "High": "101", "Low": "99", "Close": "100", "Volume": "1000"}),
                (datetime(2024, 1, 1), {"Open": "101", "High": "102", "Low": "100", "Close": "101", "Volume": "1000"}),
            ]
        ),
        currency_lookup=lambda symbol: "SEK",
    )

    with pytest.raises(ValueError, match="Duplicate timestamp"):
        provider.download_to_csv(symbol="ABC", start=datetime(2024, 1, 1), end=datetime(2024, 1, 2))


def test_network_download_error_is_reported(tmp_path: Path) -> None:
    def failing_download(symbol: str, start: datetime, end: datetime, interval: str, auto_adjust: bool) -> FakeFrame:
        raise OSError("network unavailable")

    provider = YahooFinanceDataProvider(
        cache_dir=tmp_path,
        downloader=failing_download,
        currency_lookup=lambda symbol: "SEK",
    )

    with pytest.raises(RuntimeError, match="Failed to download"):
        provider.download_to_csv(symbol="ABC", start=datetime(2024, 1, 1), end=datetime(2024, 1, 2))


def test_currency_mismatch_protection(tmp_path: Path) -> None:
    csv_path = tmp_path / "abc_daily.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n2024-01-01T00:00:00,100,101,99,100,1000\n",
        encoding="utf-8",
    )
    save_dataset_metadata(
        csv_path,
        DatasetMetadata(
            symbol="ABC",
            quote_currency="USD",
            source="yfinance",
            interval="1d",
            start_date="2024-01-01",
            end_date="2024-01-02",
        ),
    )

    with pytest.raises(ValueError, match="FX conversion is not implemented"):
        require_matching_currency(csv_path, "SEK")


def test_missing_metadata_does_not_block_existing_csv_research_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_ohlcv.csv"

    assert require_matching_currency(fixture, "SEK") is None


def test_existing_csv_research_command_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_ohlcv.csv"

    main(["research", str(fixture)])

    output = capsys.readouterr().out
    assert "TradingBot EMA 20/50 historical research" in output
    assert "Full History" in output
