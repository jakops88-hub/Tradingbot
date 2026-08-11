import inspect
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot import app
from trading_bot.ai.freshness import current_scan_download_end, latest_expected_completed_daily_candle
from trading_bot.ai.scanner import HybridMarketScanner
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle
from trading_bot.forward.evaluator import evaluate_forward_decisions
from trading_bot.persistence.sqlite_store import TradingBotSQLiteStore


def candles(symbol: str, start: datetime, count: int, close: Decimal = Decimal("100")) -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=close,
            high=close + Decimal("1"),
            low=close - Decimal("1"),
            close=close,
            volume=Decimal("1000") + Decimal(index),
        )
        for index in range(count)
    ]


def write_csv(path: Path, rows: list[Candle]) -> None:
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "\n".join(
            f"{row.timestamp.isoformat()},{row.open},{row.high},{row.low},{row.close},{row.volume}" for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_latest_completed_daily_candle_logic() -> None:
    assert latest_expected_completed_daily_candle(datetime(2026, 8, 11, 17, 0)).isoformat() == "2026-08-10"
    assert latest_expected_completed_daily_candle(datetime(2026, 8, 11, 19, 0)).isoformat() == "2026-08-11"
    assert latest_expected_completed_daily_candle(datetime(2026, 8, 9, 12, 0)).isoformat() == "2026-08-07"
    assert current_scan_download_end(datetime(2026, 8, 11, 19, 0)).isoformat() == "2026-08-12T00:00:00"


def test_current_loader_refreshes_into_separate_current_cache(monkeypatch, tmp_path: Path) -> None:
    symbols_path = tmp_path / "symbols.txt"
    symbols_path.write_text("ABC.ST\n", encoding="utf-8")
    calls = []

    class FakeProvider:
        def __init__(self, cache_dir, *, adjustment_policy):
            self.cache_dir = Path(cache_dir)
            self.adjustment_policy = adjustment_policy

        def download_to_csv(self, *, symbol, start, end):
            calls.append((self.cache_dir, symbol, start, end, self.adjustment_policy))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.cache_dir / f"{symbol}_daily.csv"
            write_csv(csv_path, candles(symbol, datetime(2026, 5, 24), 80))
            return csv_path

    monkeypatch.setattr(app, "YahooFinanceDataProvider", FakeProvider)
    monkeypatch.setattr(app, "require_matching_currency", lambda path, currency: None)

    result = app._load_current_symbol_datasets(
        symbols_path=str(symbols_path),
        output_dir=str(tmp_path / "data"),
        start=datetime(2018, 1, 1),
        scan_timestamp=datetime(2026, 8, 11, 19, 0),
    )

    assert calls[0][0] == tmp_path / "data" / "current"
    assert calls[0][2] == datetime(2018, 1, 1)
    assert calls[0][3] == datetime(2026, 8, 12)
    assert "ABC.ST" in result.datasets
    assert result.cache_dir == tmp_path / "data" / "current"


def test_stale_candles_block_openai_before_analysis() -> None:
    class RaisingAnalyst:
        def analyze(self, request):
            raise AssertionError("OpenAI must not be called for stale data")

    report = HybridMarketScanner(
        analyst=RaisingAnalyst(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
    ).scan(
        {"ABC.ST": candles("ABC.ST", datetime(2025, 10, 12), 80)},
        scan_timestamp=datetime(2026, 8, 11, 19, 0),
    )

    assert report.ranked_candidates == []
    assert report.analyzed_candidates == []
    assert len(report.stale_candidates) == 1
    assert "MARKET DATA STALE - candidate rejected before AI" in report.stale_candidates[0].reason


def test_stale_scan_report_creates_no_forward_decisions(tmp_path: Path) -> None:
    class RaisingAnalyst:
        def analyze(self, request):
            raise AssertionError("OpenAI must not be called for stale data")

    report = HybridMarketScanner(
        analyst=RaisingAnalyst(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
    ).scan(
        {"ABC.ST": candles("ABC.ST", datetime(2025, 10, 12), 80)},
        scan_timestamp=datetime(2026, 8, 11, 19, 0),
    )
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")

    assert store.save_scan(report, scan_timestamp=report.scan_timestamp, risk_profile="MEDIUM") == 0
    assert store.all_decisions() == []


def test_invalid_stale_decisions_are_excluded_from_forward_statistics(tmp_path: Path) -> None:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    connection = store._connect()
    with connection:
        connection.execute(
            """
            INSERT INTO ai_scan_decisions (
                scan_timestamp, decision_timestamp, symbol, xgboost_rank, xgboost_probability_pct,
                normalized_snapshot_json, openai_model, openai_decision, openai_confidence_pct,
                sentiment, market_regime, positive_factors_json, negative_factors_json, risk_flags_json,
                sources_json, proposed_stop_pct, proposed_stop_price, risk_profile, portfolio_currency,
                investor_country, portfolio_exposure_pct, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-08-11T19:00:00",
                "2025-12-30T00:00:00",
                "ABC.ST",
                1,
                "55",
                "{}",
                "test-model",
                "APPROVE",
                "70",
                "POSITIVE",
                "BULLISH",
                "[]",
                "[]",
                "[]",
                "[]",
                "5",
                "95",
                "MEDIUM",
                "SEK",
                "Sweden",
                "0",
                "2026-08-11T19:00:01",
            ),
        )

    assert store.invalidate_stale_decisions() == 1
    report = evaluate_forward_decisions(store=store, datasets={"ABC.ST": candles("ABC.ST", datetime(2026, 8, 1), 20)})

    assert report.pending_decisions == 0
    assert report.completed_trades == 0
    assert report.invalid_stale_decisions == 1
    assert all(group.completed_trades == 0 for group in report.groups)


def test_historical_research_commands_do_not_use_current_market_loader() -> None:
    assert "_load_current_symbol_datasets" not in inspect.getsource(app.run_research_command)
    assert "_load_current_symbol_datasets" not in inspect.getsource(app.run_ml_research_command)
    assert "_load_current_symbol_datasets" not in inspect.getsource(app.run_xgb_research_command)
