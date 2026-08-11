import inspect
import json
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal

import trading_bot.forward.evaluator as forward_module
from trading_bot.ai.candidate_snapshot import CandidateSnapshot
from trading_bot.ai.openai_analyst import (
    AnalystDecision,
    AnalystRegime,
    AnalystSentiment,
    OpenAIDiagnostics,
    OpenAIAnalysisResult,
    OpenAISource,
)
from trading_bot.ai.scanner import AnalyzedCandidate, HybridScanReport, ScanCandidate
from trading_bot.data.models import Candle
from trading_bot.forward.evaluator import evaluate_forward_decisions
from trading_bot.persistence.sqlite_store import DuplicateDecisionError, TradingBotSQLiteStore


START = datetime(2026, 1, 1)


def snapshot(symbol: str, decision_timestamp: datetime = START, close: Decimal = Decimal("100")) -> CandidateSnapshot:
    return CandidateSnapshot(
        symbol=symbol,
        decision_timestamp=decision_timestamp,
        close_price=close,
        data_age_trading_days=0,
        xgboost_probability_pct=Decimal("55"),
        return_1d_pct=Decimal("1"),
        return_5d_pct=Decimal("2"),
        return_20d_pct=Decimal("3"),
        ema20_vs_ema50_pct=Decimal("1"),
        close_vs_ema20_pct=Decimal("1"),
        rsi14=Decimal("55"),
        atr14_over_close_pct=Decimal("1"),
        volatility_20d_pct=Decimal("2"),
        volume_vs_20d_pct=Decimal("8.92"),
        portfolio_exposure_pct=Decimal("0"),
    )


def diagnostics() -> OpenAIDiagnostics:
    return OpenAIDiagnostics(
        model_used="gpt-5.6-terra",
        http_status_code=None,
        openai_error_type=None,
        openai_error_code=None,
        failure_phase=None,
        structured_output_used=True,
        web_search_used=True,
        web_search_count=1,
        source_urls=[OpenAISource(title="Source", url="https://example.com")],
        model_request_succeeded=True,
        web_search_failed=False,
    )


def analysis(symbol: str, decision: AnalystDecision = AnalystDecision.WATCH) -> OpenAIAnalysisResult:
    return OpenAIAnalysisResult(
        symbol=symbol,
        decision=decision,
        confidence=Decimal("0.7"),
        sentiment=AnalystSentiment.POSITIVE,
        regime=AnalystRegime.BULLISH,
        positive_factors=["trend"],
        negative_factors=["valuation"],
        risk_flags=["risk"],
        summary="mocked",
        diagnostics=diagnostics(),
    )


def report_for(*rows: tuple[str, AnalystDecision]) -> HybridScanReport:
    analyzed = [
        AnalyzedCandidate(
            candidate=ScanCandidate(snapshot=snapshot(symbol)),
            openai_analysis=analysis(symbol, decision),
        )
        for symbol, decision in rows
    ]
    return HybridScanReport(
        scan_timestamp=START,
        ranked_candidates=[row.candidate for row in analyzed],
        analyzed_candidates=analyzed,
        stale_candidates=[],
        data_issues=[],
        max_openai_analyses=3,
    )


def candles(symbol: str, opens: list[str], *, low_override: dict[int, str] | None = None) -> list[Candle]:
    low_override = low_override or {}
    rows: list[Candle] = []
    for index, open_text in enumerate(opens):
        open_price = Decimal(open_text)
        low = Decimal(low_override[index]) if index in low_override else open_price - Decimal("1")
        rows.append(
            Candle(
                symbol=symbol,
                timestamp=START + timedelta(days=index),
                open=open_price,
                high=open_price + Decimal("2"),
                low=low,
                close=open_price,
                volume=Decimal("1000"),
            )
        )
    return rows


def row_count(db_path) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM ai_scan_decisions").fetchone()[0])


def first_row(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute("SELECT * FROM ai_scan_decisions ORDER BY id LIMIT 1").fetchone()


def test_persistence_stores_complete_ai_scan_decision(tmp_path) -> None:
    db_path = tmp_path / "tradingbot.sqlite3"
    store = TradingBotSQLiteStore(db_path)

    saved = store.save_scan(
        report_for(("ABC", AnalystDecision.APPROVE)),
        scan_timestamp=START,
        risk_profile="MEDIUM; no leverage",
    )
    row = first_row(db_path)

    assert saved == 1
    assert row_count(db_path) == 1
    assert row["symbol"] == "ABC"
    assert row["openai_decision"] == "APPROVE"
    assert row["portfolio_currency"] == "SEK"
    assert row["investor_country"] == "Sweden"
    assert Decimal(row["proposed_stop_pct"]) == Decimal("5")
    assert Decimal(row["proposed_stop_price"]) == Decimal("95.00")
    assert json.loads(row["sources_json"])[0]["url"] == "https://example.com"
    assert json.loads(row["normalized_snapshot_json"])["values"]["volume_vs_20d_pct"] == "8.92"


def test_duplicate_scan_symbol_timestamp_is_rejected(tmp_path) -> None:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    scan = report_for(("ABC", AnalystDecision.WATCH))

    store.save_scan(scan, scan_timestamp=START, risk_profile="MEDIUM")

    try:
        store.save_scan(scan, scan_timestamp=START, risk_profile="MEDIUM")
    except DuplicateDecisionError:
        pass
    else:
        raise AssertionError("duplicate decision was not rejected")


def test_forward_completion_does_not_mutate_original_decision_snapshot(tmp_path) -> None:
    db_path = tmp_path / "tradingbot.sqlite3"
    store = TradingBotSQLiteStore(db_path)
    store.save_scan(report_for(("ABC", AnalystDecision.WATCH)), scan_timestamp=START, risk_profile="MEDIUM")
    before = first_row(db_path)["normalized_snapshot_json"]

    evaluate_forward_decisions(
        store=store,
        datasets={"ABC": candles("ABC", ["100", "100"] + ["100"] * 9 + ["110"])},
    )
    after = first_row(db_path)

    assert after["normalized_snapshot_json"] == before
    assert after["forward_status"] == "COMPLETED"


def test_forward_waits_until_future_candles_exist(tmp_path) -> None:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    store.save_scan(report_for(("ABC", AnalystDecision.WATCH)), scan_timestamp=START, risk_profile="MEDIUM")

    report = evaluate_forward_decisions(store=store, datasets={"ABC": candles("ABC", ["100"] * 5)})

    assert report.pending_decisions == 1
    assert report.completed_trades == 0


def test_forward_uses_next_open_and_10_day_exit_with_costs(tmp_path) -> None:
    db_path = tmp_path / "tradingbot.sqlite3"
    store = TradingBotSQLiteStore(db_path)
    store.save_scan(report_for(("ABC", AnalystDecision.APPROVE)), scan_timestamp=START, risk_profile="MEDIUM")

    evaluate_forward_decisions(
        store=store,
        datasets={"ABC": candles("ABC", ["100", "100"] + ["100"] * 9 + ["110"])},
    )
    row = first_row(db_path)

    assert row["entry_time"] == (START + timedelta(days=1)).isoformat()
    assert Decimal(row["entry_price"]) == Decimal("100.100")
    assert row["exit_time"] == (START + timedelta(days=11)).isoformat()
    assert Decimal(row["exit_price"]) == Decimal("109.890")
    assert row["exit_reason"] == "max_hold"
    assert Decimal(row["net_pnl_sek"]) == Decimal("9.580010")
    assert row["outcome"] == "WIN"
    assert int(row["holding_period_bars"]) == 10


def test_forward_gap_stop_is_conservative_and_records_loss(tmp_path) -> None:
    db_path = tmp_path / "tradingbot.sqlite3"
    store = TradingBotSQLiteStore(db_path)
    store.save_scan(report_for(("ABC", AnalystDecision.REJECT)), scan_timestamp=START, risk_profile="MEDIUM")

    evaluate_forward_decisions(
        store=store,
        datasets={"ABC": candles("ABC", ["100", "100", "90"] + ["90"] * 9, low_override={2: "89"})},
    )
    row = first_row(db_path)

    assert row["exit_reason"] == "stop_loss"
    assert Decimal(row["exit_price"]) == Decimal("89.910")
    assert row["outcome"] == "LOSS"
    assert int(row["holding_period_bars"]) == 1


def test_approve_watch_reject_comparison_statistics(tmp_path) -> None:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    store.save_scan(
        report_for(
            ("AAA", AnalystDecision.APPROVE),
            ("BBB", AnalystDecision.WATCH),
            ("CCC", AnalystDecision.REJECT),
        ),
        scan_timestamp=START,
        risk_profile="MEDIUM",
    )
    datasets = {
        "AAA": candles("AAA", ["100", "100"] + ["100"] * 9 + ["110"]),
        "BBB": candles("BBB", ["100", "100"] + ["100"] * 9 + ["99"]),
        "CCC": candles("CCC", ["100", "100", "90"] + ["90"] * 9, low_override={2: "89"}),
    }

    report = evaluate_forward_decisions(store=store, datasets=datasets)
    groups = {group.label: group for group in report.groups}

    assert report.completed_trades == 3
    assert groups["ALL XGBoost top-3"].completed_trades == 3
    assert groups["OpenAI APPROVE"].wins == 1
    assert groups["OpenAI WATCH"].losses == 1
    assert groups["OpenAI REJECT"].stop_exits == 1


def test_forward_evaluator_has_no_live_broker_or_order_path() -> None:
    source = inspect.getsource(forward_module)

    assert "LiveBroker" not in source
    assert "broker_api" not in source
    assert "submit_order" not in source
