import inspect
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

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
from trading_bot.dashboard.repository import DashboardRepository
from trading_bot.dashboard.web import DashboardApplication
from trading_bot.persistence.sqlite_store import ForwardOutcomeRecord, TradingBotSQLiteStore


START = datetime(2026, 8, 10)


def snapshot(symbol: str, *, probability: str, rank_close: str = "100") -> CandidateSnapshot:
    return CandidateSnapshot(
        symbol=symbol,
        decision_timestamp=START,
        close_price=Decimal(rank_close),
        data_age_trading_days=0,
        xgboost_probability_pct=Decimal(probability),
        return_1d_pct=Decimal("1.1"),
        return_5d_pct=Decimal("2.2"),
        return_20d_pct=Decimal("3.3"),
        ema20_vs_ema50_pct=Decimal("4.4"),
        close_vs_ema20_pct=Decimal("5.5"),
        rsi14=Decimal("55"),
        atr14_over_close_pct=Decimal("1.2"),
        volatility_20d_pct=Decimal("2.3"),
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
        source_urls=[OpenAISource(title="Company news", url="https://example.com/news")],
        model_request_succeeded=True,
        web_search_failed=False,
    )


def analysis(symbol: str, decision: AnalystDecision = AnalystDecision.WATCH) -> OpenAIAnalysisResult:
    return OpenAIAnalysisResult(
        symbol=symbol,
        decision=decision,
        confidence=Decimal("0.74"),
        sentiment=AnalystSentiment.POSITIVE,
        regime=AnalystRegime.BULLISH,
        positive_factors=["trend"],
        negative_factors=["valuation"],
        risk_flags=["earnings"],
        summary=f"{symbol} advisory summary",
        diagnostics=diagnostics(),
    )


def report_for(
    ranked: list[tuple[str, str]],
    analyzed: list[tuple[str, AnalystDecision]],
    *,
    scan_timestamp: datetime = START,
) -> HybridScanReport:
    candidates = [ScanCandidate(snapshot=snapshot(symbol, probability=probability)) for symbol, probability in ranked]
    by_symbol = {candidate.symbol: candidate for candidate in candidates}
    return HybridScanReport(
        scan_timestamp=scan_timestamp,
        ranked_candidates=candidates,
        analyzed_candidates=[
            AnalyzedCandidate(candidate=by_symbol[symbol], openai_analysis=analysis(symbol, decision))
            for symbol, decision in analyzed
        ],
        stale_candidates=[],
        data_issues=[],
        max_openai_analyses=3,
    )


def store_with_scan(tmp_path: Path) -> TradingBotSQLiteStore:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    store.save_scan(
        report_for(
            [("AAA.ST", "61"), ("BBB.ST", "55"), ("CCC.ST", "49")],
            [("AAA.ST", AnalystDecision.APPROVE), ("BBB.ST", AnalystDecision.WATCH)],
        ),
        scan_timestamp=START,
        risk_profile="MEDIUM; no leverage",
    )
    return store


def test_dashboard_loads_zero_data_state(tmp_path: Path) -> None:
    application = DashboardApplication(database_path=tmp_path / "empty.sqlite3", current_cache_dir=tmp_path / "current")

    status, content_type, body = application.handle_path("/")

    assert status == 200
    assert content_type.startswith("text/html")
    assert "TradingBot översikt" in body.decode()
    assert "Livehandel: LOCKED" in body.decode()


def test_dashboard_renders_database_data_and_latest_scan(tmp_path: Path) -> None:
    store = store_with_scan(tmp_path)
    store.save_scan(
        report_for(
            [("NEW.ST", "67"), ("OLD.ST", "40")],
            [("NEW.ST", AnalystDecision.REJECT)],
            scan_timestamp=START + timedelta(days=1),
        ),
        scan_timestamp=START + timedelta(days=1),
        risk_profile="MEDIUM; no leverage",
    )
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path("/scanner")

    text = body.decode()
    assert status == 200
    assert "NEW.ST" in text
    assert "OLD.ST" in text
    assert "AAA.ST" not in text
    assert "REJECT" in text


def test_overview_uses_simple_human_readable_summary(tmp_path: Path) -> None:
    store_with_scan(tmp_path)
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path("/")

    text = body.decode()
    assert status == 200
    assert "Dagens AI-beslut" in text
    assert "Portföljvärde" in text
    assert "Dagens P&amp;L" in text
    assert "30 dagar P&amp;L" in text
    assert "All-time P&amp;L" in text
    assert "Equitykurva" in text
    assert "Öppna positioner" in text
    assert "Senaste aktivitet" in text
    assert "Risk/trade: 1%" in text
    assert "MEDIUM; no leverage" not in text
    assert "2026-08-10 00:00" in text
    assert "AAA.ST advisory summary" in text
    assert "No AI summary stored" not in text


def test_overview_uses_company_names_for_known_symbols(tmp_path: Path) -> None:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    store.save_scan(
        report_for(
            [("SAND.ST", "61"), ("ERIC-B.ST", "55"), ("SHB-A.ST", "49")],
            [("SAND.ST", AnalystDecision.APPROVE), ("ERIC-B.ST", AnalystDecision.WATCH), ("SHB-A.ST", AnalystDecision.REJECT)],
        ),
        scan_timestamp=START,
        risk_profile="MEDIUM; no leverage",
    )
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path("/")

    text = body.decode()
    assert status == 200
    assert "Sandvik" in text
    assert "Ericsson B" in text
    assert "Handelsbanken A" in text


def test_overview_renders_stored_ibkr_broker_status(tmp_path: Path) -> None:
    store_with_scan(tmp_path)
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO ibkr_broker_snapshots (
                captured_at, connected, environment, account_id_masked, base_currency,
                cash_balance, net_liquidation_value, buying_power, positions_json,
                open_orders_json, recent_executions_json, reconciliation_json, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-08-11T12:00:00",
                1,
                "PAPER",
                "DU***67",
                "SEK",
                "1000",
                "1100",
                "900",
                "[]",
                "[]",
                "[]",
                '{"mismatches": []}',
                None,
                "2026-08-11T12:00:00",
            ),
        )
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path("/")

    text = body.decode()
    assert status == 200
    assert "IBKR PAPER" in text
    assert "DU***67" in text
    assert "READ ONLY" in text


def test_candidate_detail_renders_snapshot_summary_and_sources(tmp_path: Path) -> None:
    store_with_scan(tmp_path)
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path(f"/candidate?scan={START.isoformat()}&symbol=AAA.ST")

    text = body.decode()
    assert status == 200
    assert "AAA.ST advisory summary" in text
    assert "volume_vs_20d_pct" in text
    assert "Company news" in text
    assert "APPROVE" in text


def test_forward_statistics_render_completed_trade(tmp_path: Path) -> None:
    store = store_with_scan(tmp_path)
    decision_id = int(store.all_decisions()[0]["id"])
    store.complete_forward_outcome(
        ForwardOutcomeRecord(
            decision_id=decision_id,
            entry_time=START + timedelta(days=1),
            entry_price=Decimal("100"),
            exit_time=START + timedelta(days=11),
            exit_price=Decimal("106"),
            exit_reason="max_hold",
            net_pnl_sek=Decimal("6"),
            net_return_pct=Decimal("6"),
            outcome="WIN",
            holding_period_bars=10,
        )
    )
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path("/forward")

    text = body.decode()
    assert status == 200
    assert "ALL XGBoost top-3" in text
    assert "6.00 SEK" in text
    assert "100.00%" in text


def test_history_filters_by_symbol_decision_status_and_date(tmp_path: Path) -> None:
    store_with_scan(tmp_path)
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, _, body = application.handle_path("/history?symbol=AAA.ST&decision=APPROVE&status=PENDING&date=2026-08-10")

    text = body.decode()
    assert status == 200
    assert "AAA.ST" in text
    assert "BBB.ST" not in text


def test_invalid_stale_decisions_are_excluded_from_dashboard_stats(tmp_path: Path) -> None:
    store = store_with_scan(tmp_path)
    with store._connect() as connection:
        connection.execute(
            "UPDATE ai_scan_decisions SET forward_status = 'INVALID_STALE_DATA' WHERE symbol = 'AAA.ST'"
        )
    repository = DashboardRepository(tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    history = repository.history()
    stats = repository.forward_stats()
    system = repository.system_status()

    assert [row["symbol"] for row in history] == ["BBB.ST"]
    assert stats[0].pending_decisions == 1
    assert system.invalid_stale_records == 1


def test_dashboard_json_api_returns_latest_rows(tmp_path: Path) -> None:
    store_with_scan(tmp_path)
    application = DashboardApplication(database_path=tmp_path / "tradingbot.sqlite3", current_cache_dir=tmp_path / "current")

    status, content_type, body = application.handle_path("/api/scanner/latest")

    assert status == 200
    assert content_type.startswith("application/json")
    assert '"symbol": "AAA.ST"' in body.decode()
    assert '"openai_decision": "APPROVE"' in body.decode()


def test_dashboard_layers_do_not_call_openai_or_broker_or_risk_manager() -> None:
    import trading_bot.dashboard.repository as repository_module
    import trading_bot.dashboard.web as web_module

    source = inspect.getsource(repository_module) + inspect.getsource(web_module)

    assert "OpenAIAnalyst" not in source
    assert "responses.create" not in source
    assert "submit_order" not in source
    assert "PaperBroker" not in source
    assert "RiskManager" not in source
    assert "get_risk_profile" not in source
