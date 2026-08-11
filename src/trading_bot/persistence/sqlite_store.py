"""SQLite persistence for AI scan decisions and forward outcomes."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_bot.ai.freshness import (
    DEFAULT_STALENESS_TOLERANCE_TRADING_DAYS,
    latest_expected_completed_daily_candle,
    trading_day_gap,
)
from trading_bot.ai.openai_analyst import OpenAIAnalysisResult, OpenAISource
from trading_bot.ai.scanner import HybridScanReport


DEFAULT_DATABASE_PATH = Path("data/tradingbot.sqlite3")
PROPOSED_STOP_PCT = Decimal("5")


class DuplicateDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class StoredDecision:
    id: int
    scan_timestamp: datetime
    decision_timestamp: datetime
    symbol: str
    forward_status: str


@dataclass(frozen=True)
class ForwardOutcomeRecord:
    decision_id: int
    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    exit_reason: str
    net_pnl_sek: Decimal
    net_return_pct: Decimal
    outcome: str
    holding_period_bars: int


class TradingBotSQLiteStore:
    def __init__(self, path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_scan_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_timestamp TEXT NOT NULL,
                    decision_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    xgboost_rank INTEGER NOT NULL,
                    xgboost_probability_pct TEXT NOT NULL,
                    normalized_snapshot_json TEXT NOT NULL,
                    openai_model TEXT NOT NULL,
                    openai_decision TEXT NOT NULL,
                    openai_confidence_pct TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    positive_factors_json TEXT NOT NULL,
                    negative_factors_json TEXT NOT NULL,
                    risk_flags_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    openai_summary TEXT NOT NULL DEFAULT '',
                    proposed_stop_pct TEXT NOT NULL,
                    proposed_stop_price TEXT NOT NULL,
                    risk_profile TEXT NOT NULL,
                    portfolio_currency TEXT NOT NULL,
                    investor_country TEXT NOT NULL,
                    portfolio_exposure_pct TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    forward_status TEXT NOT NULL DEFAULT 'PENDING',
                    entry_time TEXT,
                    entry_price TEXT,
                    exit_time TEXT,
                    exit_price TEXT,
                    exit_reason TEXT,
                    net_pnl_sek TEXT,
                    net_return_pct TEXT,
                    outcome TEXT,
                    holding_period_bars INTEGER,
                    completed_at TEXT,
                    invalidated_at TEXT,
                    invalid_reason TEXT,
                    UNIQUE(scan_timestamp, symbol)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_scan_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_timestamp TEXT NOT NULL,
                    decision_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    xgboost_rank INTEGER NOT NULL,
                    xgboost_probability_pct TEXT NOT NULL,
                    normalized_snapshot_json TEXT NOT NULL,
                    latest_close TEXT NOT NULL,
                    data_age_trading_days INTEGER NOT NULL,
                    portfolio_exposure_pct TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(scan_timestamp, symbol)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ibkr_broker_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    connected INTEGER NOT NULL,
                    environment TEXT NOT NULL,
                    account_id_masked TEXT NOT NULL,
                    account_id_hash TEXT,
                    base_currency TEXT,
                    cash_balance TEXT,
                    net_liquidation_value TEXT,
                    buying_power TEXT,
                    positions_json TEXT NOT NULL,
                    open_orders_json TEXT NOT NULL,
                    recent_executions_json TEXT NOT NULL,
                    reconciliation_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ibkr_contract_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    tradingbot_symbol TEXT NOT NULL,
                    intended_local_symbol TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    con_id INTEGER,
                    local_symbol TEXT,
                    exchange TEXT,
                    primary_exchange TEXT,
                    currency TEXT,
                    security_type TEXT,
                    long_name TEXT,
                    error TEXT,
                    UNIQUE(captured_at, tradingbot_symbol)
                )
                """
            )
            self._ensure_column(connection, "ai_scan_decisions", "invalidated_at", "TEXT")
            self._ensure_column(connection, "ai_scan_decisions", "invalid_reason", "TEXT")
            self._ensure_column(connection, "ai_scan_decisions", "openai_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "ibkr_broker_snapshots", "account_id_hash", "TEXT")

    def save_scan(
        self,
        report: HybridScanReport,
        *,
        scan_timestamp: datetime,
        risk_profile: str,
        portfolio_currency: str = "SEK",
        investor_country: str = "Sweden",
    ) -> int:
        saved = 0
        with self._connect() as connection:
            created_at = _dt(datetime.now(timezone.utc).replace(tzinfo=None))
            for rank, candidate in enumerate(report.ranked_candidates, start=1):
                snapshot = candidate.snapshot
                try:
                    connection.execute(
                        """
                        INSERT INTO ai_scan_rankings (
                            scan_timestamp,
                            decision_timestamp,
                            symbol,
                            xgboost_rank,
                            xgboost_probability_pct,
                            normalized_snapshot_json,
                            latest_close,
                            data_age_trading_days,
                            portfolio_exposure_pct,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _dt(scan_timestamp),
                            _dt(snapshot.decision_timestamp),
                            snapshot.symbol,
                            rank,
                            str(snapshot.xgboost_probability_pct),
                            _json(snapshot.as_prompt_payload()),
                            str(snapshot.close_price),
                            int(snapshot.data_age_trading_days),
                            str(snapshot.portfolio_exposure_pct),
                            created_at,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateDecisionError(
                        f"Duplicate AI scan ranking for {snapshot.symbol} at {_dt(scan_timestamp)}"
                    ) from exc
            for rank, analyzed in enumerate(report.analyzed_candidates, start=1):
                analysis = analyzed.openai_analysis
                if analysis is None:
                    continue
                snapshot = analyzed.candidate.snapshot
                try:
                    connection.execute(
                        """
                        INSERT INTO ai_scan_decisions (
                            scan_timestamp,
                            decision_timestamp,
                            symbol,
                            xgboost_rank,
                            xgboost_probability_pct,
                            normalized_snapshot_json,
                            openai_model,
                            openai_decision,
                            openai_confidence_pct,
                            sentiment,
                            market_regime,
                            positive_factors_json,
                            negative_factors_json,
                            risk_flags_json,
                            sources_json,
                            openai_summary,
                            proposed_stop_pct,
                            proposed_stop_price,
                            risk_profile,
                            portfolio_currency,
                            investor_country,
                            portfolio_exposure_pct,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _dt(scan_timestamp),
                            _dt(snapshot.decision_timestamp),
                            snapshot.symbol,
                            rank,
                            str(snapshot.xgboost_probability_pct),
                            _json(snapshot.as_prompt_payload()),
                            analysis.diagnostics.model_used,
                            analysis.decision.value,
                            str(analysis.confidence * Decimal("100")),
                            analysis.sentiment.value,
                            analysis.regime.value,
                            _json(analysis.positive_factors),
                            _json(analysis.negative_factors),
                            _json(analysis.risk_flags),
                            _json(_sources(analysis)),
                            analysis.summary,
                            str(PROPOSED_STOP_PCT),
                            str(snapshot.close_price * Decimal("0.95")),
                            risk_profile,
                            portfolio_currency,
                            investor_country,
                            str(snapshot.portfolio_exposure_pct),
                            created_at,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateDecisionError(
                        f"Duplicate AI scan decision for {snapshot.symbol} at {_dt(scan_timestamp)}"
                    ) from exc
                saved += 1
        return saved

    def pending_decisions(self) -> list[StoredDecision]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, scan_timestamp, decision_timestamp, symbol, forward_status
                FROM ai_scan_decisions
                WHERE forward_status = 'PENDING'
                ORDER BY scan_timestamp, xgboost_rank
                """
            ).fetchall()
        return [_stored_decision(row) for row in rows]

    def invalidate_stale_decisions(
        self,
        *,
        tolerance_trading_days: int = DEFAULT_STALENESS_TOLERANCE_TRADING_DAYS,
    ) -> int:
        invalidated = 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, scan_timestamp, decision_timestamp, symbol
                FROM ai_scan_decisions
                WHERE forward_status = 'PENDING'
                """
            ).fetchall()
            for row in rows:
                scan_timestamp = datetime.fromisoformat(row["scan_timestamp"])
                decision_timestamp = datetime.fromisoformat(row["decision_timestamp"])
                expected_date = latest_expected_completed_daily_candle(scan_timestamp)
                age = trading_day_gap(decision_timestamp.date(), expected_date)
                if age <= tolerance_trading_days:
                    continue
                connection.execute(
                    """
                    UPDATE ai_scan_decisions
                    SET forward_status = 'INVALID_STALE_DATA',
                        invalidated_at = ?,
                        invalid_reason = ?
                    WHERE id = ? AND forward_status = 'PENDING'
                    """,
                    (
                        _dt(now),
                        (
                            "MARKET DATA STALE - candidate rejected before AI "
                            f"(symbol={row['symbol']}, decision candle {decision_timestamp.date().isoformat()}, "
                            f"scan expected {expected_date.isoformat()}, age {age} trading days, "
                            f"tolerance {tolerance_trading_days})"
                        ),
                        int(row["id"]),
                    ),
                )
                invalidated += 1
        return invalidated

    def completed_decisions(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT *
                    FROM ai_scan_decisions
                    WHERE forward_status = 'COMPLETED'
                    ORDER BY scan_timestamp, xgboost_rank
                    """
                ).fetchall()
            )

    def all_decisions(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM ai_scan_decisions ORDER BY scan_timestamp, xgboost_rank"
                ).fetchall()
            )

    def invalid_decisions(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT *
                    FROM ai_scan_decisions
                    WHERE forward_status = 'INVALID_STALE_DATA'
                    ORDER BY scan_timestamp, xgboost_rank
                    """
                ).fetchall()
            )

    def complete_forward_outcome(self, outcome: ForwardOutcomeRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ai_scan_decisions
                SET forward_status = 'COMPLETED',
                    entry_time = ?,
                    entry_price = ?,
                    exit_time = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    net_pnl_sek = ?,
                    net_return_pct = ?,
                    outcome = ?,
                    holding_period_bars = ?,
                    completed_at = ?
                WHERE id = ? AND forward_status = 'PENDING'
                """,
                (
                    _dt(outcome.entry_time),
                    str(outcome.entry_price),
                    _dt(outcome.exit_time),
                    str(outcome.exit_price),
                    outcome.exit_reason,
                    str(outcome.net_pnl_sek),
                    str(outcome.net_return_pct),
                    outcome.outcome,
                    outcome.holding_period_bars,
                    _dt(datetime.now(timezone.utc).replace(tzinfo=None)),
                    outcome.decision_id,
                ),
            )

    def save_ibkr_snapshot(self, snapshot: Any) -> int:
        from trading_bot.execution.ibkr_broker import account_id_hash, mask_account_id

        with self._connect() as connection:
            created_at = _dt(datetime.now(timezone.utc).replace(tzinfo=None))
            connection.execute(
                """
                INSERT INTO ibkr_broker_snapshots (
                    captured_at,
                    connected,
                    environment,
                    account_id_masked,
                    account_id_hash,
                    base_currency,
                    cash_balance,
                    net_liquidation_value,
                    buying_power,
                    positions_json,
                    open_orders_json,
                    recent_executions_json,
                    reconciliation_json,
                    error,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt(snapshot.connection.captured_at),
                    1 if snapshot.connection.connected else 0,
                    snapshot.connection.environment,
                    mask_account_id(snapshot.connection.account_id),
                    account_id_hash(snapshot.connection.account_id),
                    snapshot.connection.base_currency,
                    _optional_decimal_text(snapshot.connection.cash_balance),
                    _optional_decimal_text(snapshot.connection.net_liquidation_value),
                    _optional_decimal_text(snapshot.connection.buying_power),
                    _json([asdict(position) for position in snapshot.positions]),
                    _json([asdict(order) for order in snapshot.open_orders]),
                    _json([asdict(execution) for execution in snapshot.recent_executions]),
                    _json(asdict(snapshot.reconciliation)),
                    snapshot.connection.error,
                    created_at,
                ),
            )
            snapshot_id = int(connection.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            for contract in snapshot.contracts:
                connection.execute(
                    """
                    INSERT INTO ibkr_contract_details (
                        captured_at,
                        tradingbot_symbol,
                        intended_local_symbol,
                        verified,
                        con_id,
                        local_symbol,
                        exchange,
                        primary_exchange,
                        currency,
                        security_type,
                        long_name,
                        error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _dt(snapshot.connection.captured_at),
                        contract.tradingbot_symbol,
                        contract.intended_local_symbol,
                        1 if contract.verified else 0,
                        contract.con_id,
                        contract.local_symbol,
                        contract.exchange,
                        contract.primary_exchange,
                        contract.currency,
                        contract.security_type,
                        contract.long_name,
                        contract.error,
                    ),
                )
        return snapshot_id

    def latest_ibkr_snapshot(self) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT *
                FROM ibkr_broker_snapshots
                ORDER BY captured_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()

    def latest_ibkr_contracts(self) -> list[sqlite3.Row]:
        snapshot = self.latest_ibkr_snapshot()
        if snapshot is None:
            return []
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT *
                    FROM ibkr_contract_details
                    WHERE captured_at = ?
                    ORDER BY tradingbot_symbol
                    """,
                    (snapshot["captured_at"],),
                ).fetchall()
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _stored_decision(row: sqlite3.Row) -> StoredDecision:
    return StoredDecision(
        id=int(row["id"]),
        scan_timestamp=datetime.fromisoformat(row["scan_timestamp"]),
        decision_timestamp=datetime.fromisoformat(row["decision_timestamp"]),
        symbol=str(row["symbol"]),
        forward_status=str(row["forward_status"]),
    )


def _sources(analysis: OpenAIAnalysisResult) -> list[dict[str, str]]:
    return [asdict(source) for source in analysis.diagnostics.source_urls]


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _dt(value: datetime) -> str:
    return value.replace(tzinfo=None).isoformat()


def _optional_decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
