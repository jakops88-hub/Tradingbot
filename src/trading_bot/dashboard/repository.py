"""Read-only dashboard repository backed by TradingBot SQLite data."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

from trading_bot.forward.evaluator import ForwardComparisonStats
from trading_bot.persistence.sqlite_store import DEFAULT_DATABASE_PATH, TradingBotSQLiteStore


VALID_FORWARD_STATUSES = {"PENDING", "COMPLETED"}


@dataclass(frozen=True)
class DashboardScanRow:
    scan_timestamp: str
    decision_timestamp: str
    symbol: str
    xgboost_rank: int
    xgboost_probability_pct: Decimal
    latest_close: Decimal
    data_age_trading_days: int
    normalized_snapshot: dict[str, Any]
    decision_id: int | None
    openai_model: str | None
    openai_decision: str | None
    openai_confidence_pct: Decimal | None
    sentiment: str | None
    market_regime: str | None
    positive_factors: list[str]
    negative_factors: list[str]
    risk_flags: list[str]
    sources: list[dict[str, str]]
    summary: str | None
    forward_status: str | None


@dataclass(frozen=True)
class DashboardOverview:
    status: str
    mode: str
    current_capital_assumption: str
    risk_profile: str
    latest_market_data_candle: str | None
    latest_scan_timestamp: str | None
    openai_model: str
    openai_status: str
    current_portfolio_exposure_pct: Decimal


@dataclass(frozen=True)
class DashboardSystemStatus:
    database_path: Path
    database_exists: bool
    current_market_cache_path: Path
    current_market_cache_files: int
    latest_market_data_candle: str | None
    openai_model: str
    stored_scans: int
    pending_forward_tests: int
    invalid_stale_records: int
    live_trading_status: str


@dataclass(frozen=True)
class DashboardPerformance:
    portfolio_value: Decimal
    today_pnl_sek: Decimal
    today_pnl_pct: Decimal
    last_30_days_pnl_sek: Decimal
    last_30_days_pnl_pct: Decimal
    all_time_pnl_sek: Decimal
    all_time_pnl_pct: Decimal
    current_exposure_pct: Decimal
    realized_pnl_sek: Decimal
    unrealized_pnl_sek: Decimal


@dataclass(frozen=True)
class DashboardBrokerStatus:
    name: str
    connection_status: str
    environment: str
    account_id_masked: str
    account_value: Decimal | None
    cash: Decimal | None
    positions_count: int
    broker_sync_status: str
    trading_permissions: str
    captured_at: str | None


@dataclass(frozen=True)
class EquityCurvePoint:
    timestamp: str
    portfolio_value: Decimal
    cumulative_pnl: Decimal


@dataclass(frozen=True)
class RecentActivity:
    timestamp: str
    title: str
    detail: str
    status: str


class DashboardRepository:
    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        current_cache_dir: str | Path = "data/current",
    ) -> None:
        self.store = TradingBotSQLiteStore(database_path)
        self.database_path = Path(database_path)
        self.current_cache_dir = Path(current_cache_dir)

    def overview(self) -> DashboardOverview:
        latest_rows = self.latest_scan_rows()
        latest_decision = self._latest_decision_row()
        exposure_values = [
            row.normalized_snapshot.get("values", {}).get("portfolio_exposure_pct")
            for row in latest_rows
        ]
        exposures = [_decimal(value) for value in exposure_values if value is not None]
        return DashboardOverview(
            status="ONLINE",
            mode="RESEARCH / PAPER",
            current_capital_assumption="1000 SEK",
            risk_profile=latest_decision["risk_profile"] if latest_decision is not None else "MEDIUM",
            latest_market_data_candle=self.latest_market_data_candle(),
            latest_scan_timestamp=self.latest_scan_timestamp(),
            openai_model=latest_decision["openai_model"] if latest_decision is not None else "not available",
            openai_status="AVAILABLE FROM STORED SCANS" if latest_decision is not None else "NO STORED ANALYSES",
            current_portfolio_exposure_pct=max(exposures) if exposures else Decimal("0"),
        )

    def latest_scan_timestamp(self) -> str | None:
        with self.store._connect() as connection:
            ranking = connection.execute(
                "SELECT MAX(scan_timestamp) AS value FROM ai_scan_rankings"
            ).fetchone()
            if ranking is not None and ranking["value"]:
                return str(ranking["value"])
            decision = connection.execute(
                """
                SELECT MAX(scan_timestamp) AS value
                FROM ai_scan_decisions
                WHERE forward_status != 'INVALID_STALE_DATA'
                """
            ).fetchone()
        return str(decision["value"]) if decision is not None and decision["value"] else None

    def latest_scan_rows(self) -> list[DashboardScanRow]:
        scan_timestamp = self.latest_scan_timestamp()
        if scan_timestamp is None:
            return []
        with self.store._connect() as connection:
            ranking_count = connection.execute(
                "SELECT COUNT(*) AS count FROM ai_scan_rankings WHERE scan_timestamp = ?",
                (scan_timestamp,),
            ).fetchone()["count"]
            if int(ranking_count) > 0:
                rows = connection.execute(
                    """
                    SELECT r.*, d.id AS decision_id, d.openai_model, d.openai_decision,
                           d.openai_confidence_pct, d.sentiment, d.market_regime,
                           d.positive_factors_json, d.negative_factors_json,
                           d.risk_flags_json, d.sources_json, d.openai_summary, d.forward_status,
                           d.risk_profile
                    FROM ai_scan_rankings r
                    LEFT JOIN ai_scan_decisions d
                      ON d.scan_timestamp = r.scan_timestamp
                     AND d.symbol = r.symbol
                     AND d.forward_status != 'INVALID_STALE_DATA'
                    WHERE r.scan_timestamp = ?
                    ORDER BY r.xgboost_rank
                    """,
                    (scan_timestamp,),
                ).fetchall()
                return [_scan_row(row) for row in rows]
            rows = connection.execute(
                """
                SELECT d.*, d.id AS decision_id
                FROM ai_scan_decisions d
                WHERE d.scan_timestamp = ?
                  AND d.forward_status != 'INVALID_STALE_DATA'
                ORDER BY d.xgboost_rank
                """,
                (scan_timestamp,),
            ).fetchall()
        return [_scan_row(row) for row in rows]

    def candidate(self, *, scan_timestamp: str, symbol: str) -> DashboardScanRow | None:
        for row in self.latest_scan_rows() if scan_timestamp == self.latest_scan_timestamp() else self._scan_rows(scan_timestamp):
            if row.symbol == symbol:
                return row
        return None

    def history(
        self,
        *,
        symbol: str | None = None,
        decision: str | None = None,
        status: str | None = None,
        date_text: str | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["forward_status != 'INVALID_STALE_DATA'"]
        params: list[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if decision:
            clauses.append("openai_decision = ?")
            params.append(decision)
        if status:
            clauses.append("forward_status = ?")
            params.append(status)
        if date_text:
            clauses.append("date(scan_timestamp) = ?")
            params.append(date_text)
        with self.store._connect() as connection:
            return list(
                connection.execute(
                    f"""
                    SELECT *
                    FROM ai_scan_decisions
                    WHERE {' AND '.join(clauses)}
                    ORDER BY scan_timestamp DESC, xgboost_rank
                    """,
                    params,
                ).fetchall()
            )

    def forward_stats(self) -> list[ForwardComparisonStats]:
        rows = self.history()
        return [
            _stats("ALL XGBoost top-3", rows),
            _stats("OpenAI APPROVE", [row for row in rows if row["openai_decision"] == "APPROVE"]),
            _stats("OpenAI WATCH", [row for row in rows if row["openai_decision"] == "WATCH"]),
            _stats("OpenAI REJECT", [row for row in rows if row["openai_decision"] == "REJECT"]),
        ]

    def performance_summary(self, *, now: datetime | None = None) -> DashboardPerformance:
        now = now or datetime.now()
        rows = self.history()
        completed = [row for row in rows if row["forward_status"] == "COMPLETED"]
        today_pnl = _sum_pnl(row for row in completed if _row_date(row) == now.date())
        last_30_days_pnl = _sum_pnl(row for row in completed if _row_datetime(row) >= now - timedelta(days=30))
        all_time_pnl = _sum_pnl(completed)
        starting_capital = Decimal("1000")
        exposure = self.overview().current_portfolio_exposure_pct
        return DashboardPerformance(
            portfolio_value=starting_capital + all_time_pnl,
            today_pnl_sek=today_pnl,
            today_pnl_pct=_pnl_pct(today_pnl, starting_capital),
            last_30_days_pnl_sek=last_30_days_pnl,
            last_30_days_pnl_pct=_pnl_pct(last_30_days_pnl, starting_capital),
            all_time_pnl_sek=all_time_pnl,
            all_time_pnl_pct=_pnl_pct(all_time_pnl, starting_capital),
            current_exposure_pct=exposure,
            realized_pnl_sek=all_time_pnl,
            unrealized_pnl_sek=Decimal("0"),
        )

    def equity_curve(self) -> list[EquityCurvePoint]:
        completed = sorted(
            [row for row in self.history() if row["forward_status"] == "COMPLETED"],
            key=_row_datetime,
        )
        cumulative = Decimal("0")
        points = [EquityCurvePoint(timestamp="start", portfolio_value=Decimal("1000"), cumulative_pnl=Decimal("0"))]
        for row in completed:
            cumulative += _decimal(row["net_pnl_sek"])
            points.append(
                EquityCurvePoint(
                    timestamp=str(row["exit_time"] or row["completed_at"] or row["scan_timestamp"]),
                    portfolio_value=Decimal("1000") + cumulative,
                    cumulative_pnl=cumulative,
                )
            )
        return points

    def open_positions(self) -> list[sqlite3.Row]:
        return []

    def recent_activity(self, limit: int = 6) -> list[RecentActivity]:
        with self.store._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM ai_scan_decisions
                ORDER BY scan_timestamp DESC, xgboost_rank
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        activities: list[RecentActivity] = []
        for row in rows:
            status = str(row["forward_status"])
            if status == "COMPLETED":
                title = "Forwardtest klar"
                detail = f"{row['symbol']} stängdes med {row['net_pnl_sek']} SEK simulerat P&L"
            elif status == "INVALID_STALE_DATA":
                title = "Inaktuella data avvisade"
                detail = f"{row['symbol']} exkluderades från forwardtest"
            else:
                title = "AI-beslut sparat"
                detail = f"{row['symbol']} väntar på tillräcklig framtida data"
            activities.append(
                RecentActivity(
                    timestamp=str(row["scan_timestamp"]),
                    title=title,
                    detail=detail,
                    status=status,
                )
            )
        return activities

    def broker_status(self) -> DashboardBrokerStatus:
        snapshot = self.store.latest_ibkr_snapshot()
        if snapshot is None:
            return DashboardBrokerStatus(
                name="IBKR PAPER",
                connection_status="DISCONNECTED",
                environment="UNKNOWN",
                account_id_masked="n/a",
                account_value=None,
                cash=None,
                positions_count=0,
                broker_sync_status="No broker snapshot stored",
                trading_permissions="READ ONLY",
                captured_at=None,
            )
        reconciliation = _json_load(snapshot["reconciliation_json"], {})
        mismatches = reconciliation.get("mismatches", []) if isinstance(reconciliation, dict) else []
        return DashboardBrokerStatus(
            name="IBKR PAPER",
            connection_status="CONNECTED" if int(snapshot["connected"]) else "DISCONNECTED",
            environment=str(snapshot["environment"]),
            account_id_masked=str(snapshot["account_id_masked"]),
            account_value=_optional_decimal_value(snapshot["net_liquidation_value"]),
            cash=_optional_decimal_value(snapshot["cash_balance"]),
            positions_count=len(_json_load(snapshot["positions_json"], [])),
            broker_sync_status="Mismatch" if mismatches else "In sync",
            trading_permissions="READ ONLY",
            captured_at=str(snapshot["captured_at"]),
        )

    def system_status(self) -> DashboardSystemStatus:
        with self.store._connect() as connection:
            stored_scans = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT scan_timestamp) AS count FROM ai_scan_decisions"
                ).fetchone()["count"]
            )
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM ai_scan_decisions WHERE forward_status = 'PENDING'"
                ).fetchone()["count"]
            )
            invalid = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM ai_scan_decisions WHERE forward_status = 'INVALID_STALE_DATA'"
                ).fetchone()["count"]
            )
        return DashboardSystemStatus(
            database_path=self.database_path,
            database_exists=self.database_path.exists(),
            current_market_cache_path=self.current_cache_dir,
            current_market_cache_files=len(list(self.current_cache_dir.glob("*_daily.csv")))
            if self.current_cache_dir.exists()
            else 0,
            latest_market_data_candle=self.latest_market_data_candle(),
            openai_model=self.overview().openai_model,
            stored_scans=stored_scans,
            pending_forward_tests=pending,
            invalid_stale_records=invalid,
            live_trading_status="LOCKED / UNAVAILABLE",
        )

    def _latest_decision_row(self) -> sqlite3.Row | None:
        latest = self.latest_scan_timestamp()
        if latest is None:
            return None
        with self.store._connect() as connection:
            return connection.execute(
                """
                SELECT *
                FROM ai_scan_decisions
                WHERE scan_timestamp = ?
                  AND forward_status != 'INVALID_STALE_DATA'
                ORDER BY xgboost_rank
                LIMIT 1
                """,
                (latest,),
            ).fetchone()

    def latest_market_data_candle(self) -> str | None:
        latest: datetime | None = None
        if not self.current_cache_dir.exists():
            return None
        for csv_path in self.current_cache_dir.glob("*_daily.csv"):
            candidate = _latest_csv_timestamp(csv_path)
            if candidate is not None and (latest is None or candidate > latest):
                latest = candidate
        return latest.isoformat() if latest is not None else None

    def _scan_rows(self, scan_timestamp: str) -> list[DashboardScanRow]:
        with self.store._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, d.id AS decision_id, d.openai_model, d.openai_decision,
                       d.openai_confidence_pct, d.sentiment, d.market_regime,
                       d.positive_factors_json, d.negative_factors_json,
                       d.risk_flags_json, d.sources_json, d.openai_summary, d.forward_status,
                       d.risk_profile
                FROM ai_scan_rankings r
                LEFT JOIN ai_scan_decisions d
                  ON d.scan_timestamp = r.scan_timestamp
                 AND d.symbol = r.symbol
                 AND d.forward_status != 'INVALID_STALE_DATA'
                WHERE r.scan_timestamp = ?
                ORDER BY r.xgboost_rank
                """,
                (scan_timestamp,),
            ).fetchall()
        return [_scan_row(row) for row in rows]


def _scan_row(row: sqlite3.Row) -> DashboardScanRow:
    snapshot = _json_load(row["normalized_snapshot_json"], {})
    values = snapshot.get("values", {}) if isinstance(snapshot, dict) else {}
    close = row["latest_close"] if "latest_close" in row.keys() else values.get("close_price", "0")
    age = row["data_age_trading_days"] if "data_age_trading_days" in row.keys() else values.get("data_age_trading_days", 0)
    return DashboardScanRow(
        scan_timestamp=str(row["scan_timestamp"]),
        decision_timestamp=str(row["decision_timestamp"]),
        symbol=str(row["symbol"]),
        xgboost_rank=int(row["xgboost_rank"]),
        xgboost_probability_pct=_decimal(row["xgboost_probability_pct"]),
        latest_close=_decimal(close),
        data_age_trading_days=int(age),
        normalized_snapshot=snapshot if isinstance(snapshot, dict) else {},
        decision_id=int(row["decision_id"]) if row["decision_id"] is not None else None,
        openai_model=_optional_str(row, "openai_model"),
        openai_decision=_optional_str(row, "openai_decision"),
        openai_confidence_pct=_optional_decimal(row, "openai_confidence_pct"),
        sentiment=_optional_str(row, "sentiment"),
        market_regime=_optional_str(row, "market_regime"),
        positive_factors=_json_load(row["positive_factors_json"], []) if _has_value(row, "positive_factors_json") else [],
        negative_factors=_json_load(row["negative_factors_json"], []) if _has_value(row, "negative_factors_json") else [],
        risk_flags=_json_load(row["risk_flags_json"], []) if _has_value(row, "risk_flags_json") else [],
        sources=_json_load(row["sources_json"], []) if _has_value(row, "sources_json") else [],
        summary=_optional_str(row, "openai_summary"),
        forward_status=_optional_str(row, "forward_status"),
    )


def _stats(label: str, rows: list[sqlite3.Row]) -> ForwardComparisonStats:
    pending = sum(1 for row in rows if row["forward_status"] == "PENDING")
    completed = [row for row in rows if row["forward_status"] == "COMPLETED"]
    wins = sum(1 for row in completed if row["outcome"] == "WIN")
    losses = sum(1 for row in completed if row["outcome"] == "LOSS")
    returns = [_decimal(row["net_return_pct"]) for row in completed if row["net_return_pct"] is not None]
    pnls = [_decimal(row["net_pnl_sek"]) for row in completed if row["net_pnl_sek"] is not None]
    holding_periods = [
        Decimal(int(row["holding_period_bars"]))
        for row in completed
        if row["holding_period_bars"] is not None
    ]
    closed = wins + losses
    return ForwardComparisonStats(
        label=label,
        pending_decisions=pending,
        completed_trades=len(completed),
        wins=wins,
        losses=losses,
        win_rate=Decimal(wins) / Decimal(closed) if closed else Decimal("0"),
        average_return_pct=_average(returns),
        median_return_pct=Decimal(str(median(returns))) if returns else Decimal("0"),
        total_simulated_pnl_sek=sum(pnls, Decimal("0")),
        average_holding_period=_average(holding_periods),
        stop_exits=sum(1 for row in completed if row["exit_reason"] == "stop_loss"),
    )


def _latest_csv_timestamp(csv_path: Path) -> datetime | None:
    latest: datetime | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("timestamp")
            if not value:
                continue
            timestamp = datetime.fromisoformat(value)
            if latest is None or timestamp > latest:
                latest = timestamp
    return latest


def _optional_str(row: sqlite3.Row, key: str) -> str | None:
    return str(row[key]) if key in row.keys() and row[key] is not None else None


def _optional_decimal(row: sqlite3.Row, key: str) -> Decimal | None:
    return _decimal(row[key]) if key in row.keys() and row[key] is not None else None


def _has_value(row: sqlite3.Row, key: str) -> bool:
    return key in row.keys() and row[key] is not None


def _json_load(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _optional_decimal_value(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _average(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values)) if values else Decimal("0")


def _sum_pnl(rows) -> Decimal:
    return sum((_decimal(row["net_pnl_sek"]) for row in rows if row["net_pnl_sek"] is not None), Decimal("0"))


def _pnl_pct(value: Decimal, starting_capital: Decimal) -> Decimal:
    return (value / starting_capital) * Decimal("100") if starting_capital else Decimal("0")


def _row_datetime(row: sqlite3.Row) -> datetime:
    value = row["exit_time"] or row["completed_at"] or row["scan_timestamp"]
    return datetime.fromisoformat(str(value))


def _row_date(row: sqlite3.Row):
    return _row_datetime(row).date()
