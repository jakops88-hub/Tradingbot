"""Forward evaluation for persisted AI scan decisions."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading_bot.data.models import Candle
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.ml.dataset import TARGET_HORIZON_TRADING_DAYS, simulate_trade_outcome
from trading_bot.persistence.sqlite_store import ForwardOutcomeRecord, TradingBotSQLiteStore


LOCKED_FORWARD_COSTS = ExecutionCostConfig(
    percentage_fee=Decimal("0.001"),
    fixed_fee=Decimal("0"),
    slippage_percentage=Decimal("0.001"),
)


@dataclass(frozen=True)
class ForwardComparisonStats:
    label: str
    pending_decisions: int
    completed_trades: int
    wins: int
    losses: int
    win_rate: Decimal
    average_return_pct: Decimal
    median_return_pct: Decimal
    total_simulated_pnl_sek: Decimal
    average_holding_period: Decimal
    stop_exits: int


@dataclass(frozen=True)
class HybridForwardReport:
    pending_decisions: int
    completed_trades: int
    invalid_stale_decisions: int
    newly_completed: int
    groups: list[ForwardComparisonStats]


def evaluate_forward_decisions(
    *,
    store: TradingBotSQLiteStore,
    datasets: Mapping[str, Sequence[Candle]],
) -> HybridForwardReport:
    newly_completed = 0
    for decision in store.pending_decisions():
        candles = sorted(list(datasets.get(decision.symbol, [])), key=lambda candle: candle.timestamp)
        if not candles:
            continue
        feature_index = _decision_index(candles, decision.decision_timestamp)
        if feature_index is None:
            continue
        max_exit_index = feature_index + 1 + TARGET_HORIZON_TRADING_DAYS
        if max_exit_index >= len(candles):
            continue
        outcome = simulate_trade_outcome(
            candles,
            feature_index,
            horizon=TARGET_HORIZON_TRADING_DAYS,
            cost_config=LOCKED_FORWARD_COSTS,
            stop_loss_pct=Decimal("0.05"),
        )
        entry_index = feature_index + 1
        exit_index = _decision_index(candles, outcome.exit_time)
        if exit_index is None:
            continue
        store.complete_forward_outcome(
            ForwardOutcomeRecord(
                decision_id=decision.id,
                entry_time=outcome.entry_time,
                entry_price=outcome.entry_price,
                exit_time=outcome.exit_time,
                exit_price=outcome.exit_price,
                exit_reason=outcome.exit_reason,
                net_pnl_sek=outcome.net_pnl,
                net_return_pct=outcome.net_return * Decimal("100"),
                outcome="WIN" if outcome.net_pnl > 0 else "LOSS",
                holding_period_bars=exit_index - entry_index,
            )
        )
        newly_completed += 1

    completed_rows = store.completed_decisions()
    all_rows = store.all_decisions()
    invalid_rows = [row for row in all_rows if row["forward_status"] == "INVALID_STALE_DATA"]
    pending_count = sum(1 for row in all_rows if row["forward_status"] == "PENDING")
    return HybridForwardReport(
        pending_decisions=pending_count,
        completed_trades=len(completed_rows),
        invalid_stale_decisions=len(invalid_rows),
        newly_completed=newly_completed,
        groups=[
            _stats("ALL XGBoost top-3", completed_rows, all_rows),
            _stats("OpenAI APPROVE", [row for row in completed_rows if row["openai_decision"] == "APPROVE"], all_rows),
            _stats("OpenAI WATCH", [row for row in completed_rows if row["openai_decision"] == "WATCH"], all_rows),
            _stats("OpenAI REJECT", [row for row in completed_rows if row["openai_decision"] == "REJECT"], all_rows),
        ],
    )


def _decision_index(candles: Sequence[Candle], timestamp) -> int | None:
    for index, candle in enumerate(candles):
        if candle.timestamp == timestamp:
            return index
    return None


def _stats(
    label: str,
    completed_rows: Sequence[sqlite3.Row],
    all_rows: Sequence[sqlite3.Row],
) -> ForwardComparisonStats:
    if label == "ALL XGBoost top-3":
        pending = sum(1 for row in all_rows if row["forward_status"] == "PENDING")
    else:
        decision = label.replace("OpenAI ", "")
        pending = sum(
            1
            for row in all_rows
            if row["openai_decision"] == decision and row["forward_status"] == "PENDING"
        )
    returns = [Decimal(str(row["net_return_pct"])) for row in completed_rows]
    pnls = [Decimal(str(row["net_pnl_sek"])) for row in completed_rows]
    holding_periods = [Decimal(int(row["holding_period_bars"])) for row in completed_rows]
    wins = sum(1 for row in completed_rows if row["outcome"] == "WIN")
    losses = sum(1 for row in completed_rows if row["outcome"] == "LOSS")
    closed = wins + losses
    return ForwardComparisonStats(
        label=label,
        pending_decisions=pending,
        completed_trades=len(completed_rows),
        wins=wins,
        losses=losses,
        win_rate=Decimal(wins) / Decimal(closed) if closed else Decimal("0"),
        average_return_pct=_average(returns),
        median_return_pct=_median(returns),
        total_simulated_pnl_sek=sum(pnls, Decimal("0")),
        average_holding_period=_average(holding_periods),
        stop_exits=sum(1 for row in completed_rows if row["exit_reason"] == "stop_loss"),
    )


def _average(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, Decimal("0")) / Decimal(len(values))


def _median(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / Decimal("2")
