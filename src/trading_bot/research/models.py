"""Typed models for historical research reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_bot.backtest.engine import BacktestResult


@dataclass(frozen=True)
class ResearchPeriod:
    start: datetime
    end: datetime
    label: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("period end must be greater than or equal to start")


@dataclass(frozen=True)
class PeriodResult:
    period: ResearchPeriod
    starting_capital: Decimal
    ending_capital: Decimal
    strategy_return_pct: Decimal
    benchmark_return_pct: Decimal
    difference_vs_benchmark_pct: Decimal
    total_trades: int
    win_rate: Decimal
    net_pnl: Decimal
    max_drawdown: Decimal
    profit_factor: Decimal | None
    average_position_value: Decimal
    average_portfolio_exposure_pct: Decimal
    largest_position_value: Decimal
    maximum_portfolio_exposure_pct: Decimal
    stop_loss_exits: int
    average_monetary_risk_at_entry: Decimal

    @classmethod
    def from_backtest(cls, period: ResearchPeriod, result: BacktestResult) -> "PeriodResult":
        return cls(
            period=period,
            starting_capital=result.starting_capital,
            ending_capital=result.ending_capital,
            strategy_return_pct=result.strategy_return_pct,
            benchmark_return_pct=result.benchmark_return_pct,
            difference_vs_benchmark_pct=result.difference_vs_benchmark_pct,
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            net_pnl=result.net_pnl,
            max_drawdown=result.max_drawdown,
            profit_factor=result.profit_factor,
            average_position_value=result.average_position_value,
            average_portfolio_exposure_pct=result.average_portfolio_exposure_pct,
            largest_position_value=result.largest_position_value,
            maximum_portfolio_exposure_pct=result.maximum_portfolio_exposure_pct,
            stop_loss_exits=result.stop_loss_exits,
            average_monetary_risk_at_entry=result.average_monetary_risk_at_entry,
        )


@dataclass(frozen=True)
class SkippedPeriod:
    period: ResearchPeriod
    reason: str


@dataclass(frozen=True)
class AggregateResearchStats:
    average_strategy_return_pct: Decimal
    median_strategy_return_pct: Decimal
    average_benchmark_return_pct: Decimal
    profitable_periods: int
    losing_periods: int
    best_period: PeriodResult | None
    worst_period: PeriodResult | None
    average_max_drawdown: Decimal
    total_trades: int


@dataclass(frozen=True)
class ResearchReport:
    period_results: list[PeriodResult]
    skipped_periods: list[SkippedPeriod]
    aggregate: AggregateResearchStats
    full_history: PeriodResult
