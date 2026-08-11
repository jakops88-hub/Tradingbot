"""Run strategy research across independent historical periods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle
from trading_bot.execution.broker import Broker
from trading_bot.research.models import (
    AggregateResearchStats,
    PeriodResult,
    ResearchPeriod,
    ResearchReport,
    SkippedPeriod,
)
from trading_bot.strategies.base import Strategy


StrategyFactory = Callable[[], Strategy]
BrokerFactory = Callable[[], Broker]


class ResearchEvaluator:
    def __init__(
        self,
        *,
        strategy_factory: StrategyFactory,
        broker_factory: BrokerFactory,
        risk_profile: RiskProfile,
        starting_capital: Decimal,
        close_open_positions: bool = True,
    ) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        self.strategy_factory = strategy_factory
        self.broker_factory = broker_factory
        self.risk_profile = risk_profile
        self.starting_capital = starting_capital
        self.close_open_positions = close_open_positions

    def evaluate(
        self,
        candles: Sequence[Candle],
        periods: Sequence[ResearchPeriod],
    ) -> ResearchReport:
        sorted_candles = _validated_chronological_candles(candles)
        if not sorted_candles:
            raise ValueError("candles cannot be empty")

        period_results: list[PeriodResult] = []
        skipped_periods: list[SkippedPeriod] = []
        for period in periods:
            period_candles = _filter_candles(sorted_candles, period)
            if not period_candles:
                skipped_periods.append(SkippedPeriod(period, "no candles in period"))
                continue
            period_results.append(self._run_period(period, period_candles))

        full_period = ResearchPeriod(
            start=sorted_candles[0].timestamp,
            end=sorted_candles[-1].timestamp,
            label="Full History",
        )
        full_history = self._run_period(full_period, sorted_candles)

        return ResearchReport(
            period_results=period_results,
            skipped_periods=skipped_periods,
            aggregate=_aggregate(period_results),
            full_history=full_history,
        )

    def _run_period(self, period: ResearchPeriod, candles: list[Candle]) -> PeriodResult:
        engine = BacktestEngine(
            strategy=self.strategy_factory(),
            risk_profile=self.risk_profile,
            broker=self.broker_factory(),
            starting_cash=self.starting_capital,
            close_open_positions=self.close_open_positions,
        )
        return PeriodResult.from_backtest(period, engine.run(candles))


def yearly_periods(candles: Sequence[Candle]) -> list[ResearchPeriod]:
    sorted_candles = _validated_chronological_candles(candles)
    if not sorted_candles:
        return []

    first_year = sorted_candles[0].timestamp.year
    last_year = sorted_candles[-1].timestamp.year
    return [
        ResearchPeriod(
            start=datetime(year, 1, 1),
            end=datetime(year, 12, 31, 23, 59, 59, 999999),
            label=str(year),
        )
        for year in range(first_year, last_year + 1)
    ]


def _filter_candles(candles: Sequence[Candle], period: ResearchPeriod) -> list[Candle]:
    return [
        candle
        for candle in candles
        if period.start <= candle.timestamp <= period.end
    ]


def _validated_chronological_candles(candles: Sequence[Candle]) -> list[Candle]:
    sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
    timestamps = [candle.timestamp for candle in sorted_candles]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("duplicate candle timestamps are not allowed")
    return sorted_candles


def _aggregate(period_results: Sequence[PeriodResult]) -> AggregateResearchStats:
    if not period_results:
        return AggregateResearchStats(
            average_strategy_return_pct=Decimal("0"),
            median_strategy_return_pct=Decimal("0"),
            average_benchmark_return_pct=Decimal("0"),
            average_benchmark_max_drawdown=Decimal("0"),
            profitable_periods=0,
            losing_periods=0,
            best_period=None,
            worst_period=None,
            average_max_drawdown=Decimal("0"),
            total_trades=0,
        )

    strategy_returns = [period.strategy_return_pct for period in period_results]
    benchmark_returns = [period.benchmark_return_pct for period in period_results]
    benchmark_drawdowns = [period.benchmark_max_drawdown for period in period_results]
    drawdowns = [period.max_drawdown for period in period_results]
    return AggregateResearchStats(
        average_strategy_return_pct=sum(strategy_returns, Decimal("0")) / Decimal(len(strategy_returns)),
        median_strategy_return_pct=_median(strategy_returns),
        average_benchmark_return_pct=sum(benchmark_returns, Decimal("0")) / Decimal(len(benchmark_returns)),
        average_benchmark_max_drawdown=sum(benchmark_drawdowns, Decimal("0")) / Decimal(len(benchmark_drawdowns)),
        profitable_periods=sum(1 for period in period_results if period.net_pnl > 0),
        losing_periods=sum(1 for period in period_results if period.net_pnl < 0),
        best_period=max(period_results, key=lambda period: period.strategy_return_pct),
        worst_period=min(period_results, key=lambda period: period.strategy_return_pct),
        average_max_drawdown=sum(drawdowns, Decimal("0")) / Decimal(len(drawdowns)),
        total_trades=sum(period.total_trades for period in period_results),
    )


def _median(values: Sequence[Decimal]) -> Decimal:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / Decimal("2")
