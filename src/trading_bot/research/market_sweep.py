"""Evaluate one locked strategy independently across multiple instruments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.market_data import load_csv_candles
from trading_bot.data.metadata import DatasetMetadata, require_matching_currency
from trading_bot.data.models import Candle
from trading_bot.execution.broker import Broker
from trading_bot.strategies.base import Strategy


DatasetFetcher = Callable[[str], Path]
StrategyFactory = Callable[[], Strategy]
BrokerFactory = Callable[[], Broker]


@dataclass(frozen=True)
class MarketSweepInstrumentResult:
    symbol: str
    ending_capital: Decimal
    strategy_return_pct: Decimal
    benchmark_return_pct: Decimal
    difference_vs_benchmark_pct: Decimal
    max_drawdown: Decimal
    total_trades: int
    stop_loss_exits: int
    win_rate: Decimal
    average_exposure_pct: Decimal
    adjustment_policy: str
    candle_count: int
    repaired_ohlc_rows: int
    largest_repaired_ohlc_violation_pct: Decimal
    data_quality_status: str


@dataclass(frozen=True)
class MarketSweepFailure:
    symbol: str
    reason: str


@dataclass(frozen=True)
class MarketSweepSummary:
    profitable_instruments: int
    losing_instruments: int
    average_strategy_return_pct: Decimal
    median_strategy_return_pct: Decimal
    average_benchmark_return_pct: Decimal
    average_max_drawdown: Decimal
    best_strategy_instrument: MarketSweepInstrumentResult | None
    worst_strategy_instrument: MarketSweepInstrumentResult | None
    strategy_beats_buy_and_hold_count: int


@dataclass(frozen=True)
class MarketSweepReport:
    results: list[MarketSweepInstrumentResult]
    failures: list[MarketSweepFailure]
    summary: MarketSweepSummary
    ranking: list[MarketSweepInstrumentResult]
    start_date: datetime
    end_date: datetime
    adjustment_policy: str


class MarketSweepEvaluator:
    def __init__(
        self,
        *,
        dataset_fetcher: DatasetFetcher,
        strategy_factory: StrategyFactory,
        broker_factory: BrokerFactory,
        risk_profile: RiskProfile,
        starting_capital: Decimal,
        portfolio_currency: str,
        expected_adjustment_policy: str,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        self.dataset_fetcher = dataset_fetcher
        self.strategy_factory = strategy_factory
        self.broker_factory = broker_factory
        self.risk_profile = risk_profile
        self.starting_capital = starting_capital
        self.portfolio_currency = portfolio_currency
        self.expected_adjustment_policy = expected_adjustment_policy
        self.start_date = start_date
        self.end_date = end_date

    def evaluate(self, symbols: Sequence[str]) -> MarketSweepReport:
        results: list[MarketSweepInstrumentResult] = []
        failures: list[MarketSweepFailure] = []
        for symbol in symbols:
            try:
                results.append(self._evaluate_symbol(symbol))
            except Exception as exc:
                failures.append(MarketSweepFailure(symbol=symbol, reason=str(exc)))

        ranking = sorted(results, key=lambda result: result.strategy_return_pct, reverse=True)
        return MarketSweepReport(
            results=results,
            failures=failures,
            summary=_summarize(results),
            ranking=ranking,
            start_date=self.start_date,
            end_date=self.end_date,
            adjustment_policy=self.expected_adjustment_policy,
        )

    def _evaluate_symbol(self, symbol: str) -> MarketSweepInstrumentResult:
        csv_path = self.dataset_fetcher(symbol)
        metadata = require_matching_currency(csv_path, self.portfolio_currency)
        adjustment_policy = _require_adjustment_policy(
            symbol=symbol,
            metadata=metadata,
            expected_adjustment_policy=self.expected_adjustment_policy,
        )
        candles = [
            candle
            for candle in load_csv_candles(csv_path, symbol)
            if self.start_date <= candle.timestamp <= self.end_date
        ]
        if not candles:
            raise ValueError("no candles in requested sweep period")
        result = BacktestEngine(
            strategy=self.strategy_factory(),
            risk_profile=self.risk_profile,
            broker=self.broker_factory(),
            starting_cash=self.starting_capital,
            close_open_positions=True,
        ).run(candles)
        return MarketSweepInstrumentResult(
            symbol=symbol,
            ending_capital=result.ending_capital,
            strategy_return_pct=result.strategy_return_pct,
            benchmark_return_pct=result.benchmark_return_pct,
            difference_vs_benchmark_pct=result.difference_vs_benchmark_pct,
            max_drawdown=result.max_drawdown,
            total_trades=result.total_trades,
            stop_loss_exits=result.stop_loss_exits,
            win_rate=result.win_rate,
            average_exposure_pct=result.average_portfolio_exposure_pct,
            adjustment_policy=adjustment_policy,
            candle_count=len(candles),
            repaired_ohlc_rows=metadata.repaired_ohlc_rows if metadata else 0,
            largest_repaired_ohlc_violation_pct=Decimal(
                metadata.largest_repaired_ohlc_violation_pct if metadata else "0"
            ),
            data_quality_status="PASS",
        )


def load_symbol_config(path: str | Path) -> list[str]:
    source = Path(path)
    symbols = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not symbols:
        raise ValueError(f"No symbols configured in {source}")
    return symbols


def _require_adjustment_policy(
    *,
    symbol: str,
    metadata: DatasetMetadata | None,
    expected_adjustment_policy: str,
) -> str:
    if metadata is None:
        raise ValueError(f"missing dataset metadata for {symbol}")
    if metadata.adjustment_policy != expected_adjustment_policy:
        raise ValueError(
            f"adjustment policy mismatch for {symbol}: "
            f"{metadata.adjustment_policy} != {expected_adjustment_policy}"
        )
    return metadata.adjustment_policy


def _summarize(results: Sequence[MarketSweepInstrumentResult]) -> MarketSweepSummary:
    if not results:
        return MarketSweepSummary(
            profitable_instruments=0,
            losing_instruments=0,
            average_strategy_return_pct=Decimal("0"),
            median_strategy_return_pct=Decimal("0"),
            average_benchmark_return_pct=Decimal("0"),
            average_max_drawdown=Decimal("0"),
            best_strategy_instrument=None,
            worst_strategy_instrument=None,
            strategy_beats_buy_and_hold_count=0,
        )
    strategy_returns = [result.strategy_return_pct for result in results]
    benchmark_returns = [result.benchmark_return_pct for result in results]
    drawdowns = [result.max_drawdown for result in results]
    return MarketSweepSummary(
        profitable_instruments=sum(1 for result in results if result.strategy_return_pct > 0),
        losing_instruments=sum(1 for result in results if result.strategy_return_pct < 0),
        average_strategy_return_pct=sum(strategy_returns, Decimal("0")) / Decimal(len(results)),
        median_strategy_return_pct=_median(strategy_returns),
        average_benchmark_return_pct=sum(benchmark_returns, Decimal("0")) / Decimal(len(results)),
        average_max_drawdown=sum(drawdowns, Decimal("0")) / Decimal(len(results)),
        best_strategy_instrument=max(results, key=lambda result: result.strategy_return_pct),
        worst_strategy_instrument=min(results, key=lambda result: result.strategy_return_pct),
        strategy_beats_buy_and_hold_count=sum(
            1 for result in results if result.strategy_return_pct > result.benchmark_return_pct
        ),
    )


def _median(values: Sequence[Decimal]) -> Decimal:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / Decimal("2")
