"""Walk-forward ML research across pooled symbols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle
from trading_bot.execution.broker import Broker
from trading_bot.metrics.performance import buy_and_hold_return, max_drawdown
from trading_bot.ml.dataset import MLSample, build_labeled_samples
from trading_bot.ml.model import PredictionMetrics, SklearnLogisticDecisionModel, prediction_metrics
from trading_bot.strategies.ema_trend import EMATrendConfig, EMATrendStrategy
from trading_bot.strategies.ml_decision import MLDecisionConfig, MLDecisionStrategy


BrokerFactory = Callable[[], Broker]


@dataclass(frozen=True)
class WalkForwardFold:
    train_start_year: int
    train_end_year: int
    test_year: int

    @property
    def test_start(self) -> datetime:
        return datetime(self.test_year, 1, 1)

    @property
    def test_end(self) -> datetime:
        return datetime(self.test_year, 12, 31, 23, 59, 59, 999999)


@dataclass(frozen=True)
class StrategyYearResult:
    year: int
    symbol: str
    strategy_name: str
    return_pct: Decimal
    max_drawdown: Decimal
    trades: int
    win_rate: Decimal
    ending_capital: Decimal


@dataclass(frozen=True)
class SymbolYearComparison:
    year: int
    symbol: str
    ml: StrategyYearResult
    ema: StrategyYearResult
    buy_and_hold: StrategyYearResult


@dataclass(frozen=True)
class FoldResearchResult:
    fold: WalkForwardFold
    training_samples: int
    test_samples: int
    prediction_metrics: PredictionMetrics
    symbol_results: list[SymbolYearComparison]


@dataclass(frozen=True)
class MLResearchReport:
    folds: list[FoldResearchResult]


def default_walk_forward_folds() -> list[WalkForwardFold]:
    return [
        WalkForwardFold(2018, 2021, 2022),
        WalkForwardFold(2018, 2022, 2023),
        WalkForwardFold(2018, 2023, 2024),
        WalkForwardFold(2018, 2024, 2025),
    ]


class MLWalkForwardEvaluator:
    def __init__(
        self,
        *,
        risk_profile: RiskProfile,
        broker_factory: BrokerFactory,
        starting_capital: Decimal,
        probability_threshold: Decimal = Decimal("0.60"),
    ) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        self.risk_profile = risk_profile
        self.broker_factory = broker_factory
        self.starting_capital = starting_capital
        self.probability_threshold = probability_threshold

    def evaluate(
        self,
        datasets: dict[str, Sequence[Candle]],
        folds: Sequence[WalkForwardFold] | None = None,
    ) -> MLResearchReport:
        all_folds = list(folds or default_walk_forward_folds())
        samples_by_symbol = {
            symbol: build_labeled_samples(candles)
            for symbol, candles in datasets.items()
        }
        fold_results: list[FoldResearchResult] = []
        for fold in all_folds:
            train_samples = select_training_samples(samples_by_symbol, fold)
            test_samples = select_test_samples(samples_by_symbol, fold)
            model = SklearnLogisticDecisionModel()
            model.fit(train_samples)
            probabilities = model.predict_probabilities(test_samples)
            metrics = prediction_metrics(test_samples, probabilities, float(self.probability_threshold))
            symbol_results = [
                self._evaluate_symbol_year(symbol, list(candles), fold, model)
                for symbol, candles in datasets.items()
            ]
            fold_results.append(
                FoldResearchResult(
                    fold=fold,
                    training_samples=len(train_samples),
                    test_samples=len(test_samples),
                    prediction_metrics=metrics,
                    symbol_results=symbol_results,
                )
            )
        return MLResearchReport(folds=fold_results)

    def _evaluate_symbol_year(
        self,
        symbol: str,
        candles: list[Candle],
        fold: WalkForwardFold,
        model: SklearnLogisticDecisionModel,
    ) -> SymbolYearComparison:
        sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
        backtest_candles = [candle for candle in sorted_candles if candle.timestamp <= fold.test_end]
        test_candles = [candle for candle in sorted_candles if fold.test_start <= candle.timestamp <= fold.test_end]
        if len(test_candles) < 2:
            raise ValueError(f"not enough candles for {symbol} in {fold.test_year}")

        ml_result = BacktestEngine(
            strategy=TradingWindowStrategy(
                MLDecisionStrategy(
                    model,
                    MLDecisionConfig(probability_threshold=self.probability_threshold),
                ),
                start=fold.test_start,
                end=fold.test_end,
            ),
            risk_profile=self.risk_profile,
            broker=self.broker_factory(),
            starting_cash=self.starting_capital,
            close_open_positions=True,
        ).run(backtest_candles)
        ema_result = BacktestEngine(
            strategy=TradingWindowStrategy(
                EMATrendStrategy(EMATrendConfig(fast_period=20, slow_period=50, stop_loss_pct=Decimal("0.05"))),
                start=fold.test_start,
                end=fold.test_end,
            ),
            risk_profile=self.risk_profile,
            broker=self.broker_factory(),
            starting_cash=self.starting_capital,
            close_open_positions=True,
        ).run(backtest_candles)
        buy_hold_return_pct = buy_and_hold_return(test_candles[0].close, test_candles[-1].close) * Decimal("100")
        buy_hold_curve = [
            self.starting_capital * (candle.close / test_candles[0].close)
            for candle in test_candles
        ]
        return SymbolYearComparison(
            year=fold.test_year,
            symbol=symbol,
            ml=_strategy_result(fold.test_year, symbol, "ML", ml_result),
            ema=_strategy_result(fold.test_year, symbol, "EMA20/50", ema_result),
            buy_and_hold=StrategyYearResult(
                year=fold.test_year,
                symbol=symbol,
                strategy_name="Buy & Hold",
                return_pct=buy_hold_return_pct,
                max_drawdown=max_drawdown(buy_hold_curve),
                trades=1,
                win_rate=Decimal("1") if buy_hold_return_pct > 0 else Decimal("0"),
                ending_capital=self.starting_capital * (Decimal("1") + (buy_hold_return_pct / Decimal("100"))),
            ),
        )


class TradingWindowStrategy:
    def __init__(self, strategy, *, start: datetime, end: datetime) -> None:
        self.strategy = strategy
        self.start = start
        self.end = end
        self.name = f"{strategy.name}_windowed"

    def generate_signal(self, candles, snapshot):
        latest = candles[-1]
        if latest.timestamp < self.start or latest.timestamp > self.end:
            from trading_bot.data.models import Signal, SignalAction

            return Signal(latest.symbol, SignalAction.HOLD, latest.timestamp, reason="outside test window")
        return self.strategy.generate_signal(candles, snapshot)


def select_training_samples(samples_by_symbol: dict[str, Sequence[MLSample]], fold: WalkForwardFold) -> list[MLSample]:
    samples: list[MLSample] = []
    for symbol_samples in samples_by_symbol.values():
        for sample in symbol_samples:
            if sample.feature_time.year < fold.train_start_year or sample.feature_time.year > fold.train_end_year:
                continue
            if target_window_overlaps_period(sample, fold.test_start, fold.test_end):
                continue
            samples.append(sample)
    return samples


def select_test_samples(samples_by_symbol: dict[str, Sequence[MLSample]], fold: WalkForwardFold) -> list[MLSample]:
    return [
        sample
        for symbol_samples in samples_by_symbol.values()
        for sample in symbol_samples
        if sample.feature_time.year == fold.test_year
    ]


def target_window_overlaps_period(sample: MLSample, start: datetime, end: datetime) -> bool:
    return sample.entry_time <= end and sample.exit_time >= start


def _strategy_result(year: int, symbol: str, strategy_name: str, result: BacktestResult) -> StrategyYearResult:
    return StrategyYearResult(
        year=year,
        symbol=symbol,
        strategy_name=strategy_name,
        return_pct=result.strategy_return_pct,
        max_drawdown=result.max_drawdown,
        trades=result.total_trades,
        win_rate=result.win_rate,
        ending_capital=result.ending_capital,
    )
