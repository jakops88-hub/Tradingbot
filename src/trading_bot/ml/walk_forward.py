"""Walk-forward ML research across pooled symbols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle
from trading_bot.execution.broker import Broker
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.metrics.performance import buy_and_hold_return, max_drawdown
from trading_bot.ml.dataset import MLTargetMode, MLSample, build_labeled_samples
from trading_bot.ml.model import (
    PredictionMetrics,
    ProbabilityDecisionModel,
    SklearnLogisticDecisionModel,
    prediction_metrics,
)
from trading_bot.strategies.ema_trend import EMATrendConfig, EMATrendStrategy
from trading_bot.strategies.ml_decision import MLDecisionConfig, MLDecisionStrategy


BrokerFactory = Callable[[], Broker]
ModelFactory = Callable[[], ProbabilityDecisionModel]
CALIBRATION_CANDIDATE_THRESHOLDS = (
    Decimal("0.50"),
    Decimal("0.525"),
    Decimal("0.55"),
    Decimal("0.575"),
    Decimal("0.60"),
)
MIN_VALIDATION_TRADES = 4
CALIBRATION_FALLBACK_THRESHOLD = Decimal("0.60")


class MLStrategyVariant(str, Enum):
    RAW_TARGET_FIXED = "raw_target_fixed"
    TRADE_ALIGNED_FIXED = "trade_aligned_fixed"
    TRADE_ALIGNED_CALIBRATED = "trade_aligned_calibrated"
    XGBOOST_TRADE_ALIGNED_CALIBRATED = "xgboost_trade_aligned_calibrated"
    LOGISTIC_TRADE_ALIGNED_CALIBRATED = "logistic_trade_aligned_calibrated"


@dataclass(frozen=True)
class ModelResearchSpec:
    name: str
    variant: MLStrategyVariant
    factory: ModelFactory


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
    winning_trades: int = 0
    losing_trades: int = 0


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
    threshold: Decimal
    calibration: CalibrationResult | None = None
    outer_test_buy_rate: Decimal = Decimal("0")
    outer_test_trades: int = 0


@dataclass(frozen=True)
class AggregateTradingComparison:
    average_return_pct: Decimal
    median_return_pct: Decimal
    profitable_symbol_years: int
    losing_symbol_years: int
    flat_symbol_years: int
    average_max_drawdown: Decimal
    total_trades: int
    weighted_win_rate: Decimal
    ml_vs_ema_wins: int
    ml_vs_buy_and_hold_wins: int


@dataclass(frozen=True)
class CandidateThresholdResult:
    threshold: Decimal
    validation_return_pct: Decimal
    validation_max_drawdown: Decimal
    validation_trades: int
    weighted_win_rate: Decimal
    score: Decimal
    eligible: bool


@dataclass(frozen=True)
class CalibrationResult:
    chosen_threshold: Decimal
    validation_start: datetime
    validation_end: datetime
    validation_trades: int
    validation_return_pct: Decimal
    validation_max_drawdown: Decimal
    candidate_results: list[CandidateThresholdResult]
    internal_training_samples: int
    validation_samples: int
    refit_training_samples: int


@dataclass(frozen=True)
class TargetVariantResearchReport:
    variant: MLStrategyVariant
    model_name: str
    target_mode: MLTargetMode
    folds: list[FoldResearchResult]
    aggregate: AggregateTradingComparison


@dataclass(frozen=True)
class MLResearchReport:
    variants: list[TargetVariantResearchReport]


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
        cost_config: ExecutionCostConfig | None = None,
        model_factory: ModelFactory = SklearnLogisticDecisionModel,
        model_name: str = "LogisticRegression",
    ) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        self.risk_profile = risk_profile
        self.broker_factory = broker_factory
        self.starting_capital = starting_capital
        self.probability_threshold = probability_threshold
        self.cost_config = cost_config or ExecutionCostConfig()
        self.model_factory = model_factory
        self.model_name = model_name

    def evaluate(
        self,
        datasets: dict[str, Sequence[Candle]],
        folds: Sequence[WalkForwardFold] | None = None,
    ) -> MLResearchReport:
        all_folds = list(folds or default_walk_forward_folds())
        variants = [
            self._evaluate_fixed_variant(
                datasets,
                all_folds,
                MLTargetMode.RAW_RETURN,
                MLStrategyVariant.RAW_TARGET_FIXED,
                self.model_factory,
                self.model_name,
            ),
            self._evaluate_fixed_variant(
                datasets,
                all_folds,
                MLTargetMode.TRADE_ALIGNED,
                MLStrategyVariant.TRADE_ALIGNED_FIXED,
                self.model_factory,
                self.model_name,
            ),
            self._evaluate_calibrated_trade_aligned_variant(
                datasets,
                all_folds,
                self.model_factory,
                MLStrategyVariant.TRADE_ALIGNED_CALIBRATED,
                self.model_name,
            ),
        ]
        return MLResearchReport(variants=variants)

    def evaluate_model_comparison(
        self,
        datasets: dict[str, Sequence[Candle]],
        specs: Sequence[ModelResearchSpec],
        folds: Sequence[WalkForwardFold] | None = None,
    ) -> MLResearchReport:
        all_folds = list(folds or default_walk_forward_folds())
        variants = [
            self._evaluate_calibrated_trade_aligned_variant(
                datasets,
                all_folds,
                spec.factory,
                spec.variant,
                spec.name,
            )
            for spec in specs
        ]
        return MLResearchReport(variants=variants)

    def _evaluate_fixed_variant(
        self,
        datasets: dict[str, Sequence[Candle]],
        folds: Sequence[WalkForwardFold],
        target_mode: MLTargetMode,
        variant: MLStrategyVariant,
        model_factory: ModelFactory,
        model_name: str,
    ) -> TargetVariantResearchReport:
        samples_by_symbol = {
            symbol: build_labeled_samples(
                candles,
                target_mode=target_mode,
                cost_config=self.cost_config,
                stop_loss_pct=Decimal("0.05"),
            )
            for symbol, candles in datasets.items()
        }
        fold_results: list[FoldResearchResult] = []
        for fold in folds:
            train_samples = select_training_samples(samples_by_symbol, fold)
            test_samples = select_test_samples(samples_by_symbol, fold)
            model = model_factory()
            model.fit(train_samples)
            probabilities = model.predict_probabilities(test_samples)
            metrics = prediction_metrics(test_samples, probabilities, float(self.probability_threshold))
            symbol_results = [
                self._evaluate_symbol_year(symbol, list(candles), fold, model, self.probability_threshold, model_name)
                for symbol, candles in datasets.items()
            ]
            fold_results.append(
                FoldResearchResult(
                    fold=fold,
                    training_samples=len(train_samples),
                    test_samples=len(test_samples),
                    prediction_metrics=metrics,
                    symbol_results=symbol_results,
                    threshold=self.probability_threshold,
                    outer_test_buy_rate=metrics.predicted_buy_rate,
                    outer_test_trades=sum(comparison.ml.trades for comparison in symbol_results),
                )
            )
        return TargetVariantResearchReport(
            variant=variant,
            model_name=model_name,
            target_mode=target_mode,
            folds=fold_results,
            aggregate=_aggregate_trading_comparison(fold_results),
        )

    def _evaluate_calibrated_trade_aligned_variant(
        self,
        datasets: dict[str, Sequence[Candle]],
        folds: Sequence[WalkForwardFold],
        model_factory: ModelFactory,
        variant: MLStrategyVariant,
        model_name: str,
    ) -> TargetVariantResearchReport:
        samples_by_symbol = {
            symbol: build_labeled_samples(
                candles,
                target_mode=MLTargetMode.TRADE_ALIGNED,
                cost_config=self.cost_config,
                stop_loss_pct=Decimal("0.05"),
            )
            for symbol, candles in datasets.items()
        }
        fold_results: list[FoldResearchResult] = []
        for fold in folds:
            calibration = self._calibrate_threshold(datasets, samples_by_symbol, fold, model_factory, model_name)
            train_samples = select_training_samples(samples_by_symbol, fold)
            test_samples = select_test_samples(samples_by_symbol, fold)
            model = model_factory()
            model.fit(train_samples)
            probabilities = model.predict_probabilities(test_samples)
            metrics = prediction_metrics(test_samples, probabilities, float(calibration.chosen_threshold))
            symbol_results = [
                self._evaluate_symbol_year(symbol, list(candles), fold, model, calibration.chosen_threshold, model_name)
                for symbol, candles in datasets.items()
            ]
            fold_results.append(
                FoldResearchResult(
                    fold=fold,
                    training_samples=len(train_samples),
                    test_samples=len(test_samples),
                    prediction_metrics=metrics,
                    symbol_results=symbol_results,
                    threshold=calibration.chosen_threshold,
                    calibration=CalibrationResult(
                        chosen_threshold=calibration.chosen_threshold,
                        validation_start=calibration.validation_start,
                        validation_end=calibration.validation_end,
                        validation_trades=calibration.validation_trades,
                        validation_return_pct=calibration.validation_return_pct,
                        validation_max_drawdown=calibration.validation_max_drawdown,
                        candidate_results=calibration.candidate_results,
                        internal_training_samples=calibration.internal_training_samples,
                        validation_samples=calibration.validation_samples,
                        refit_training_samples=len(train_samples),
                    ),
                    outer_test_buy_rate=metrics.predicted_buy_rate,
                    outer_test_trades=sum(comparison.ml.trades for comparison in symbol_results),
                )
            )
        return TargetVariantResearchReport(
            variant=variant,
            model_name=model_name,
            target_mode=MLTargetMode.TRADE_ALIGNED,
            folds=fold_results,
            aggregate=_aggregate_trading_comparison(fold_results),
        )

    def _calibrate_threshold(
        self,
        datasets: dict[str, Sequence[Candle]],
        samples_by_symbol: dict[str, Sequence[MLSample]],
        outer_fold: WalkForwardFold,
        model_factory: ModelFactory,
        model_name: str,
    ) -> CalibrationResult:
        validation_fold = validation_fold_for_outer_fold(outer_fold)
        internal_train_samples = select_training_samples(samples_by_symbol, validation_fold)
        validation_samples = select_test_samples(samples_by_symbol, validation_fold)
        model = model_factory()
        model.fit(internal_train_samples)
        candidate_results: list[CandidateThresholdResult] = []
        for threshold in CALIBRATION_CANDIDATE_THRESHOLDS:
            symbol_results = [
                self._evaluate_symbol_year(symbol, list(candles), validation_fold, model, threshold, model_name)
                for symbol, candles in datasets.items()
            ]
            candidate_results.append(evaluate_threshold_candidate(threshold, symbol_results))
        selected = select_calibrated_threshold(candidate_results)
        return CalibrationResult(
            chosen_threshold=selected.threshold,
            validation_start=validation_fold.test_start,
            validation_end=validation_fold.test_end,
            validation_trades=selected.validation_trades,
            validation_return_pct=selected.validation_return_pct,
            validation_max_drawdown=selected.validation_max_drawdown,
            candidate_results=candidate_results,
            internal_training_samples=len(internal_train_samples),
            validation_samples=len(validation_samples),
            refit_training_samples=0,
        )

    def _evaluate_symbol_year(
        self,
        symbol: str,
        candles: list[Candle],
        fold: WalkForwardFold,
        model: ProbabilityDecisionModel,
        threshold: Decimal,
        model_name: str,
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
                    MLDecisionConfig(probability_threshold=threshold),
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
            ml=_strategy_result(fold.test_year, symbol, model_name, ml_result),
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
                winning_trades=1 if buy_hold_return_pct > 0 else 0,
                losing_trades=1 if buy_hold_return_pct < 0 else 0,
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


def validation_fold_for_outer_fold(fold: WalkForwardFold) -> WalkForwardFold:
    validation_year = fold.train_end_year
    internal_train_end_year = validation_year - 1
    if internal_train_end_year < fold.train_start_year:
        raise ValueError("outer fold does not contain enough history for validation calibration")
    return WalkForwardFold(fold.train_start_year, internal_train_end_year, validation_year)


def evaluate_threshold_candidate(
    threshold: Decimal,
    symbol_results: Sequence[SymbolYearComparison],
) -> CandidateThresholdResult:
    ml_results = [comparison.ml for comparison in symbol_results]
    returns = [result.return_pct for result in ml_results]
    drawdowns = [result.max_drawdown for result in ml_results]
    validation_trades = sum(result.trades for result in ml_results)
    winning_trades = sum(result.winning_trades for result in ml_results)
    losing_trades = sum(result.losing_trades for result in ml_results)
    closed_trades = winning_trades + losing_trades
    average_return = sum(returns, Decimal("0")) / Decimal(len(returns)) if returns else Decimal("0")
    average_drawdown = sum(drawdowns, Decimal("0")) / Decimal(len(drawdowns)) if drawdowns else Decimal("0")
    eligible = validation_trades >= MIN_VALIDATION_TRADES
    score = average_return - (average_drawdown * Decimal("100")) if eligible else Decimal("-Infinity")
    return CandidateThresholdResult(
        threshold=threshold,
        validation_return_pct=average_return,
        validation_max_drawdown=average_drawdown,
        validation_trades=validation_trades,
        weighted_win_rate=Decimal(winning_trades) / Decimal(closed_trades) if closed_trades else Decimal("0"),
        score=score,
        eligible=eligible,
    )


def select_calibrated_threshold(
    candidates: Sequence[CandidateThresholdResult],
) -> CandidateThresholdResult:
    if not candidates:
        raise ValueError("candidate threshold results cannot be empty")
    for candidate in candidates:
        if candidate.threshold not in CALIBRATION_CANDIDATE_THRESHOLDS:
            raise ValueError(f"unsupported calibration threshold: {candidate.threshold}")
    eligible_candidates = [candidate for candidate in candidates if candidate.eligible]
    if not eligible_candidates:
        fallback = next(
            (
                candidate
                for candidate in candidates
                if candidate.threshold == CALIBRATION_FALLBACK_THRESHOLD
            ),
            None,
        )
        if fallback is None:
            raise ValueError("fallback threshold result is missing")
        return fallback
    return sorted(
        eligible_candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.validation_return_pct,
            -candidate.validation_max_drawdown,
            -candidate.threshold,
        ),
        reverse=True,
    )[0]


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
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
    )


def _aggregate_trading_comparison(folds: Sequence[FoldResearchResult]) -> AggregateTradingComparison:
    comparisons = [
        comparison
        for fold in folds
        for comparison in fold.symbol_results
    ]
    if not comparisons:
        return AggregateTradingComparison(
            average_return_pct=Decimal("0"),
            median_return_pct=Decimal("0"),
            profitable_symbol_years=0,
            losing_symbol_years=0,
            flat_symbol_years=0,
            average_max_drawdown=Decimal("0"),
            total_trades=0,
            weighted_win_rate=Decimal("0"),
            ml_vs_ema_wins=0,
            ml_vs_buy_and_hold_wins=0,
        )

    ml_results = [comparison.ml for comparison in comparisons]
    returns = [result.return_pct for result in ml_results]
    drawdowns = [result.max_drawdown for result in ml_results]
    winning_trades = sum(result.winning_trades for result in ml_results)
    losing_trades = sum(result.losing_trades for result in ml_results)
    closed_trades = winning_trades + losing_trades
    return AggregateTradingComparison(
        average_return_pct=sum(returns, Decimal("0")) / Decimal(len(returns)),
        median_return_pct=_median(returns),
        profitable_symbol_years=sum(1 for result in ml_results if result.return_pct > 0),
        losing_symbol_years=sum(1 for result in ml_results if result.return_pct < 0),
        flat_symbol_years=sum(1 for result in ml_results if result.return_pct == 0),
        average_max_drawdown=sum(drawdowns, Decimal("0")) / Decimal(len(drawdowns)),
        total_trades=sum(result.trades for result in ml_results),
        weighted_win_rate=Decimal(winning_trades) / Decimal(closed_trades) if closed_trades else Decimal("0"),
        ml_vs_ema_wins=sum(1 for comparison in comparisons if comparison.ml.return_pct > comparison.ema.return_pct),
        ml_vs_buy_and_hold_wins=sum(
            1 for comparison in comparisons if comparison.ml.return_pct > comparison.buy_and_hold.return_pct
        ),
    )


def _median(values: Sequence[Decimal]) -> Decimal:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / Decimal("2")
