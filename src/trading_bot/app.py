"""Command-line entrypoint for offline historical backtesting."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.market_data import load_csv_candles
from trading_bot.data.metadata import require_matching_currency
from trading_bot.data.models import Candle
from trading_bot.data.yahoo_finance import YahooFinanceDataProvider
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.ml.dataset import MLTargetMode
from trading_bot.ml.model import SklearnLogisticDecisionModel, XGBoostDecisionModel
from trading_bot.ml.walk_forward import MLResearchReport, MLStrategyVariant, MLWalkForwardEvaluator, ModelResearchSpec
from trading_bot.research.evaluator import ResearchEvaluator, yearly_periods
from trading_bot.research.market_sweep import (
    MarketSweepEvaluator,
    MarketSweepInstrumentResult,
    MarketSweepReport,
    load_symbol_config,
)
from trading_bot.research.models import PeriodResult, ResearchReport
from trading_bot.strategies.ema_trend import EMATrendConfig, EMATrendStrategy


def demo_candles() -> list[Candle]:
    start = datetime(2024, 1, 1)
    prices = [Decimal("100") - Decimal(index) for index in range(40)]
    prices.extend(Decimal("61") + Decimal(index * 3) for index in range(35))
    return [
        Candle(
            symbol="ABC",
            timestamp=start + timedelta(days=index),
            open=price,
            high=price + Decimal("1"),
            low=price - Decimal("1"),
            close=price,
            volume=Decimal("1000"),
        )
        for index, price in enumerate(prices)
    ]


DEFAULT_COSTS = ExecutionCostConfig(
    percentage_fee=Decimal("0.001"),
    fixed_fee=Decimal("0.05"),
    slippage_percentage=Decimal("0.001"),
)
LOCKED_START_DATE = datetime(2018, 1, 1)
LOCKED_END_DATE = datetime(2026, 1, 1)
LOCKED_ADJUSTMENT_POLICY = "adjusted"


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "research":
        run_research_command(args)
        return
    if args.command == "download":
        run_download_command(args)
        return
    if args.command == "market-sweep":
        run_market_sweep_command(args)
        return
    if args.command == "ml-research":
        run_ml_research_command(args)
        return
    if args.command == "xgb-research":
        run_xgb_research_command(args)
        return
    if args.command == "ai-scan":
        run_ai_scan_command(args)
        return

    run_demo()


def run_demo() -> None:
    engine = BacktestEngine(
        strategy=EMATrendStrategy(EMATrendConfig(fast_period=20, slow_period=50)),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(DEFAULT_COSTS),
        starting_cash=Decimal("1000"),
        close_open_positions=True,
    )
    result = engine.run(demo_candles())

    print("TradingBot EMA 20/50 historical simulation")
    print("This is not evidence of future profitability.")
    print("Starting capital: 1000 SEK")
    print(f"Ending capital: {result.ending_capital} SEK")
    print(f"Strategy return: {result.strategy_return_pct}%")
    print(f"Buy & hold return: {result.benchmark_return_pct}%")
    print(f"Difference vs benchmark: {result.difference_vs_benchmark_pct}%")
    print(f"Trades: {result.total_trades}")
    print(f"Win rate: {result.win_rate * Decimal('100')}%")
    print(f"Net PnL: {result.net_pnl} SEK")
    print(f"Max drawdown: {result.max_drawdown * Decimal('100')}%")
    print(f"Buy & hold max drawdown: {result.benchmark_max_drawdown * Decimal('100')}%")
    print(f"Profit factor: {result.profit_factor}")
    print("Real-money trading: unavailable")


def run_research_command(args: argparse.Namespace) -> None:
    data_path = Path(args.csv_path)
    risk_mode = RiskMode[args.risk.upper()]
    require_matching_currency(data_path, args.portfolio_currency)
    costs = ExecutionCostConfig(
        percentage_fee=Decimal(args.percentage_fee),
        fixed_fee=Decimal(args.fixed_fee),
        slippage_percentage=Decimal(args.slippage),
    )
    candles = load_csv_candles(data_path, args.symbol)
    evaluator = ResearchEvaluator(
        strategy_factory=lambda: EMATrendStrategy(EMATrendConfig(fast_period=20, slow_period=50)),
        broker_factory=lambda: PaperBroker(costs),
        risk_profile=get_risk_profile(risk_mode),
        starting_capital=Decimal("1000"),
        close_open_positions=True,
    )
    report = evaluator.evaluate(candles, yearly_periods(candles))
    print_research_report(report)


def run_download_command(args: argparse.Namespace) -> None:
    provider = YahooFinanceDataProvider(
        cache_dir=args.output_dir,
        adjustment_policy=args.adjustment_policy,
    )
    csv_path = provider.download_to_csv(
        symbol=args.symbol,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    print(f"Downloaded normalized daily OHLCV data: {csv_path}")
    print(f"Metadata: {csv_path}.metadata.json")
    print(f"Adjustment policy: {args.adjustment_policy}")
    print("Use this file with the research command.")


def run_market_sweep_command(args: argparse.Namespace) -> None:
    symbols = load_symbol_config(args.symbols)
    output_dir = Path(args.output_dir)
    costs = ExecutionCostConfig(
        percentage_fee=Decimal("0.001"),
        fixed_fee=Decimal("0"),
        slippage_percentage=Decimal("0.001"),
    )
    provider = YahooFinanceDataProvider(
        cache_dir=output_dir,
        adjustment_policy=LOCKED_ADJUSTMENT_POLICY,
    )

    def fetch_dataset(symbol: str) -> Path:
        return provider.download_to_csv(
            symbol=symbol,
            start=LOCKED_START_DATE,
            end=LOCKED_END_DATE,
        )

    evaluator = MarketSweepEvaluator(
        dataset_fetcher=fetch_dataset,
        strategy_factory=lambda: EMATrendStrategy(
            EMATrendConfig(fast_period=20, slow_period=50, stop_loss_pct=Decimal("0.05"))
        ),
        broker_factory=lambda: PaperBroker(costs),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        starting_capital=Decimal("1000"),
        portfolio_currency="SEK",
        expected_adjustment_policy=LOCKED_ADJUSTMENT_POLICY,
        start_date=LOCKED_START_DATE,
        end_date=LOCKED_END_DATE,
    )
    print_market_sweep_report(evaluator.evaluate(symbols))


def run_ml_research_command(args: argparse.Namespace) -> None:
    costs = ExecutionCostConfig(
        percentage_fee=Decimal("0.001"),
        fixed_fee=Decimal("0"),
        slippage_percentage=Decimal("0.001"),
    )
    datasets = _load_symbol_datasets(
        symbols_path=args.symbols,
        output_dir=args.output_dir,
        start=LOCKED_START_DATE,
        end=LOCKED_END_DATE,
    )

    evaluator = MLWalkForwardEvaluator(
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker_factory=lambda: PaperBroker(costs),
        starting_capital=Decimal("1000"),
        probability_threshold=Decimal("0.60"),
        cost_config=costs,
    )
    print_ml_research_report(evaluator.evaluate(datasets))


def run_xgb_research_command(args: argparse.Namespace) -> None:
    costs = ExecutionCostConfig(
        percentage_fee=Decimal("0.001"),
        fixed_fee=Decimal("0"),
        slippage_percentage=Decimal("0.001"),
    )
    datasets = _load_symbol_datasets(
        symbols_path=args.symbols,
        output_dir=args.output_dir,
        start=LOCKED_START_DATE,
        end=LOCKED_END_DATE,
    )
    evaluator = MLWalkForwardEvaluator(
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker_factory=lambda: PaperBroker(costs),
        starting_capital=Decimal("1000"),
        probability_threshold=Decimal("0.60"),
        cost_config=costs,
    )
    report = evaluator.evaluate_model_comparison(
        datasets,
        [
            ModelResearchSpec(
                name="XGBoost",
                variant=MLStrategyVariant.XGBOOST_TRADE_ALIGNED_CALIBRATED,
                factory=XGBoostDecisionModel,
            ),
            ModelResearchSpec(
                name="LogisticRegression",
                variant=MLStrategyVariant.LOGISTIC_TRADE_ALIGNED_CALIBRATED,
                factory=SklearnLogisticDecisionModel,
            ),
        ],
    )
    print_ml_research_report(report)


def run_ai_scan_command(args: argparse.Namespace) -> None:
    from trading_bot.ai.openai_analyst import OpenAIAnalyst
    from trading_bot.ai.scanner import HybridMarketScanner

    costs = ExecutionCostConfig(
        percentage_fee=Decimal("0.001"),
        fixed_fee=Decimal("0"),
        slippage_percentage=Decimal("0.001"),
    )
    scan_end = _parse_date(args.end) if args.end else LOCKED_END_DATE
    datasets = _load_symbol_datasets(
        symbols_path=args.symbols,
        output_dir=args.output_dir,
        start=LOCKED_START_DATE,
        end=scan_end,
    )
    scanner = HybridMarketScanner(
        analyst=OpenAIAnalyst(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        cost_config=costs,
    )
    print_ai_scan_report(scanner.scan(datasets))


def _load_symbol_datasets(
    *,
    symbols_path: str,
    output_dir: str,
    start: datetime,
    end: datetime,
) -> dict[str, list[Candle]]:
    symbols = load_symbol_config(symbols_path)
    provider = YahooFinanceDataProvider(
        cache_dir=Path(output_dir),
        adjustment_policy=LOCKED_ADJUSTMENT_POLICY,
    )
    datasets: dict[str, list[Candle]] = {}
    for symbol in symbols:
        csv_path = provider.download_to_csv(
            symbol=symbol,
            start=start,
            end=end,
        )
        require_matching_currency(csv_path, "SEK")
        candles = [
            candle
            for candle in load_csv_candles(csv_path, symbol)
            if start <= candle.timestamp <= end
        ]
        if candles:
            datasets[symbol] = candles
    if not datasets:
        raise ValueError("No datasets available")
    return datasets


def print_research_report(report: ResearchReport) -> None:
    print("TradingBot EMA 20/50 historical research")
    print("This is a historical simulation, not evidence of future profitability.")
    print("")
    print(
        f"{'Period':<12}{'Strategy':>12}{'Buy&Hold':>12}{'Difference':>14}"
        f"{'Trades':>8}{'Max DD':>10}{'B&H DD':>10}"
    )
    for result in report.period_results:
        print(_period_row(result))
    for skipped in report.skipped_periods:
        print(f"{skipped.period.label:<12}{'skipped':>12}  {skipped.reason}")
    print("")
    print("Diagnostics")
    print(f"Average position value: {report.full_history.average_position_value} SEK")
    print(f"Average portfolio exposure: {report.full_history.average_portfolio_exposure_pct.quantize(Decimal('0.01'))}%")
    print(f"Largest position value: {report.full_history.largest_position_value} SEK")
    print(f"Maximum portfolio exposure: {report.full_history.maximum_portfolio_exposure_pct.quantize(Decimal('0.01'))}%")
    print(f"Stop-loss exits: {report.full_history.stop_loss_exits}")
    print(f"Average monetary risk at entry: {report.full_history.average_monetary_risk_at_entry} SEK")
    print("")
    print("Aggregate")
    aggregate = report.aggregate
    print(f"Average strategy return: {_format_pct(aggregate.average_strategy_return_pct)}")
    print(f"Median strategy return: {_format_pct(aggregate.median_strategy_return_pct)}")
    print(f"Average buy & hold return: {_format_pct(aggregate.average_benchmark_return_pct)}")
    print(f"Average buy & hold max drawdown: {_format_pct(aggregate.average_benchmark_max_drawdown * Decimal('100'), signed=False)}")
    print(f"Profitable periods: {aggregate.profitable_periods}")
    print(f"Losing periods: {aggregate.losing_periods}")
    print(f"Best period: {aggregate.best_period.period.label if aggregate.best_period else 'n/a'}")
    print(f"Worst period: {aggregate.worst_period.period.label if aggregate.worst_period else 'n/a'}")
    print(f"Average max drawdown: {_format_pct(aggregate.average_max_drawdown * Decimal('100'), signed=False)}")
    print(f"Total trades: {aggregate.total_trades}")
    print("")
    print("Full History")
    print(_period_row(report.full_history))
    print("Real-money trading: unavailable")


def print_market_sweep_report(report: MarketSweepReport) -> None:
    print("TradingBot locked EMA 20/50 Swedish large-cap market sweep")
    print("This is a historical simulation, not evidence of future profitability.")
    print(f"Period: {report.start_date.date().isoformat()} to {report.end_date.date().isoformat()}")
    print(f"Adjustment policy: {report.adjustment_policy}")
    print("Locked settings: EMA 20/50, 5% stop, MEDIUM risk, 1000 SEK, 0.1% fee, 0.1% slippage")
    print("")
    print(
        f"{'Symbol':<12}{'Strategy':>12}{'Buy&Hold':>12}{'Diff':>10}"
        f"{'Max DD':>10}{'B&H DD':>10}{'Trades':>8}{'Stops':>8}{'Win':>8}{'Avg Exp':>10}{'Ending':>12}"
    )
    for result in report.results:
        print(_market_sweep_row(result))
    if report.failures:
        print("")
        print("Failed symbols")
        for failure in report.failures:
            print(f"{failure.symbol}: {failure.reason}")
    print("")
    print("Data Quality")
    print(f"{'Symbol':<12}{'Candles':>10}{'Repaired':>12}{'Largest Repair':>18}{'Status':>10}")
    for result in report.results:
        print(
            f"{result.symbol:<12}"
            f"{result.candle_count:>10}"
            f"{result.repaired_ohlc_rows:>12}"
            f"{_format_data_quality_pct(result.largest_repaired_ohlc_violation_pct):>18}"
            f"{result.data_quality_status:>10}"
        )
    for failure in report.failures:
        print(f"{failure.symbol:<12}{'n/a':>10}{'n/a':>12}{'n/a':>18}{'FAIL':>10}  {failure.reason}")
    print("")
    print("Cross-market summary")
    summary = report.summary
    print(f"Profitable instruments: {summary.profitable_instruments}")
    print(f"Losing instruments: {summary.losing_instruments}")
    print(f"Average strategy return: {_format_pct(summary.average_strategy_return_pct)}")
    print(f"Median strategy return: {_format_pct(summary.median_strategy_return_pct)}")
    print(f"Average buy & hold return: {_format_pct(summary.average_benchmark_return_pct)}")
    print(f"Average max drawdown: {_format_pct(summary.average_max_drawdown * Decimal('100'), signed=False)}")
    print(f"Average buy & hold max drawdown: {_format_pct(summary.average_benchmark_max_drawdown * Decimal('100'), signed=False)}")
    print(f"Best strategy instrument: {summary.best_strategy_instrument.symbol if summary.best_strategy_instrument else 'n/a'}")
    print(f"Worst strategy instrument: {summary.worst_strategy_instrument.symbol if summary.worst_strategy_instrument else 'n/a'}")
    print(f"Strategy beat buy & hold count: {summary.strategy_beats_buy_and_hold_count}")
    print("")
    print("Ranking")
    for index, result in enumerate(report.ranking, start=1):
        print(f"{index:>2}. {result.symbol:<12} {_format_pct(result.strategy_return_pct):>10}")
    print("Real-money trading: unavailable")


def print_ml_research_report(report: MLResearchReport) -> None:
    print("TradingBot ML walk-forward research")
    print("This is out-of-sample historical research, not evidence of future profitability.")
    print("Models: XGBoost primary where present; LogisticRegression retained as a baseline.")
    print("Targets: trade-aligned net PnL after stop, fees, slippage, and max hold; raw target appears only in legacy baseline reports.")
    print("Locked settings: 5% stop, 10-day max hold, MEDIUM risk, 1000 SEK, 0.1% fee, 0.1% slippage")
    print("Thresholds: fixed 60% baseline plus leakage-safe calibrated trade-aligned thresholds from 50%, 52.5%, 55%, 57.5%, 60%.")
    print("OpenAI is not used in historical backtests.")
    print("")
    for variant in report.variants:
        print(_variant_heading(variant))
        print("")
        for fold in variant.folds:
            metrics = fold.prediction_metrics
            print(f"Fold: train {fold.fold.train_start_year}-{fold.fold.train_end_year}, test {fold.fold.test_year}")
            print(f"ML threshold used: {_format_pct(fold.threshold * Decimal('100'), signed=False)}")
            print(f"Training samples: {fold.training_samples}")
            print(f"Out-of-sample predictions: {metrics.predictions}")
            print(
                "Prediction metrics: "
                f"class_balance={_format_pct(metrics.positive_class_rate * Decimal('100'), signed=False)}, "
                f"buy_rate={_format_pct(metrics.predicted_buy_rate * Decimal('100'), signed=False)}, "
                f"accuracy={_format_decimal(metrics.accuracy)}, "
                f"precision={_format_decimal(metrics.precision)}, "
                f"recall={_format_decimal(metrics.recall)}, "
                f"roc_auc={_format_decimal(metrics.roc_auc) if metrics.roc_auc is not None else 'n/a'}"
            )
            print(
                "Probability diagnostics: "
                f"avg={_format_pct(metrics.average_predicted_probability * Decimal('100'), signed=False)}, "
                f"avg_pos={_format_pct(metrics.average_probability_for_positive_labels * Decimal('100'), signed=False)}, "
                f"avg_neg={_format_pct(metrics.average_probability_for_negative_labels * Decimal('100'), signed=False)}"
            )
            if fold.calibration is not None:
                calibration = fold.calibration
                print("Calibration diagnostics")
                print(
                    f"Validation period: {calibration.validation_start.date().isoformat()} "
                    f"to {calibration.validation_end.date().isoformat()}"
                )
                print(f"Internal training samples: {calibration.internal_training_samples}")
                print(f"Validation samples: {calibration.validation_samples}")
                print(f"Refit training samples: {calibration.refit_training_samples}")
                print(f"Chosen threshold: {_format_pct(calibration.chosen_threshold * Decimal('100'), signed=False)}")
                print(f"Validation trades: {calibration.validation_trades}")
                print(f"Validation return: {_format_pct(calibration.validation_return_pct)}")
                print(f"Validation max drawdown: {_format_pct(calibration.validation_max_drawdown * Decimal('100'), signed=False)}")
                print(f"Outer test buy rate: {_format_pct(fold.outer_test_buy_rate * Decimal('100'), signed=False)}")
                print(f"Outer test trades: {fold.outer_test_trades}")
                print(f"{'Threshold':<12}{'Eligible':>10}{'Return':>10}{'Max DD':>10}{'Trades':>8}{'Win':>8}{'Score':>12}")
                for candidate in calibration.candidate_results:
                    print(
                        f"{_format_pct(candidate.threshold * Decimal('100'), signed=False):<12}"
                        f"{str(candidate.eligible):>10}"
                        f"{_format_pct(candidate.validation_return_pct):>10}"
                        f"{_format_pct(candidate.validation_max_drawdown * Decimal('100'), signed=False):>10}"
                        f"{candidate.validation_trades:>8}"
                        f"{_format_pct(candidate.weighted_win_rate * Decimal('100'), signed=False):>8}"
                        f"{_format_candidate_score(candidate.score):>12}"
                    )
            print(f"{'Bucket':<10}{'Predictions':>12}{'Positive':>12}{'Avg Net Ret':>14}")
            for bucket in metrics.probability_buckets:
                print(
                    f"{bucket.label:<10}"
                    f"{bucket.predictions:>12}"
                    f"{_format_pct(bucket.actual_positive_rate * Decimal('100'), signed=False):>12}"
                    f"{_format_pct(bucket.average_net_trade_return * Decimal('100')):>14}"
                )
            print(
                f"{'Symbol':<12}{'Strategy':<12}{'Return':>10}{'Max DD':>10}"
                f"{'Trades':>8}{'Win':>8}{'Ending':>12}"
            )
            for comparison in fold.symbol_results:
                for result in (comparison.ml, comparison.ema, comparison.buy_and_hold):
                    print(
                        f"{comparison.symbol:<12}"
                        f"{result.strategy_name:<12}"
                        f"{_format_pct(result.return_pct):>10}"
                        f"{_format_pct(result.max_drawdown * Decimal('100'), signed=False):>10}"
                        f"{result.trades:>8}"
                        f"{_format_pct(result.win_rate * Decimal('100'), signed=False):>8}"
                        f"{result.ending_capital.quantize(Decimal('0.01')):>12}"
                    )
            print("")
        aggregate = variant.aggregate
        print("Aggregate out-of-sample ML trading comparison")
        print(f"Average return: {_format_pct(aggregate.average_return_pct)}")
        print(f"Median return: {_format_pct(aggregate.median_return_pct)}")
        print(f"Profitable symbol-years: {aggregate.profitable_symbol_years}")
        print(f"Losing symbol-years: {aggregate.losing_symbol_years}")
        print(f"Flat symbol-years: {aggregate.flat_symbol_years}")
        print(f"Average max drawdown: {_format_pct(aggregate.average_max_drawdown * Decimal('100'), signed=False)}")
        print(f"Total trades: {aggregate.total_trades}")
        print(f"Weighted win rate: {_format_pct(aggregate.weighted_win_rate * Decimal('100'), signed=False)}")
        print(f"ML vs EMA wins: {aggregate.ml_vs_ema_wins}")
        print(f"ML vs buy & hold wins: {aggregate.ml_vs_buy_and_hold_wins}")
        print("")
    print("Real-money trading: unavailable")


def print_ai_scan_report(report) -> None:
    print("TradingBot current-market XGBoost + OpenAI advisory scan")
    print("This is a research scan only. It cannot trade real money.")
    print("OpenAI is advisory only and cannot change risk settings, stops, exposure, or call a broker.")
    print("")
    print(f"{'Rank':>4}  {'Symbol':<12}{'XGB Prob':>10}{'Momentum':>12}{'Volatility':>12}{'Volume':>12}")
    for index, candidate in enumerate(report.ranked_candidates, start=1):
        print(
            f"{index:>4}  "
            f"{candidate.symbol:<12}"
            f"{_format_pct(candidate.xgboost_probability * Decimal('100'), signed=False):>10}"
            f"{_format_pct(candidate.momentum * Decimal('100')):>12}"
            f"{_format_pct(candidate.volatility * Decimal('100'), signed=False):>12}"
            f"{_format_pct(candidate.volume_vs_average * Decimal('100')):>12}"
        )
    print("")
    print(f"OpenAI analyses requested for top {report.max_openai_analyses} candidates")
    for analyzed in report.analyzed_candidates:
        candidate = analyzed.candidate
        analysis = analyzed.openai_analysis
        print("")
        print(f"{candidate.symbol} | XGBoost probability: {_format_pct(candidate.xgboost_probability * Decimal('100'), signed=False)}")
        if analysis is None:
            print("OpenAI: not analyzed")
            continue
        print(
            f"OpenAI: {analysis.decision.value}, confidence {_format_pct(analysis.confidence * Decimal('100'), signed=False)}, "
            f"sentiment {analysis.sentiment.value}, regime {analysis.regime.value}"
        )
        if analysis.safe_failure:
            print(f"Safe failure: {analysis.error}")
        print(f"Summary: {analysis.summary}")
        print(f"Positive factors: {_format_list(analysis.positive_factors)}")
        print(f"Negative factors: {_format_list(analysis.negative_factors)}")
        print(f"Risk flags: {_format_list(analysis.risk_flags)}")
    print("")
    print("Real-money trading: unavailable")


def _market_sweep_row(result: MarketSweepInstrumentResult) -> str:
    return (
        f"{result.symbol:<12}"
        f"{_format_pct(result.strategy_return_pct):>12}"
        f"{_format_pct(result.benchmark_return_pct):>12}"
        f"{_format_pct(result.difference_vs_benchmark_pct):>10}"
        f"{_format_pct(result.max_drawdown * Decimal('100'), signed=False):>10}"
        f"{_format_pct(result.benchmark_max_drawdown * Decimal('100'), signed=False):>10}"
        f"{result.total_trades:>8}"
        f"{result.stop_loss_exits:>8}"
        f"{_format_pct(result.win_rate * Decimal('100'), signed=False):>8}"
        f"{_format_pct(result.average_exposure_pct, signed=False):>10}"
        f"{result.ending_capital.quantize(Decimal('0.01')):>12}"
    )


def _period_row(result: PeriodResult) -> str:
    return (
        f"{result.period.label:<12}"
        f"{_format_pct(result.strategy_return_pct):>12}"
        f"{_format_pct(result.benchmark_return_pct):>12}"
        f"{_format_pct(result.difference_vs_benchmark_pct):>14}"
        f"{result.total_trades:>8}"
        f"{_format_pct(result.max_drawdown * Decimal('100'), signed=False):>10}"
        f"{_format_pct(result.benchmark_max_drawdown * Decimal('100'), signed=False):>10}"
    )


def _format_pct(value: Decimal, *, signed: bool = True) -> str:
    sign = "+" if signed else ""
    return f"{value.quantize(Decimal('0.01')):{sign}}%"


def _format_data_quality_pct(value: Decimal) -> str:
    if value == 0:
        return "0%"
    if abs(value) >= Decimal("0.01"):
        return _format_pct(value, signed=False)
    return f"{value:.2E}%"


def _format_decimal(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.0001")))


def _format_candidate_score(value: Decimal) -> str:
    if value.is_infinite():
        return "ineligible"
    return str(value.quantize(Decimal("0.0001")))


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _variant_heading(variant) -> str:
    if variant.variant.value == "xgboost_trade_aligned_calibrated":
        return "XGBOOST - TRADE-ALIGNED TARGET - CALIBRATED THRESHOLD"
    if variant.variant.value == "logistic_trade_aligned_calibrated":
        return "LOGISTIC REGRESSION BASELINE - TRADE-ALIGNED TARGET - CALIBRATED THRESHOLD"
    if variant.variant.value == "raw_target_fixed":
        return f"{variant.model_name.upper()} RAW TARGET BASELINE - FIXED 60%"
    if variant.variant.value == "trade_aligned_fixed":
        return f"{variant.model_name.upper()} TRADE-ALIGNED TARGET - FIXED 60%"
    if variant.variant.value == "trade_aligned_calibrated":
        return f"{variant.model_name.upper()} TRADE-ALIGNED TARGET - CALIBRATED THRESHOLD"
    if variant.target_mode == MLTargetMode.RAW_RETURN:
        return "RAW TARGET"
    if variant.target_mode == MLTargetMode.TRADE_ALIGNED:
        return "TRADE-ALIGNED TARGET"
    return str(variant.target_mode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TradingBot offline tools")
    subparsers = parser.add_subparsers(dest="command")
    research = subparsers.add_parser("research", help="Run EMA 20/50 research on a local OHLCV CSV")
    research.add_argument("csv_path", help="Path to a local OHLCV CSV file")
    research.add_argument("--symbol", default="ABC", help="Symbol label for the CSV data")
    research.add_argument("--risk", default="MEDIUM", choices=[mode.value for mode in RiskMode])
    research.add_argument("--portfolio-currency", default="SEK")
    research.add_argument("--percentage-fee", default="0.001")
    research.add_argument("--fixed-fee", default="0.05")
    research.add_argument("--slippage", default="0.001")
    download = subparsers.add_parser("download", help="Download normalized daily OHLCV data with yfinance")
    download.add_argument("symbol", help="Yahoo Finance ticker, for example VOLV-B.ST")
    download.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    download.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    download.add_argument("--output-dir", default="data", help="Directory for normalized CSV cache")
    download.add_argument(
        "--adjustment-policy",
        default=LOCKED_ADJUSTMENT_POLICY,
        choices=["adjusted", "unadjusted"],
        help="Use adjusted OHLC prices by default for equity research",
    )
    market_sweep = subparsers.add_parser("market-sweep", help="Run locked EMA 20/50 sweep across configured Swedish symbols")
    market_sweep.add_argument("--symbols", default="config/swedish_large_caps.txt")
    market_sweep.add_argument("--output-dir", default="data")
    ml_research = subparsers.add_parser("ml-research", help="Run walk-forward ML decision research")
    ml_research.add_argument("--symbols", default="config/swedish_large_caps.txt")
    ml_research.add_argument("--output-dir", default="data")
    xgb_research = subparsers.add_parser("xgb-research", help="Run XGBoost vs LogisticRegression historical research")
    xgb_research.add_argument("--symbols", default="config/swedish_large_caps.txt")
    xgb_research.add_argument("--output-dir", default="data")
    ai_scan = subparsers.add_parser("ai-scan", help="Run current XGBoost + OpenAI advisory scan")
    ai_scan.add_argument("--symbols", default="config/swedish_large_caps.txt")
    ai_scan.add_argument("--output-dir", default="data")
    ai_scan.add_argument("--end", default=None, help="Optional end date, YYYY-MM-DD. Defaults to 2026-01-01.")
    return parser


def _parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


if __name__ == "__main__":
    main()
