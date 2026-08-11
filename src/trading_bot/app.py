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
from trading_bot.research.evaluator import ResearchEvaluator, yearly_periods
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


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "research":
        run_research_command(args)
        return
    if args.command == "download":
        run_download_command(args)
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
    provider = YahooFinanceDataProvider(cache_dir=args.output_dir)
    csv_path = provider.download_to_csv(
        symbol=args.symbol,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    print(f"Downloaded normalized daily OHLCV data: {csv_path}")
    print(f"Metadata: {csv_path}.metadata.json")
    print("Use this file with the research command.")


def print_research_report(report: ResearchReport) -> None:
    print("TradingBot EMA 20/50 historical research")
    print("This is a historical simulation, not evidence of future profitability.")
    print("")
    print(f"{'Period':<12}{'Strategy':>12}{'Buy&Hold':>12}{'Difference':>14}{'Trades':>8}{'Max DD':>10}")
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


def _period_row(result: PeriodResult) -> str:
    return (
        f"{result.period.label:<12}"
        f"{_format_pct(result.strategy_return_pct):>12}"
        f"{_format_pct(result.benchmark_return_pct):>12}"
        f"{_format_pct(result.difference_vs_benchmark_pct):>14}"
        f"{result.total_trades:>8}"
        f"{_format_pct(result.max_drawdown * Decimal('100'), signed=False):>10}"
    )


def _format_pct(value: Decimal, *, signed: bool = True) -> str:
    sign = "+" if signed else ""
    return f"{value.quantize(Decimal('0.01')):{sign}}%"


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
    return parser


def _parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


if __name__ == "__main__":
    main()
