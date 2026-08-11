"""Command-line entrypoint for offline historical backtesting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.market_data import HistoricalDataProvider
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.strategies.base import Strategy


class DemoThresholdStrategy(Strategy):
    name = "demo_threshold"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        latest = candles[-1]
        if snapshot.open_positions == 0 and latest.close == Decimal("100"):
            action = SignalAction.BUY
        elif snapshot.open_positions > 0 and latest.close in {Decimal("110"), Decimal("90")}:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        return Signal(
            symbol=latest.symbol,
            action=action,
            generated_at=latest.timestamp,
            reason="demo threshold",
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "tests" / "fixtures" / "sample_ohlcv.csv"
    provider = HistoricalDataProvider(data_path)
    candles = list(
        provider.historical_candles(
            "ABC",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )
    )
    engine = BacktestEngine(
        strategy=DemoThresholdStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(
            ExecutionCostConfig(
                percentage_fee=Decimal("0.001"),
                fixed_fee=Decimal("0.05"),
                slippage_percentage=Decimal("0.001"),
            )
        ),
        starting_cash=Decimal("1000"),
    )
    result = engine.run(candles)

    print("TradingBot offline backtest demo")
    print("Data: tests/fixtures/sample_ohlcv.csv")
    print("Starting capital: 1000 SEK")
    print(f"Ending capital: {result.ending_capital} SEK")
    print(f"Strategy return: {result.strategy_return_pct}%")
    print(f"Buy & hold return: {result.benchmark_return_pct}%")
    print(f"Difference vs benchmark: {result.difference_vs_benchmark_pct}%")
    print(f"Total trades: {result.total_trades}")
    print(f"Win rate: {result.win_rate * Decimal('100')}%")
    print(f"Gross PnL: {result.gross_pnl} SEK")
    print(f"Fees/costs: {result.total_execution_costs} SEK")
    print(f"Net PnL: {result.net_pnl} SEK")
    print(f"Max drawdown: {result.max_drawdown * Decimal('100')}%")
    print(f"Profit factor: {result.profit_factor}")
    print("Real-money trading: unavailable")


if __name__ == "__main__":
    main()
