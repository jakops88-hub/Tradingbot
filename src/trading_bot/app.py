"""Command-line entrypoint for offline historical backtesting."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
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


def main() -> None:
    engine = BacktestEngine(
        strategy=EMATrendStrategy(EMATrendConfig(fast_period=20, slow_period=50)),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(
            ExecutionCostConfig(
                percentage_fee=Decimal("0.001"),
                fixed_fee=Decimal("0.05"),
                slippage_percentage=Decimal("0.001"),
            )
        ),
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


if __name__ == "__main__":
    main()
