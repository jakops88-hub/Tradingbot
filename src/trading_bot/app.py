"""Command-line entrypoint for the trading bot."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import get_risk_profile
from trading_bot.data.models import Bar
from trading_bot.strategies.momentum import MomentumStrategy


def demo_bars(symbol: str) -> list[Bar]:
    start = datetime(2024, 1, 1)
    prices = [100, 101, 102, 104, 106, 105, 107, 110, 112, 115, 117, 116, 119]
    return [
        Bar(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1_000_000,
        )
        for index, price in enumerate(prices)
    ]


def main() -> None:
    profile_name = os.getenv("TRADING_BOT_PROFILE", "balanced")
    symbol = os.getenv("TRADING_BOT_SYMBOL", "AAPL")
    starting_cash = float(os.getenv("TRADING_BOT_INITIAL_CASH", "100000"))

    engine = BacktestEngine(
        strategy=MomentumStrategy(lookback=3, threshold=0.02),
        risk_profile=get_risk_profile(profile_name),
        starting_cash=starting_cash,
    )
    result = engine.run(demo_bars(symbol))

    print(f"Strategy: momentum")
    print(f"Risk profile: {profile_name}")
    print(f"Ending equity: {result.ending_equity:.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Trades: {result.trades}")


if __name__ == "__main__":
    main()
