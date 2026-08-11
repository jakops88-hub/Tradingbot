from datetime import datetime, timedelta

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import get_risk_profile
from trading_bot.data.models import Bar
from trading_bot.strategies.momentum import MomentumStrategy


def make_bars(prices: list[float]) -> list[Bar]:
    start = datetime(2024, 1, 1)
    return [
        Bar(
            symbol="TEST",
            timestamp=start + timedelta(days=index),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000,
        )
        for index, price in enumerate(prices)
    ]


def test_backtest_runs_and_records_equity_curve() -> None:
    bars = make_bars([100, 101, 102, 105, 108, 110, 109])
    engine = BacktestEngine(
        strategy=MomentumStrategy(lookback=2, threshold=0.02),
        risk_profile=get_risk_profile("balanced"),
        starting_cash=10_000,
    )

    result = engine.run(bars)

    assert len(result.equity_curve) == len(bars)
    assert result.ending_equity > 0
    assert result.trades >= 1
