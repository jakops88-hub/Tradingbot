from datetime import datetime, timedelta

from trading_bot.data.models import Bar, SignalAction
from trading_bot.strategies.mean_reversion import MeanReversionStrategy
from trading_bot.strategies.momentum import MomentumStrategy
from trading_bot.strategies.trend import TrendStrategy


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


def test_momentum_emits_buy_signal() -> None:
    strategy = MomentumStrategy(lookback=2, threshold=0.02)

    signal = strategy.generate_signal(make_bars([100, 101, 105]))

    assert signal.action == SignalAction.BUY


def test_trend_emits_buy_signal_when_short_average_is_above_long_average() -> None:
    strategy = TrendStrategy(short_window=2, long_window=4)

    signal = strategy.generate_signal(make_bars([100, 100, 105, 110]))

    assert signal.action == SignalAction.BUY


def test_mean_reversion_emits_sell_signal_for_large_positive_zscore() -> None:
    strategy = MeanReversionStrategy(window=5, zscore_threshold=1.0)

    signal = strategy.generate_signal(make_bars([100, 100, 100, 100, 110]))

    assert signal.action == SignalAction.SELL
