from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.data.models import Candle, PortfolioSnapshot, SignalAction
from trading_bot.strategies.ema_trend import EMATrendConfig, EMATrendStrategy
from trading_bot.strategies.indicators import exponential_moving_average


START = datetime(2024, 1, 1)


def make_candles(prices: list[str]) -> list[Candle]:
    return [
        Candle(
            symbol="ABC",
            timestamp=START + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price) + Decimal("1"),
            low=max(Decimal(price) - Decimal("1"), Decimal("0.01")),
            close=Decimal(price),
            volume=Decimal("1000"),
        )
        for index, price in enumerate(prices)
    ]


def make_snapshot(candles: list[Candle]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        generated_at=candles[-1].timestamp,
        cash=Decimal("1000"),
        positions_value=Decimal("0"),
        total_equity=Decimal("1000"),
        open_positions=0,
    )


def signal_for(prices: list[str]) -> SignalAction:
    candles = make_candles(prices)
    strategy = EMATrendStrategy(EMATrendConfig(fast_period=2, slow_period=3))
    return strategy.generate_signal(candles, make_snapshot(candles)).action


def test_ema_calculation_uses_sma_seed_then_multiplier() -> None:
    values = [Decimal("10"), Decimal("11"), Decimal("12"), Decimal("13")]

    assert exponential_moving_average(values, period=3) == [Decimal("11"), Decimal("12.0")]


def test_invalid_ema_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="fast_period"):
        EMATrendConfig(fast_period=0, slow_period=3)
    with pytest.raises(ValueError, match="slow_period"):
        EMATrendConfig(fast_period=3, slow_period=3)


def test_warm_up_period_emits_hold() -> None:
    assert signal_for(["10", "9", "8"]) == SignalAction.HOLD


def test_bullish_crossover_produces_buy() -> None:
    assert signal_for(["10", "9", "8", "12"]) == SignalAction.BUY


def test_bearish_crossover_produces_sell() -> None:
    assert signal_for(["10", "11", "12", "8"]) == SignalAction.SELL


def test_no_repeated_buy_while_trend_remains_bullish() -> None:
    assert signal_for(["10", "9", "8", "12", "13"]) == SignalAction.HOLD


def test_no_repeated_sell_while_trend_remains_bearish() -> None:
    assert signal_for(["10", "11", "12", "8", "7"]) == SignalAction.HOLD


def test_ema_signal_uses_only_supplied_candles() -> None:
    prefix = make_candles(["10", "9", "8", "12"])
    with_future_data = make_candles(["10", "9", "8", "12", "1", "1"])
    strategy = EMATrendStrategy(EMATrendConfig(fast_period=2, slow_period=3))

    assert strategy.generate_signal(prefix, make_snapshot(prefix)).action == SignalAction.BUY
    assert strategy.generate_signal(with_future_data[:4], make_snapshot(with_future_data[:4])).action == SignalAction.BUY
