"""Moving-average trend strategy."""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.data.models import Bar, Signal, SignalAction
from trading_bot.strategies.base import Strategy


def simple_moving_average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    return sum(values) / len(values)


class TrendStrategy(Strategy):
    name = "trend"

    def __init__(self, short_window: int = 10, long_window: int = 30) -> None:
        if short_window < 1:
            raise ValueError("short_window must be >= 1")
        if long_window <= short_window:
            raise ValueError("long_window must be greater than short_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, bars: Sequence[Bar]) -> Signal:
        if not self.require_bars(bars, self.long_window):
            return self.hold_signal(bars, "not enough bars")

        latest = bars[-1]
        closes = [bar.close for bar in bars]
        short_average = simple_moving_average(closes[-self.short_window :])
        long_average = simple_moving_average(closes[-self.long_window :])
        spread = (short_average - long_average) / long_average
        confidence = min(abs(spread) / 0.05, 1.0)

        if short_average > long_average:
            action = SignalAction.BUY
        elif short_average < long_average:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
            confidence = 0.0

        return Signal(
            symbol=latest.symbol,
            action=action,
            confidence=confidence,
            generated_at=latest.timestamp,
            reason=f"SMA({self.short_window})={short_average:.2f}, SMA({self.long_window})={long_average:.2f}",
        )
