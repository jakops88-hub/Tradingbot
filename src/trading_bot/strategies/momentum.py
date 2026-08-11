"""Momentum strategy."""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.data.models import Bar, Signal, SignalAction
from trading_bot.strategies.base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, lookback: int = 10, threshold: float = 0.02) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signal(self, bars: Sequence[Bar]) -> Signal:
        if not self.require_bars(bars, self.lookback + 1):
            return self.hold_signal(bars, "not enough bars")

        latest = bars[-1]
        previous = bars[-self.lookback - 1]
        momentum = (latest.close - previous.close) / previous.close
        confidence = min(abs(momentum) / self.threshold, 1.0)

        if momentum > self.threshold:
            action = SignalAction.BUY
        elif momentum < -self.threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
            confidence = 0.0

        return Signal(
            symbol=latest.symbol,
            action=action,
            confidence=confidence,
            generated_at=latest.timestamp,
            reason=f"{self.lookback}-bar momentum {momentum:.2%}",
        )
