"""Mean-reversion strategy."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, pstdev

from trading_bot.data.models import Bar, Signal, SignalAction
from trading_bot.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self, window: int = 20, zscore_threshold: float = 2.0) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if zscore_threshold <= 0:
            raise ValueError("zscore_threshold must be positive")
        self.window = window
        self.zscore_threshold = zscore_threshold

    def generate_signal(self, bars: Sequence[Bar]) -> Signal:
        if not self.require_bars(bars, self.window):
            return self.hold_signal(bars, "not enough bars")

        latest = bars[-1]
        closes = [bar.close for bar in bars[-self.window :]]
        average = mean(closes)
        deviation = pstdev(closes)
        if deviation == 0:
            return self.hold_signal(bars, "zero price deviation")

        zscore = (latest.close - average) / deviation
        confidence = min(abs(zscore) / self.zscore_threshold, 1.0)

        if zscore <= -self.zscore_threshold:
            action = SignalAction.BUY
        elif zscore >= self.zscore_threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
            confidence = 0.0

        return Signal(
            symbol=latest.symbol,
            action=action,
            confidence=confidence,
            generated_at=latest.timestamp,
            reason=f"{self.window}-bar z-score {zscore:.2f}",
        )
