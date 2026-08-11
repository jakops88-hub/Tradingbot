"""Base types for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from trading_bot.data.models import Bar, Signal, SignalAction


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, bars: Sequence[Bar]) -> Signal:
        """Return a trading signal for the provided bars."""

    def hold_signal(self, bars: Sequence[Bar], reason: str) -> Signal:
        symbol = bars[-1].symbol if bars else ""
        generated_at = bars[-1].timestamp if bars else datetime.utcnow()
        return Signal(
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            generated_at=generated_at,
            reason=reason,
        )

    def require_bars(self, bars: Sequence[Bar], count: int) -> bool:
        return len(bars) >= count
