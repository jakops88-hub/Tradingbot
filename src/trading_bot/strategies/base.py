"""Strategy abstraction.

Strategies produce signals only. They must never submit orders or call brokers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from trading_bot.data.models import Candle, PortfolioSnapshot, Signal


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        """Return BUY, SELL, or HOLD for the supplied market and portfolio state."""
