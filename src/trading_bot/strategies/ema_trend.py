"""Long-only EMA crossover trend-following strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.strategies.base import Strategy
from trading_bot.strategies.indicators import exponential_moving_average


@dataclass(frozen=True)
class EMATrendConfig:
    fast_period: int = 20
    slow_period: int = 50
    stop_loss_pct: Decimal = Decimal("0.05")

    def __post_init__(self) -> None:
        if self.fast_period <= 0:
            raise ValueError("fast_period must be positive")
        if self.slow_period <= 0:
            raise ValueError("slow_period must be positive")
        if self.slow_period <= self.fast_period:
            raise ValueError("slow_period must be greater than fast_period")
        if not Decimal("0") < self.stop_loss_pct < Decimal("1"):
            raise ValueError("stop_loss_pct must be between 0 and 1")


class EMATrendStrategy(Strategy):
    name = "ema_trend"

    def __init__(self, config: EMATrendConfig | None = None) -> None:
        self.config = config or EMATrendConfig()

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        latest = candles[-1]
        if len(candles) < self.config.slow_period + 1:
            return Signal(
                symbol=latest.symbol,
                action=SignalAction.HOLD,
                generated_at=latest.timestamp,
                reason="EMA warm-up",
            )

        closes = [candle.close for candle in candles]
        fast_values = exponential_moving_average(closes, self.config.fast_period)
        slow_values = exponential_moving_average(closes, self.config.slow_period)

        previous_fast = _aligned_ema_at(fast_values, self.config.fast_period, len(closes) - 2)
        current_fast = _aligned_ema_at(fast_values, self.config.fast_period, len(closes) - 1)
        previous_slow = _aligned_ema_at(slow_values, self.config.slow_period, len(closes) - 2)
        current_slow = _aligned_ema_at(slow_values, self.config.slow_period, len(closes) - 1)

        action = SignalAction.HOLD
        reason = "no EMA crossover"
        if previous_fast <= previous_slow and current_fast > current_slow:
            action = SignalAction.BUY
            reason = "fast EMA crossed above slow EMA"
        elif previous_fast >= previous_slow and current_fast < current_slow:
            action = SignalAction.SELL
            reason = "fast EMA crossed below slow EMA"

        return Signal(
            symbol=latest.symbol,
            action=action,
            generated_at=latest.timestamp,
            confidence=_ema_confidence(current_fast, current_slow),
            reason=reason,
            stop_loss_price=_initial_stop_loss(latest.close, self.config.stop_loss_pct)
            if action == SignalAction.BUY
            else None,
        )


def _aligned_ema_at(ema_values: Sequence[Decimal], period: int, price_index: int) -> Decimal:
    ema_index = price_index - period + 1
    if ema_index < 0:
        raise ValueError("EMA value is not available for requested price index")
    return ema_values[ema_index]


def _ema_confidence(fast_ema: Decimal, slow_ema: Decimal) -> Decimal:
    if slow_ema == 0:
        return Decimal("0")
    spread = abs((fast_ema - slow_ema) / slow_ema)
    return min(spread, Decimal("1"))


def _initial_stop_loss(entry_price: Decimal, stop_loss_pct: Decimal) -> Decimal:
    return entry_price * (Decimal("1") - stop_loss_pct)
