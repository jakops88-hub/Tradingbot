"""Deterministic technical indicators used by strategies."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal


def exponential_moving_average(values: Sequence[Decimal], period: int) -> list[Decimal]:
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return []

    multiplier = Decimal("2") / Decimal(period + 1)
    ema_values: list[Decimal] = [sum(values[:period], Decimal("0")) / Decimal(period)]
    for value in values[period:]:
        ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])

    return ema_values
