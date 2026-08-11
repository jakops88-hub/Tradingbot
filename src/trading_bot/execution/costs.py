"""Execution cost configuration for offline backtests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ExecutionCostConfig:
    percentage_fee: Decimal = Decimal("0")
    fixed_fee: Decimal = Decimal("0")
    slippage_percentage: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        values = {
            "percentage_fee": self.percentage_fee,
            "fixed_fee": self.fixed_fee,
            "slippage_percentage": self.slippage_percentage,
        }
        for field_name, value in values.items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.slippage_percentage >= Decimal("1"):
            raise ValueError("slippage_percentage must be less than 1")
