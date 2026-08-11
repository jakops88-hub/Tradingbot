"""Paper broker with immediate fills."""

from __future__ import annotations

from datetime import datetime

from trading_bot.data.models import Fill, Order
from trading_bot.execution.broker import Broker


class PaperBroker(Broker):
    def __init__(self, commission_per_trade: float = 0.0, slippage_bps: float = 0.0) -> None:
        if commission_per_trade < 0:
            raise ValueError("commission_per_trade must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps

    def submit_order(self, order: Order, price: float) -> Fill:
        if price <= 0:
            raise ValueError("price must be positive")
        slippage_multiplier = 1 + (self.slippage_bps / 10_000)
        fill_price = price * slippage_multiplier
        return Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            filled_at=order.created_at or datetime.utcnow(),
            commission=self.commission_per_trade,
        )
