"""Paper broker with immediate fills."""

from __future__ import annotations

from decimal import Decimal

from trading_bot.data.models import Order, OrderSide, Trade
from trading_bot.execution.broker import Broker
from trading_bot.execution.costs import ExecutionCostConfig


class PaperBroker(Broker):
    def __init__(
        self,
        cost_config: ExecutionCostConfig | None = None,
        commission_per_trade: Decimal = Decimal("0"),
        slippage_bps: Decimal = Decimal("0"),
    ) -> None:
        if commission_per_trade < 0:
            raise ValueError("commission_per_trade must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if cost_config is not None and (commission_per_trade != 0 or slippage_bps != 0):
            raise ValueError("Use either cost_config or legacy commission/slippage arguments")

        self.cost_config = cost_config or ExecutionCostConfig(
            fixed_fee=commission_per_trade,
            slippage_percentage=slippage_bps / Decimal("10000"),
        )

    def submit_order(self, order: Order, market_price: Decimal) -> Trade:
        if market_price <= 0:
            raise ValueError("market_price must be positive")

        slippage = self.cost_config.slippage_percentage
        if order.side == OrderSide.BUY:
            fill_price = market_price * (Decimal("1") + slippage)
        else:
            fill_price = market_price * (Decimal("1") - slippage)

        if fill_price <= 0:
            raise ValueError("slippage produced a non-positive fill price")

        gross_value = order.quantity * fill_price
        percentage_fee = gross_value * self.cost_config.percentage_fee
        fixed_fee = self.cost_config.fixed_fee
        total_fee = percentage_fee + fixed_fee
        slippage_cost = abs(fill_price - market_price) * order.quantity

        return Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            executed_at=order.created_at,
            commission=total_fee,
            market_price=market_price,
            percentage_fee=percentage_fee,
            fixed_fee=fixed_fee,
            slippage_cost=slippage_cost,
        )
