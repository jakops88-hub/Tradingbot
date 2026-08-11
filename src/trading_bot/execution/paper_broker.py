"""Paper broker with immediate fills."""

from __future__ import annotations

from decimal import Decimal

from trading_bot.data.models import Order, OrderSide, Trade
from trading_bot.execution.broker import Broker
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.simulation import simulated_fill_price, stop_loss_from_entry


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

        fill_price = simulated_fill_price(order.side, market_price, self.cost_config.slippage_percentage)

        gross_value = order.quantity * fill_price
        percentage_fee = gross_value * self.cost_config.percentage_fee
        fixed_fee = self.cost_config.fixed_fee
        total_fee = percentage_fee + fixed_fee
        slippage_cost = abs(fill_price - market_price) * order.quantity
        stop_loss_price = order.stop_loss_price
        if order.side == OrderSide.BUY and order.stop_loss_pct is not None:
            stop_loss_price = stop_loss_from_entry(fill_price, order.stop_loss_pct)
        monetary_risk = Decimal("0")
        if order.side == OrderSide.BUY and stop_loss_price is not None:
            monetary_risk = abs(fill_price - stop_loss_price) * order.quantity

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
            stop_loss_price=stop_loss_price,
            monetary_risk=monetary_risk,
            exit_reason=order.exit_reason,
        )
