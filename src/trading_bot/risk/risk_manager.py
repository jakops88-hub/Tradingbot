"""Risk checks for orders."""

from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Order, OrderSide


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str = ""


class RiskManager:
    def __init__(self, profile: RiskProfile) -> None:
        self.profile = profile

    def evaluate_order(
        self,
        order: Order,
        *,
        price: float,
        cash: float,
        equity: float,
        current_position_value: float = 0.0,
        realized_daily_pnl: float = 0.0,
    ) -> RiskDecision:
        if price <= 0:
            return RiskDecision(False, "price must be positive")
        if equity <= 0:
            return RiskDecision(False, "equity must be positive")
        if realized_daily_pnl <= -(equity * self.profile.max_daily_loss_fraction):
            return RiskDecision(False, "daily loss limit reached")

        order_value = order.quantity * price
        max_position_value = equity * self.profile.max_position_fraction

        if order.side == OrderSide.BUY:
            reserve_cash = equity * self.profile.min_cash_reserve_fraction
            if order_value > max(cash - reserve_cash, 0):
                return RiskDecision(False, "insufficient available cash")
            if current_position_value + order_value > max_position_value:
                return RiskDecision(False, "max position size exceeded")

        return RiskDecision(True, "approved")
