"""Risk manager that converts approved signals into orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Order, OrderSide, PortfolioSnapshot, Position, Signal, SignalAction
from trading_bot.risk.position_sizing import calculate_buy_quantity


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str = ""
    order: Order | None = None


class RiskManager:
    def __init__(self, profile: RiskProfile) -> None:
        self.profile = profile

    def evaluate_signal(
        self,
        signal: Signal,
        *,
        snapshot: PortfolioSnapshot,
        positions: dict[str, Position],
        current_price: Decimal,
        starting_equity: Decimal,
    ) -> RiskDecision:
        if current_price <= 0:
            return RiskDecision(False, "price must be positive")
        if starting_equity <= 0:
            return RiskDecision(False, "starting_equity must be positive")
        if snapshot.total_equity <= 0:
            return RiskDecision(False, "total equity must be positive")

        drawdown = Decimal("1") - (snapshot.total_equity / starting_equity)
        if drawdown >= self.profile.max_drawdown:
            return RiskDecision(False, "max drawdown reached")

        if signal.action == SignalAction.HOLD:
            return RiskDecision(True, "hold signal")

        if signal.action == SignalAction.SELL:
            position = positions.get(signal.symbol)
            if position is None or position.quantity <= 0:
                return RiskDecision(False, "no open position to sell")
            return RiskDecision(
                approved=True,
                reason="sell approved",
                order=Order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    created_at=signal.generated_at,
                ),
            )

        if signal.action == SignalAction.BUY:
            existing_position = positions.get(signal.symbol)
            if existing_position is None and snapshot.open_positions >= self.profile.max_open_positions:
                return RiskDecision(False, "max open positions reached")
            if signal.stop_loss_price is None:
                return RiskDecision(False, "buy signals require stop_loss_price")

            quantity = calculate_buy_quantity(
                cash=snapshot.cash,
                total_equity=snapshot.total_equity,
                current_exposure=snapshot.positions_value,
                price=current_price,
                stop_loss_price=signal.stop_loss_price,
                profile=self.profile,
            )
            if quantity <= 0:
                return RiskDecision(False, "insufficient cash or exposure capacity")
            if quantity * current_price > snapshot.cash:
                return RiskDecision(False, "order would require leverage")

            return RiskDecision(
                approved=True,
                reason="buy approved",
                order=Order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    created_at=signal.generated_at,
                    stop_loss_price=signal.stop_loss_price,
                    stop_loss_pct=signal.stop_loss_pct,
                ),
            )

        raise ValueError(f"Unsupported signal action: {signal.action}")
