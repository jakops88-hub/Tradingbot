"""Portfolio state and fill accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from trading_bot.data.models import OrderSide, PortfolioSnapshot, Position, Trade


@dataclass
class Portfolio:
    cash: Decimal
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise ValueError("cash cannot be negative")

    def position_for(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def apply_trade(self, trade: Trade) -> None:
        position = self.position_for(trade.symbol)

        if trade.side == OrderSide.BUY:
            total_cost = position.average_price * position.quantity + trade.gross_value
            new_quantity = position.quantity + trade.quantity
            cash_after_trade = self.cash + trade.cash_effect
            if cash_after_trade < 0:
                raise ValueError("trade would make cash negative")
            self.cash = cash_after_trade
            position.quantity = new_quantity
            position.average_price = total_cost / new_quantity
            return

        if trade.quantity > position.quantity:
            raise ValueError("cannot sell more than current position")

        self.cash += trade.cash_effect
        self.realized_pnl += (trade.price - position.average_price) * trade.quantity - trade.commission
        position.quantity -= trade.quantity
        if position.quantity == 0:
            position.average_price = Decimal("0")

    def positions_value(self, prices: dict[str, Decimal]) -> Decimal:
        return sum(
            position.market_value(prices.get(symbol, position.average_price))
            for symbol, position in self.positions.items()
            if position.quantity > 0
        )

    def equity(self, prices: dict[str, Decimal]) -> Decimal:
        return self.cash + self.positions_value(prices)

    def snapshot(self, prices: dict[str, Decimal], generated_at: datetime) -> PortfolioSnapshot:
        positions_value = self.positions_value(prices)
        open_positions = sum(1 for position in self.positions.values() if position.quantity > 0)
        return PortfolioSnapshot(
            generated_at=generated_at,
            cash=self.cash,
            positions_value=positions_value,
            total_equity=self.cash + positions_value,
            open_positions=open_positions,
            realized_pnl=self.realized_pnl,
        )
