"""Portfolio state and fill accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_bot.data.models import Fill, OrderSide, Position


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def position_for(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def apply_fill(self, fill: Fill) -> None:
        position = self.position_for(fill.symbol)
        self.cash += fill.net_cash_effect

        if fill.side == OrderSide.BUY:
            new_quantity = position.quantity + fill.quantity
            total_cost = position.average_price * position.quantity + fill.gross_value
            position.quantity = new_quantity
            position.average_price = total_cost / new_quantity
            return

        if fill.quantity > position.quantity:
            raise ValueError("cannot sell more than current position")

        self.realized_pnl += (fill.price - position.average_price) * fill.quantity - fill.commission
        position.quantity -= fill.quantity
        if position.quantity == 0:
            position.average_price = 0.0

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(
            position.market_value(prices.get(symbol, position.average_price))
            for symbol, position in self.positions.items()
        )

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)
