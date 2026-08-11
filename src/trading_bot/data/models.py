"""Shared domain models for market data, signals, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, close, and high")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")


@dataclass(frozen=True)
class Signal:
    symbol: str
    action: SignalAction
    confidence: float
    generated_at: datetime
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    created_at: datetime | None = None
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    filled_at: datetime
    commission: float = 0.0

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price

    @property
    def net_cash_effect(self) -> float:
        if self.side == OrderSide.BUY:
            return -(self.gross_value + self.commission)
        return self.gross_value - self.commission


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    average_price: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.average_price) * self.quantity
