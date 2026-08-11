"""Typed domain models shared across the trading platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
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
class Candle:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, close, and high")
        if min(self.open, self.high, self.low, self.close) <= Decimal("0"):
            raise ValueError("prices must be positive")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")


@dataclass(frozen=True)
class Signal:
    symbol: str
    action: SignalAction
    generated_at: datetime
    confidence: Decimal = Decimal("0")
    reason: str = ""
    stop_loss_price: Decimal | None = None

    def __post_init__(self) -> None:
        if not Decimal("0") <= self.confidence <= Decimal("1"):
            raise ValueError("confidence must be between 0 and 1")
        if self.stop_loss_price is not None and self.stop_loss_price <= 0:
            raise ValueError("stop_loss_price must be positive")


@dataclass(frozen=True)
class Order:
    symbol: str
    side: OrderSide
    quantity: Decimal
    created_at: datetime
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    exit_reason: str = "signal"

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive")
        if self.stop_loss_price is not None and self.stop_loss_price <= 0:
            raise ValueError("stop_loss_price must be positive")


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    executed_at: datetime
    commission: Decimal = Decimal("0")
    market_price: Decimal | None = None
    percentage_fee: Decimal = Decimal("0")
    fixed_fee: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    stop_loss_price: Decimal | None = None
    monetary_risk: Decimal = Decimal("0")
    exit_reason: str = "signal"

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.commission < 0:
            raise ValueError("commission must be non-negative")
        if self.market_price is not None and self.market_price <= 0:
            raise ValueError("market_price must be positive")
        if self.percentage_fee < 0:
            raise ValueError("percentage_fee must be non-negative")
        if self.fixed_fee < 0:
            raise ValueError("fixed_fee must be non-negative")
        if self.slippage_cost < 0:
            raise ValueError("slippage_cost must be non-negative")
        if self.stop_loss_price is not None and self.stop_loss_price <= 0:
            raise ValueError("stop_loss_price must be positive")
        if self.monetary_risk < 0:
            raise ValueError("monetary_risk must be non-negative")

    @property
    def gross_value(self) -> Decimal:
        return self.quantity * self.price

    @property
    def cash_effect(self) -> Decimal:
        if self.side == OrderSide.BUY:
            return -(self.gross_value + self.commission)
        return self.gross_value - self.commission


@dataclass
class Position:
    symbol: str
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    stop_loss_price: Decimal | None = None

    def market_value(self, price: Decimal) -> Decimal:
        if price <= 0:
            raise ValueError("price must be positive")
        return self.quantity * price

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        if self.quantity == 0:
            return Decimal("0")
        return (price - self.average_price) * self.quantity


@dataclass(frozen=True)
class PortfolioSnapshot:
    generated_at: datetime
    cash: Decimal
    positions_value: Decimal
    total_equity: Decimal
    open_positions: int
    realized_pnl: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise ValueError("cash cannot be negative")
        if self.positions_value < 0:
            raise ValueError("positions_value cannot be negative")
        if self.total_equity != self.cash + self.positions_value:
            raise ValueError("total_equity must equal cash plus positions_value")
        if self.open_positions < 0:
            raise ValueError("open_positions cannot be negative")


Bar = Candle
Fill = Trade
