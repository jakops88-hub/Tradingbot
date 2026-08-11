"""Shared deterministic execution rules for paper trading and labels."""

from __future__ import annotations

from decimal import Decimal

from trading_bot.data.models import Candle, OrderSide


def simulated_fill_price(side: OrderSide, market_price: Decimal, slippage_percentage: Decimal) -> Decimal:
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    if slippage_percentage < 0:
        raise ValueError("slippage_percentage must be non-negative")
    if side == OrderSide.BUY:
        fill_price = market_price * (Decimal("1") + slippage_percentage)
    else:
        fill_price = market_price * (Decimal("1") - slippage_percentage)
    if fill_price <= 0:
        raise ValueError("slippage produced a non-positive fill price")
    return fill_price


def stop_loss_from_entry(entry_price: Decimal, stop_loss_pct: Decimal) -> Decimal:
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if not Decimal("0") < stop_loss_pct < Decimal("1"):
        raise ValueError("stop_loss_pct must be between 0 and 1")
    return entry_price * (Decimal("1") - stop_loss_pct)


def long_stop_market_price(candle: Candle, stop_loss_price: Decimal) -> Decimal | None:
    if stop_loss_price <= 0:
        raise ValueError("stop_loss_price must be positive")
    if candle.open < stop_loss_price:
        return candle.open
    if candle.low <= stop_loss_price:
        return stop_loss_price
    return None
