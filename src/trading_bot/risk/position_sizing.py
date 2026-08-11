"""Position-sizing helpers."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

from trading_bot.config.risk_profiles import RiskProfile


def calculate_buy_quantity(
    *,
    cash: Decimal,
    total_equity: Decimal,
    current_exposure: Decimal,
    price: Decimal,
    stop_loss_price: Decimal,
    profile: RiskProfile,
) -> Decimal:
    if cash < 0:
        raise ValueError("cash cannot be negative")
    if total_equity <= 0:
        raise ValueError("total_equity must be positive")
    if current_exposure < 0:
        raise ValueError("current_exposure cannot be negative")
    if price <= 0:
        raise ValueError("price must be positive")
    if stop_loss_price <= 0:
        raise ValueError("stop_loss_price must be positive")

    risk_per_unit = abs(price - stop_loss_price)
    if risk_per_unit <= 0:
        raise ValueError("stop_loss_price must differ from price")

    risk_amount = total_equity * profile.risk_per_trade
    risk_limited_quantity = risk_amount / risk_per_unit
    exposure_limit = total_equity * profile.max_exposure
    remaining_exposure = max(exposure_limit - current_exposure, Decimal("0"))
    max_exposure_quantity = remaining_exposure / price
    max_cash_quantity = cash / price
    quantity = min(risk_limited_quantity, max_exposure_quantity, max_cash_quantity)
    if quantity <= 0:
        return Decimal("0")

    return quantity.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


def calculate_risk_amount(total_equity: Decimal, profile: RiskProfile) -> Decimal:
    if total_equity <= 0:
        raise ValueError("total_equity must be positive")
    return total_equity * profile.risk_per_trade


def calculate_risk_per_unit(entry_price: Decimal, stop_loss_price: Decimal) -> Decimal:
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if stop_loss_price <= 0:
        raise ValueError("stop_loss_price must be positive")
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit <= 0:
        raise ValueError("stop_loss_price must differ from entry_price")
    return risk_per_unit
