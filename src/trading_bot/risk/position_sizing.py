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

    exposure_limit = total_equity * profile.max_exposure
    remaining_exposure = max(exposure_limit - current_exposure, Decimal("0"))
    trade_budget = min(total_equity * profile.risk_per_trade, remaining_exposure, cash)
    if trade_budget <= 0:
        return Decimal("0")

    return (trade_budget / price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
