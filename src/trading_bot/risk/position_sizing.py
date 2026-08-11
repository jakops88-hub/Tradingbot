"""Position-sizing helpers."""

from __future__ import annotations

from trading_bot.config.risk_profiles import RiskProfile


def fixed_fraction_size(equity: float, price: float, profile: RiskProfile) -> int:
    if equity <= 0:
        return 0
    if price <= 0:
        raise ValueError("price must be positive")

    position_value = equity * profile.max_position_fraction
    return int(position_value // price)


def risk_based_size(equity: float, price: float, stop_price: float, profile: RiskProfile) -> int:
    if equity <= 0:
        return 0
    if price <= 0 or stop_price <= 0:
        raise ValueError("price and stop_price must be positive")

    risk_per_share = abs(price - stop_price)
    if risk_per_share == 0:
        return 0

    risk_budget = equity * profile.max_trade_risk_fraction
    max_by_risk = int(risk_budget // risk_per_share)
    max_by_fraction = fixed_fraction_size(equity, price, profile)
    return max(0, min(max_by_risk, max_by_fraction))
