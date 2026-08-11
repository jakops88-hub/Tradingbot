"""Performance metrics for tracking, not risk escalation."""

from __future__ import annotations

from decimal import Decimal


def total_return(equity_curve: list[Decimal]) -> Decimal:
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return Decimal("0")
    return (equity_curve[-1] / equity_curve[0]) - Decimal("1")


def buy_and_hold_return(start_price: Decimal, end_price: Decimal) -> Decimal:
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if end_price <= 0:
        raise ValueError("end_price must be positive")
    return (end_price / start_price) - Decimal("1")


def max_drawdown(equity_curve: list[Decimal]) -> Decimal:
    peak = Decimal("0")
    worst = Decimal("0")
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            drawdown = (equity / peak) - Decimal("1")
            worst = min(worst, drawdown)
    return worst


def win_rate(winning_trades: int, losing_trades: int) -> Decimal:
    closed_trades = winning_trades + losing_trades
    if closed_trades == 0:
        return Decimal("0")
    return Decimal(winning_trades) / Decimal(closed_trades)


def profit_factor(gross_profit: Decimal, gross_loss: Decimal) -> Decimal | None:
    if gross_loss < 0:
        raise ValueError("gross_loss must be non-negative")
    if gross_loss == 0:
        return None if gross_profit == 0 else Decimal("Infinity")
    return gross_profit / gross_loss


def goal_progress(current_equity: Decimal, target_equity: Decimal) -> Decimal:
    if current_equity < 0:
        raise ValueError("current_equity cannot be negative")
    if target_equity <= 0:
        raise ValueError("target_equity must be positive")
    return min(current_equity / target_equity, Decimal("1"))
