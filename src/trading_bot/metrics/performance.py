"""Performance metrics for tracking, not risk escalation."""

from __future__ import annotations

from decimal import Decimal


def total_return(equity_curve: list[Decimal]) -> Decimal:
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return Decimal("0")
    return (equity_curve[-1] / equity_curve[0]) - Decimal("1")


def max_drawdown(equity_curve: list[Decimal]) -> Decimal:
    peak = Decimal("0")
    worst = Decimal("0")
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            drawdown = (equity / peak) - Decimal("1")
            worst = min(worst, drawdown)
    return worst


def goal_progress(current_equity: Decimal, target_equity: Decimal) -> Decimal:
    if current_equity < 0:
        raise ValueError("current_equity cannot be negative")
    if target_equity <= 0:
        raise ValueError("target_equity must be positive")
    return min(current_equity / target_equity, Decimal("1"))
