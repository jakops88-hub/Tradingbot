"""Backtest metric calculations."""

from __future__ import annotations

from math import sqrt


def total_return(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return (equity_curve[-1] / equity_curve[0]) - 1


def max_drawdown(equity_curve: list[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, (equity / peak) - 1)
    return worst


def sharpe_ratio(equity_curve: list[float], periods_per_year: int = 252) -> float:
    if len(equity_curve) < 3:
        return 0.0
    returns = [
        (equity_curve[index] / equity_curve[index - 1]) - 1
        for index in range(1, len(equity_curve))
        if equity_curve[index - 1] != 0
    ]
    if len(returns) < 2:
        return 0.0
    average = sum(returns) / len(returns)
    variance = sum((value - average) ** 2 for value in returns) / (len(returns) - 1)
    if variance == 0:
        return 0.0
    return (average / sqrt(variance)) * sqrt(periods_per_year)
