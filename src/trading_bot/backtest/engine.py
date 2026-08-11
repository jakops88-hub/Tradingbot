"""Backtest orchestration for the core trading flow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle, Trade
from trading_bot.execution.broker import Broker
from trading_bot.metrics.performance import max_drawdown, total_return
from trading_bot.portfolio.portfolio import Portfolio
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    starting_cash: Decimal
    ending_equity: Decimal
    total_return: Decimal
    max_drawdown: Decimal
    trades: int
    equity_curve: list[Decimal]
    trade_log: list[Trade]


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk_profile: RiskProfile,
        broker: Broker,
        starting_cash: Decimal,
    ) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.strategy = strategy
        self.risk_manager = RiskManager(risk_profile)
        self.starting_cash = starting_cash
        self.broker = broker

    def run(self, candles: list[Candle]) -> BacktestResult:
        if not candles:
            raise ValueError("candles cannot be empty")

        portfolio = Portfolio(cash=self.starting_cash)
        equity_curve: list[Decimal] = []
        trade_log: list[Trade] = []

        for index in range(len(candles)):
            window = candles[: index + 1]
            latest = window[-1]
            prices = {latest.symbol: latest.close}
            snapshot = portfolio.snapshot(prices, latest.timestamp)
            signal = self.strategy.generate_signal(window, snapshot)
            decision = self.risk_manager.evaluate_signal(
                signal,
                snapshot=snapshot,
                positions=portfolio.positions,
                current_price=latest.close,
                starting_equity=self.starting_cash,
            )

            if decision.approved and decision.order is not None:
                trade = self.broker.submit_order(decision.order, latest.close)
                portfolio.apply_trade(trade)
                trade_log.append(trade)

            equity_curve.append(portfolio.equity(prices))

        ending_equity = equity_curve[-1]
        return BacktestResult(
            starting_cash=self.starting_cash,
            ending_equity=ending_equity,
            total_return=total_return(equity_curve),
            max_drawdown=max_drawdown(equity_curve),
            trades=len(trade_log),
            equity_curve=equity_curve,
            trade_log=trade_log,
        )
