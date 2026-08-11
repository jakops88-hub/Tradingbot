"""Backtest orchestration for the core trading flow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle, OrderSide, Trade
from trading_bot.execution.broker import Broker
from trading_bot.metrics.performance import max_drawdown, profit_factor, total_return, win_rate
from trading_bot.portfolio.portfolio import Portfolio
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    starting_capital: Decimal
    ending_capital: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    max_drawdown: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    realized_pnl: Decimal
    profit_factor: Decimal | None
    equity_curve: list[Decimal]
    trade_log: list[Trade]

    @property
    def starting_cash(self) -> Decimal:
        return self.starting_capital

    @property
    def ending_equity(self) -> Decimal:
        return self.ending_capital

    @property
    def trades(self) -> int:
        return self.total_trades


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

        sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
        portfolio = Portfolio(cash=self.starting_cash)
        equity_curve: list[Decimal] = []
        trade_log: list[Trade] = []
        closed_trade_pnls: list[Decimal] = []

        for index in range(len(sorted_candles)):
            window = sorted_candles[: index + 1]
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
                position_before_trade = portfolio.positions.get(decision.order.symbol)
                average_price_before_trade = (
                    position_before_trade.average_price
                    if position_before_trade is not None
                    else Decimal("0")
                )
                trade = self.broker.submit_order(decision.order, latest.close)
                portfolio.apply_trade(trade)
                trade_log.append(trade)
                if trade.side == OrderSide.SELL:
                    closed_trade_pnls.append(
                        (trade.price - average_price_before_trade) * trade.quantity - trade.commission
                    )

            equity_curve.append(portfolio.equity(prices))

        ending_equity = equity_curve[-1]
        winning_trades = sum(1 for pnl in closed_trade_pnls if pnl > 0)
        losing_trades = sum(1 for pnl in closed_trade_pnls if pnl < 0)
        gross_profit = sum((pnl for pnl in closed_trade_pnls if pnl > 0), Decimal("0"))
        gross_loss = abs(sum((pnl for pnl in closed_trade_pnls if pnl < 0), Decimal("0")))
        return_value = total_return(equity_curve)
        return BacktestResult(
            starting_capital=self.starting_cash,
            ending_capital=ending_equity,
            total_return=return_value,
            total_return_pct=return_value * Decimal("100"),
            max_drawdown=max_drawdown(equity_curve),
            total_trades=len(trade_log),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate(winning_trades, losing_trades),
            realized_pnl=portfolio.realized_pnl,
            profit_factor=profit_factor(gross_profit, gross_loss),
            equity_curve=equity_curve,
            trade_log=trade_log,
        )
