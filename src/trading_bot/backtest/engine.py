"""Simple backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_bot.backtest.metrics import max_drawdown, sharpe_ratio, total_return
from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Bar, Order, OrderSide, SignalAction
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.portfolio.portfolio import Portfolio
from trading_bot.risk.position_sizing import fixed_fraction_size
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    starting_cash: float
    ending_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: int
    equity_curve: list[float]


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk_profile: RiskProfile,
        starting_cash: float = 100_000.0,
        broker: PaperBroker | None = None,
    ) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.strategy = strategy
        self.risk_manager = RiskManager(risk_profile)
        self.risk_profile = risk_profile
        self.starting_cash = starting_cash
        self.broker = broker or PaperBroker()

    def run(self, bars: list[Bar]) -> BacktestResult:
        if not bars:
            raise ValueError("bars cannot be empty")

        portfolio = Portfolio(cash=self.starting_cash)
        equity_curve: list[float] = []
        trades = 0

        for index in range(len(bars)):
            window = bars[: index + 1]
            latest = window[-1]
            prices = {latest.symbol: latest.close}
            equity = portfolio.equity(prices)
            signal = self.strategy.generate_signal(window)

            if signal.action != SignalAction.HOLD:
                order = self._order_from_signal(signal.action, latest, portfolio, equity)
                if order is not None:
                    position = portfolio.position_for(latest.symbol)
                    decision = self.risk_manager.evaluate_order(
                        order,
                        price=latest.close,
                        cash=portfolio.cash,
                        equity=equity,
                        current_position_value=position.market_value(latest.close),
                        realized_daily_pnl=portfolio.realized_pnl,
                    )
                    if decision.approved:
                        fill = self.broker.submit_order(order, latest.close)
                        portfolio.apply_fill(fill)
                        trades += 1

            equity_curve.append(portfolio.equity(prices))

        ending_equity = equity_curve[-1]
        return BacktestResult(
            starting_cash=self.starting_cash,
            ending_equity=ending_equity,
            total_return=total_return(equity_curve),
            max_drawdown=max_drawdown(equity_curve),
            sharpe_ratio=sharpe_ratio(equity_curve),
            trades=trades,
            equity_curve=equity_curve,
        )

    def _order_from_signal(
        self,
        action: SignalAction,
        latest: Bar,
        portfolio: Portfolio,
        equity: float,
    ) -> Order | None:
        if action == SignalAction.BUY:
            quantity = fixed_fraction_size(equity, latest.close, self.risk_profile)
            if quantity <= 0:
                return None
            return Order(
                symbol=latest.symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                created_at=latest.timestamp,
            )

        if action == SignalAction.SELL:
            position = portfolio.position_for(latest.symbol)
            if position.quantity <= 0:
                return None
            return Order(
                symbol=latest.symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                created_at=latest.timestamp,
            )

        return None
