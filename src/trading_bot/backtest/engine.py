"""Backtest orchestration for the core trading flow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle, Order, OrderSide, Trade
from trading_bot.execution.broker import Broker
from trading_bot.metrics.performance import (
    buy_and_hold_return,
    max_drawdown,
    profit_factor,
    win_rate,
)
from trading_bot.portfolio.portfolio import Portfolio
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    starting_capital: Decimal
    ending_capital: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    strategy_return_pct: Decimal
    benchmark_return_pct: Decimal
    difference_vs_benchmark_pct: Decimal
    max_drawdown: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    open_positions: int
    positions_value: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    total_fees_paid: Decimal
    total_execution_costs: Decimal
    profit_factor: Decimal | None
    average_position_value: Decimal
    average_portfolio_exposure_pct: Decimal
    largest_position_value: Decimal
    maximum_portfolio_exposure_pct: Decimal
    stop_loss_exits: int
    average_monetary_risk_at_entry: Decimal
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
        close_open_positions: bool = False,
    ) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.strategy = strategy
        self.risk_manager = RiskManager(risk_profile)
        self.starting_cash = starting_cash
        self.broker = broker
        self.close_open_positions = close_open_positions

    def run(self, candles: list[Candle]) -> BacktestResult:
        if not candles:
            raise ValueError("candles cannot be empty")

        sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
        portfolio = Portfolio(cash=self.starting_cash)
        equity_curve: list[Decimal] = []
        trade_log: list[Trade] = []
        closed_trade_pnls: list[Decimal] = []
        position_values_at_entry: list[Decimal] = []
        exposure_pcts_at_entry: list[Decimal] = []
        monetary_risks_at_entry: list[Decimal] = []

        for index in range(len(sorted_candles)):
            window = sorted_candles[: index + 1]
            latest = window[-1]
            prices = {latest.symbol: latest.close}
            self._execute_stop_losses(
                portfolio=portfolio,
                candle=latest,
                trade_log=trade_log,
                closed_trade_pnls=closed_trade_pnls,
            )
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
                elif trade.side == OrderSide.BUY:
                    entry_prices = {trade.symbol: trade.price}
                    entry_equity = portfolio.equity(entry_prices)
                    position_value = trade.gross_value
                    exposure_pct = position_value / entry_equity if entry_equity > 0 else Decimal("0")
                    position_values_at_entry.append(position_value)
                    exposure_pcts_at_entry.append(exposure_pct)
                    monetary_risks_at_entry.append(trade.monetary_risk)

            equity_curve.append(portfolio.equity(prices))

        final_candle = sorted_candles[-1]
        final_prices = {final_candle.symbol: final_candle.close}
        if self.close_open_positions:
            for symbol, position in list(portfolio.positions.items()):
                if position.quantity <= 0:
                    continue
                average_price_before_trade = position.average_price
                close_order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    created_at=final_candle.timestamp,
                )
                trade = self.broker.submit_order(close_order, final_prices[symbol])
                portfolio.apply_trade(trade)
                trade_log.append(trade)
                closed_trade_pnls.append(
                    (trade.price - average_price_before_trade) * trade.quantity - trade.commission
                )
            equity_curve[-1] = portfolio.equity(final_prices)

        final_snapshot = portfolio.snapshot(final_prices, final_candle.timestamp)
        unrealized_pnl = sum(
            (
                position.unrealized_pnl(final_prices.get(symbol, position.average_price))
                for symbol, position in portfolio.positions.items()
                if position.quantity > 0
            ),
            Decimal("0"),
        )
        ending_equity = equity_curve[-1]
        net_pnl = ending_equity - self.starting_cash
        total_fees_paid = sum((trade.commission for trade in trade_log), Decimal("0"))
        total_execution_costs = sum(
            (trade.commission + trade.slippage_cost for trade in trade_log),
            Decimal("0"),
        )
        gross_pnl = net_pnl + total_execution_costs
        winning_trades = sum(1 for pnl in closed_trade_pnls if pnl > 0)
        losing_trades = sum(1 for pnl in closed_trade_pnls if pnl < 0)
        gross_profit = sum((pnl for pnl in closed_trade_pnls if pnl > 0), Decimal("0"))
        gross_loss = abs(sum((pnl for pnl in closed_trade_pnls if pnl < 0), Decimal("0")))
        return_value = (ending_equity / self.starting_cash) - Decimal("1")
        strategy_return_pct = return_value * Decimal("100")
        benchmark_return_pct = (
            buy_and_hold_return(sorted_candles[0].close, sorted_candles[-1].close)
            * Decimal("100")
        )
        stop_loss_exits = sum(1 for trade in trade_log if trade.exit_reason == "stop_loss")
        return BacktestResult(
            starting_capital=self.starting_cash,
            ending_capital=ending_equity,
            total_return=return_value,
            total_return_pct=strategy_return_pct,
            strategy_return_pct=strategy_return_pct,
            benchmark_return_pct=benchmark_return_pct,
            difference_vs_benchmark_pct=strategy_return_pct - benchmark_return_pct,
            max_drawdown=max_drawdown(equity_curve),
            total_trades=len(trade_log),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate(winning_trades, losing_trades),
            realized_pnl=portfolio.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            open_positions=final_snapshot.open_positions,
            positions_value=final_snapshot.positions_value,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_fees_paid=total_fees_paid,
            total_execution_costs=total_execution_costs,
            profit_factor=profit_factor(gross_profit, gross_loss),
            average_position_value=_average(position_values_at_entry),
            average_portfolio_exposure_pct=_average(exposure_pcts_at_entry) * Decimal("100"),
            largest_position_value=max(position_values_at_entry, default=Decimal("0")),
            maximum_portfolio_exposure_pct=max(exposure_pcts_at_entry, default=Decimal("0")) * Decimal("100"),
            stop_loss_exits=stop_loss_exits,
            average_monetary_risk_at_entry=_average(monetary_risks_at_entry),
            equity_curve=equity_curve,
            trade_log=trade_log,
        )

    def _execute_stop_losses(
        self,
        *,
        portfolio: Portfolio,
        candle: Candle,
        trade_log: list[Trade],
        closed_trade_pnls: list[Decimal],
    ) -> None:
        for symbol, position in list(portfolio.positions.items()):
            if symbol != candle.symbol or position.quantity <= 0 or position.stop_loss_price is None:
                continue
            if candle.low > position.stop_loss_price:
                continue
            average_price_before_trade = position.average_price
            stop_order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                created_at=candle.timestamp,
                exit_reason="stop_loss",
            )
            trade = self.broker.submit_order(stop_order, position.stop_loss_price)
            portfolio.apply_trade(trade)
            trade_log.append(trade)
            closed_trade_pnls.append(
                (trade.price - average_price_before_trade) * trade.quantity - trade.commission
            )


def _average(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, Decimal("0")) / Decimal(len(values))
