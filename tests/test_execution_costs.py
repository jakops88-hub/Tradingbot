from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle, Order, OrderSide, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.portfolio.portfolio import Portfolio
from trading_bot.strategies.base import Strategy


NOW = datetime(2024, 1, 1)


class BuyThenSellStrategy(Strategy):
    name = "buy_then_sell"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        if len(candles) == 1:
            action = SignalAction.BUY
        elif len(candles) == 2:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
        return Signal(candles[-1].symbol, action, candles[-1].timestamp)


def make_order(side: OrderSide) -> Order:
    return Order(
        symbol="ABC",
        side=side,
        quantity=Decimal("2"),
        created_at=NOW,
    )


def make_candles(prices: list[str]) -> list[Candle]:
    return [
        Candle(
            symbol="ABC",
            timestamp=NOW + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price) + Decimal("1"),
            low=Decimal(price) - Decimal("1"),
            close=Decimal(price),
            volume=Decimal("1000"),
        )
        for index, price in enumerate(prices)
    ]


def test_zero_cost_paper_broker_preserves_existing_execution_behavior() -> None:
    broker = PaperBroker()

    trade = broker.submit_order(make_order(OrderSide.BUY), Decimal("100"))

    assert trade.price == Decimal("100")
    assert trade.commission == Decimal("0")
    assert trade.slippage_cost == Decimal("0")
    assert trade.cash_effect == Decimal("-200")


def test_percentage_fee_is_recorded_on_trade() -> None:
    broker = PaperBroker(ExecutionCostConfig(percentage_fee=Decimal("0.01")))

    trade = broker.submit_order(make_order(OrderSide.BUY), Decimal("100"))

    assert trade.percentage_fee == Decimal("2.00")
    assert trade.fixed_fee == Decimal("0")
    assert trade.commission == Decimal("2.00")
    assert trade.cash_effect == Decimal("-202.00")


def test_fixed_fee_is_recorded_on_trade() -> None:
    broker = PaperBroker(ExecutionCostConfig(fixed_fee=Decimal("1.25")))

    trade = broker.submit_order(make_order(OrderSide.BUY), Decimal("100"))

    assert trade.fixed_fee == Decimal("1.25")
    assert trade.commission == Decimal("1.25")
    assert trade.cash_effect == Decimal("-201.25")


def test_buy_slippage_increases_execution_price() -> None:
    broker = PaperBroker(ExecutionCostConfig(slippage_percentage=Decimal("0.01")))

    trade = broker.submit_order(make_order(OrderSide.BUY), Decimal("100"))

    assert trade.price == Decimal("101.00")
    assert trade.slippage_cost == Decimal("2.00")


def test_sell_slippage_decreases_execution_price() -> None:
    broker = PaperBroker(ExecutionCostConfig(slippage_percentage=Decimal("0.01")))

    trade = broker.submit_order(make_order(OrderSide.SELL), Decimal("100"))

    assert trade.price == Decimal("99.00")
    assert trade.slippage_cost == Decimal("2.00")


def test_portfolio_accounting_includes_buy_and_sell_costs() -> None:
    broker = PaperBroker(
        ExecutionCostConfig(
            percentage_fee=Decimal("0.01"),
            fixed_fee=Decimal("1"),
            slippage_percentage=Decimal("0.01"),
        )
    )
    portfolio = Portfolio(cash=Decimal("1000"))
    buy = broker.submit_order(make_order(OrderSide.BUY), Decimal("100"))
    sell = broker.submit_order(make_order(OrderSide.SELL), Decimal("110"))

    portfolio.apply_trade(buy)
    portfolio.apply_trade(sell)

    assert portfolio.cash == Decimal("1009.6020")
    assert portfolio.realized_pnl == Decimal("9.6020")
    assert portfolio.positions["ABC"].quantity == Decimal("0")


def test_backtest_reports_gross_vs_net_pnl_and_benchmark() -> None:
    engine = BacktestEngine(
        strategy=BuyThenSellStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(
            ExecutionCostConfig(
                percentage_fee=Decimal("0.01"),
                fixed_fee=Decimal("1"),
                slippage_percentage=Decimal("0.01"),
            )
        ),
        starting_cash=Decimal("1000"),
    )

    result = engine.run(make_candles(["100", "110"]))

    assert result.total_trades == 2
    assert result.total_fees_paid == Decimal("2.209900000000")
    assert result.total_execution_costs == Decimal("2.419900000000")
    assert result.net_pnl == Decimal("-1.419900000000")
    assert result.gross_pnl == Decimal("1.000000000000")
    assert result.realized_pnl == Decimal("-1.419900000000")
    assert result.benchmark_return_pct == Decimal("10.0")
    assert result.strategy_return_pct == Decimal("-0.141990000000")
    assert result.difference_vs_benchmark_pct == Decimal("-10.141990000000")
