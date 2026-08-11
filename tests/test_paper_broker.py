from datetime import datetime
from decimal import Decimal

from trading_bot.data.models import Order, OrderSide
from trading_bot.execution.paper_broker import PaperBroker


def test_paper_broker_returns_simulated_buy_trade_with_costs() -> None:
    now = datetime(2024, 1, 1)
    broker = PaperBroker(
        commission_per_trade=Decimal("1"),
        slippage_bps=Decimal("10"),
    )
    order = Order(
        symbol="ABC",
        side=OrderSide.BUY,
        quantity=Decimal("2"),
        created_at=now,
    )

    trade = broker.submit_order(order, Decimal("100"))

    assert trade.price == Decimal("100.100")
    assert trade.gross_value == Decimal("200.200")
    assert trade.cash_effect == Decimal("-201.200")
    assert trade.executed_at == now


def test_paper_broker_applies_sell_slippage() -> None:
    now = datetime(2024, 1, 1)
    broker = PaperBroker(slippage_bps=Decimal("10"))
    order = Order(
        symbol="ABC",
        side=OrderSide.SELL,
        quantity=Decimal("2"),
        created_at=now,
    )

    trade = broker.submit_order(order, Decimal("100"))

    assert trade.price == Decimal("99.900")
    assert trade.cash_effect == Decimal("199.800")
