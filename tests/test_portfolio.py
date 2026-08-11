from datetime import datetime
from decimal import Decimal

import pytest

from trading_bot.data.models import OrderSide, Trade
from trading_bot.portfolio.portfolio import Portfolio


NOW = datetime(2024, 1, 1)


def test_portfolio_applies_buy_and_sell_trades() -> None:
    portfolio = Portfolio(cash=Decimal("1000"))

    portfolio.apply_trade(
        Trade(
            symbol="ABC",
            side=OrderSide.BUY,
            quantity=Decimal("2"),
            price=Decimal("100"),
            executed_at=NOW,
            commission=Decimal("1"),
        )
    )
    portfolio.apply_trade(
        Trade(
            symbol="ABC",
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            price=Decimal("120"),
            executed_at=NOW,
            commission=Decimal("1"),
        )
    )

    position = portfolio.positions["ABC"]
    assert portfolio.cash == Decimal("918")
    assert position.quantity == Decimal("1")
    assert position.average_price == Decimal("100.5")
    assert portfolio.realized_pnl == Decimal("18.5")


def test_portfolio_rejects_sell_larger_than_position() -> None:
    portfolio = Portfolio(cash=Decimal("1000"))

    with pytest.raises(ValueError, match="cannot sell"):
        portfolio.apply_trade(
            Trade(
                symbol="ABC",
                side=OrderSide.SELL,
                quantity=Decimal("1"),
                price=Decimal("100"),
                executed_at=NOW,
            )
        )


def test_portfolio_snapshot_reports_equity() -> None:
    portfolio = Portfolio(cash=Decimal("500"))
    portfolio.apply_trade(
        Trade(
            symbol="ABC",
            side=OrderSide.BUY,
            quantity=Decimal("2"),
            price=Decimal("100"),
            executed_at=NOW,
        )
    )

    snapshot = portfolio.snapshot({"ABC": Decimal("110")}, NOW)

    assert snapshot.cash == Decimal("300")
    assert snapshot.positions_value == Decimal("220")
    assert snapshot.total_equity == Decimal("520")
    assert snapshot.open_positions == 1
