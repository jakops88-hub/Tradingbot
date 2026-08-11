from datetime import datetime
from decimal import Decimal

import pytest

from trading_bot.data.models import (
    Candle,
    Order,
    OrderSide,
    PortfolioSnapshot,
    Signal,
    SignalAction,
    Trade,
)


NOW = datetime(2024, 1, 1)


def test_candle_requires_valid_ohlc_values() -> None:
    candle = Candle(
        symbol="ABC",
        timestamp=NOW,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("1000"),
    )

    assert candle.close == Decimal("105")


def test_candle_rejects_invalid_high() -> None:
    with pytest.raises(ValueError, match="high"):
        Candle(
            symbol="ABC",
            timestamp=NOW,
            open=Decimal("100"),
            high=Decimal("99"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )


def test_signals_support_buy_sell_and_hold() -> None:
    actions = {
        Signal("ABC", SignalAction.BUY, NOW).action,
        Signal("ABC", SignalAction.SELL, NOW).action,
        Signal("ABC", SignalAction.HOLD, NOW).action,
    }

    assert actions == {SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD}


def test_signal_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Signal("ABC", SignalAction.BUY, NOW, confidence=Decimal("1.1"))


def test_order_and_trade_use_decimal_accounting() -> None:
    order = Order(
        symbol="ABC",
        side=OrderSide.BUY,
        quantity=Decimal("2.5"),
        created_at=NOW,
    )
    trade = Trade(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=Decimal("100"),
        executed_at=NOW,
        commission=Decimal("1"),
    )

    assert trade.gross_value == Decimal("250.0")
    assert trade.cash_effect == Decimal("-251.0")


def test_portfolio_snapshot_requires_consistent_equity() -> None:
    with pytest.raises(ValueError, match="total_equity"):
        PortfolioSnapshot(
            generated_at=NOW,
            cash=Decimal("100"),
            positions_value=Decimal("50"),
            total_equity=Decimal("140"),
            open_positions=1,
        )
