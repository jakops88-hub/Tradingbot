from datetime import datetime
from decimal import Decimal

from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import PortfolioSnapshot, Position, Signal, SignalAction
from trading_bot.risk.risk_manager import RiskManager


NOW = datetime(2024, 1, 1)


def make_snapshot(
    *,
    cash: Decimal = Decimal("1000"),
    positions_value: Decimal = Decimal("0"),
    open_positions: int = 0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        generated_at=NOW,
        cash=cash,
        positions_value=positions_value,
        total_equity=cash + positions_value,
        open_positions=open_positions,
    )


def test_hold_signal_does_not_create_order() -> None:
    manager = RiskManager(get_risk_profile(RiskMode.LOW))

    decision = manager.evaluate_signal(
        Signal("ABC", SignalAction.HOLD, NOW),
        snapshot=make_snapshot(),
        positions={},
        current_price=Decimal("100"),
        starting_equity=Decimal("1000"),
    )

    assert decision.approved
    assert decision.order is None


def test_buy_signal_creates_order_with_profile_limited_size() -> None:
    manager = RiskManager(get_risk_profile(RiskMode.MEDIUM))

    decision = manager.evaluate_signal(
        Signal("ABC", SignalAction.BUY, NOW, stop_loss_price=Decimal("95")),
        snapshot=make_snapshot(),
        positions={},
        current_price=Decimal("100"),
        starting_equity=Decimal("1000"),
    )

    assert decision.approved
    assert decision.order is not None
    assert decision.order.quantity == Decimal("2.00000000")
    assert decision.order.stop_loss_price == Decimal("95")


def test_buy_signal_rejects_when_max_open_positions_reached() -> None:
    manager = RiskManager(get_risk_profile(RiskMode.LOW))

    decision = manager.evaluate_signal(
        Signal("ABC", SignalAction.BUY, NOW, stop_loss_price=Decimal("95")),
        snapshot=make_snapshot(open_positions=2),
        positions={},
        current_price=Decimal("100"),
        starting_equity=Decimal("1000"),
    )

    assert not decision.approved
    assert decision.order is None


def test_buy_signal_requires_stop_loss_price() -> None:
    manager = RiskManager(get_risk_profile(RiskMode.MEDIUM))

    decision = manager.evaluate_signal(
        Signal("ABC", SignalAction.BUY, NOW),
        snapshot=make_snapshot(),
        positions={},
        current_price=Decimal("100"),
        starting_equity=Decimal("1000"),
    )

    assert not decision.approved
    assert decision.reason == "buy signals require stop_loss_price"


def test_sell_signal_creates_order_from_existing_position() -> None:
    manager = RiskManager(get_risk_profile(RiskMode.LOW))

    decision = manager.evaluate_signal(
        Signal("ABC", SignalAction.SELL, NOW),
        snapshot=make_snapshot(positions_value=Decimal("200")),
        positions={"ABC": Position("ABC", quantity=Decimal("2"), average_price=Decimal("100"))},
        current_price=Decimal("100"),
        starting_equity=Decimal("1000"),
    )

    assert decision.approved
    assert decision.order is not None
    assert decision.order.quantity == Decimal("2")


def test_drawdown_limit_blocks_new_orders() -> None:
    manager = RiskManager(get_risk_profile(RiskMode.LOW))

    decision = manager.evaluate_signal(
        Signal("ABC", SignalAction.BUY, NOW, stop_loss_price=Decimal("95")),
        snapshot=make_snapshot(cash=Decimal("910")),
        positions={},
        current_price=Decimal("100"),
        starting_equity=Decimal("1000"),
    )

    assert not decision.approved
    assert decision.reason == "max drawdown reached"
