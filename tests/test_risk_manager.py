from trading_bot.config.risk_profiles import get_risk_profile
from trading_bot.data.models import Order, OrderSide
from trading_bot.risk.risk_manager import RiskManager


def test_rejects_order_that_exceeds_position_limit() -> None:
    manager = RiskManager(get_risk_profile("conservative"))
    order = Order(symbol="TEST", side=OrderSide.BUY, quantity=20)

    decision = manager.evaluate_order(
        order,
        price=100,
        cash=10_000,
        equity=10_000,
        current_position_value=0,
    )

    assert not decision.approved
    assert decision.reason == "max position size exceeded"


def test_approves_order_within_limits() -> None:
    manager = RiskManager(get_risk_profile("balanced"))
    order = Order(symbol="TEST", side=OrderSide.BUY, quantity=10)

    decision = manager.evaluate_order(
        order,
        price=100,
        cash=10_000,
        equity=10_000,
        current_position_value=0,
    )

    assert decision.approved
