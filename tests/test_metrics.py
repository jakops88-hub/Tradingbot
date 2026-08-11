from decimal import Decimal

from trading_bot.metrics.performance import max_drawdown


def test_max_drawdown_is_positive_loss_magnitude() -> None:
    assert max_drawdown([Decimal("1000"), Decimal("900"), Decimal("950")]) == Decimal("0.1")
