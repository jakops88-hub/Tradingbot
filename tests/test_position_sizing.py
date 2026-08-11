from decimal import Decimal

from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.risk.position_sizing import (
    calculate_buy_quantity,
    calculate_risk_amount,
    calculate_risk_per_unit,
)


def test_risk_amount_calculation() -> None:
    assert calculate_risk_amount(Decimal("1000"), get_risk_profile(RiskMode.MEDIUM)) == Decimal("10.00")


def test_stop_distance_calculation() -> None:
    assert calculate_risk_per_unit(Decimal("100"), Decimal("95")) == Decimal("5")


def test_quantity_uses_risk_amount_divided_by_stop_distance() -> None:
    quantity = calculate_buy_quantity(
        cash=Decimal("1000"),
        total_equity=Decimal("1000"),
        current_exposure=Decimal("0"),
        price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        profile=get_risk_profile(RiskMode.MEDIUM),
    )

    assert quantity == Decimal("2.00000000")


def test_quantity_is_capped_by_max_exposure() -> None:
    quantity = calculate_buy_quantity(
        cash=Decimal("1000"),
        total_equity=Decimal("1000"),
        current_exposure=Decimal("0"),
        price=Decimal("100"),
        stop_loss_price=Decimal("99"),
        profile=get_risk_profile(RiskMode.LOW),
    )

    assert quantity == Decimal("3.00000000")


def test_quantity_is_capped_by_available_cash() -> None:
    quantity = calculate_buy_quantity(
        cash=Decimal("150"),
        total_equity=Decimal("1000"),
        current_exposure=Decimal("0"),
        price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        profile=get_risk_profile(RiskMode.HIGH),
    )

    assert quantity == Decimal("1.50000000")


def test_low_medium_high_profiles_scale_risk_based_quantity() -> None:
    quantities = [
        calculate_buy_quantity(
            cash=Decimal("10000"),
            total_equity=Decimal("1000"),
            current_exposure=Decimal("0"),
            price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            profile=get_risk_profile(mode),
        )
        for mode in [RiskMode.LOW, RiskMode.MEDIUM, RiskMode.HIGH]
    ]

    assert quantities == [Decimal("1.00000000"), Decimal("2.00000000"), Decimal("4.00000000")]


def test_quantity_never_uses_leverage() -> None:
    quantity = calculate_buy_quantity(
        cash=Decimal("50"),
        total_equity=Decimal("1000"),
        current_exposure=Decimal("0"),
        price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        profile=get_risk_profile(RiskMode.HIGH),
    )

    assert quantity == Decimal("0.50000000")
