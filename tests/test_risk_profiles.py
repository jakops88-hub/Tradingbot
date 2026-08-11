from decimal import Decimal

import pytest

from trading_bot.config.risk_profiles import RiskMode, get_risk_profile


def test_low_risk_profile_matches_required_preset() -> None:
    profile = get_risk_profile(RiskMode.LOW)

    assert profile.risk_per_trade == Decimal("0.005")
    assert profile.max_exposure == Decimal("0.30")
    assert profile.max_drawdown == Decimal("0.08")
    assert profile.max_open_positions == 2
    assert not profile.leverage_allowed


def test_medium_risk_profile_matches_required_preset() -> None:
    profile = get_risk_profile("MEDIUM")

    assert profile.risk_per_trade == Decimal("0.01")
    assert profile.max_exposure == Decimal("0.60")
    assert profile.max_drawdown == Decimal("0.15")
    assert profile.max_open_positions == 4
    assert not profile.leverage_allowed


def test_high_risk_profile_matches_required_preset() -> None:
    profile = get_risk_profile("high")

    assert profile.risk_per_trade == Decimal("0.02")
    assert profile.max_exposure == Decimal("1.00")
    assert profile.max_drawdown == Decimal("0.25")
    assert profile.max_open_positions == 6
    assert not profile.leverage_allowed


def test_unknown_risk_profile_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown risk profile"):
        get_risk_profile("maximum")
