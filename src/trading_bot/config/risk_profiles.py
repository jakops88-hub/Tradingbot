"""Risk profile presets used by the trading platform."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum


class RiskMode(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class RiskProfile:
    mode: RiskMode
    risk_per_trade: Decimal
    max_exposure: Decimal
    max_drawdown: Decimal
    max_open_positions: int
    leverage_allowed: bool = False

    def __post_init__(self) -> None:
        percentages = {
            "risk_per_trade": self.risk_per_trade,
            "max_exposure": self.max_exposure,
            "max_drawdown": self.max_drawdown,
        }
        for field_name, value in percentages.items():
            if not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{field_name} must be non-negative")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be positive")
        if self.leverage_allowed:
            raise ValueError("leverage is not allowed in foundation risk profiles")


RISK_PROFILES: dict[RiskMode, RiskProfile] = {
    RiskMode.LOW: RiskProfile(
        mode=RiskMode.LOW,
        risk_per_trade=Decimal("0.005"),
        max_exposure=Decimal("0.30"),
        max_drawdown=Decimal("0.08"),
        max_open_positions=2,
    ),
    RiskMode.MEDIUM: RiskProfile(
        mode=RiskMode.MEDIUM,
        risk_per_trade=Decimal("0.01"),
        max_exposure=Decimal("0.60"),
        max_drawdown=Decimal("0.15"),
        max_open_positions=4,
    ),
    RiskMode.HIGH: RiskProfile(
        mode=RiskMode.HIGH,
        risk_per_trade=Decimal("0.02"),
        max_exposure=Decimal("1.00"),
        max_drawdown=Decimal("0.25"),
        max_open_positions=6,
    ),
}


def get_risk_profile(mode: RiskMode | str = RiskMode.MEDIUM) -> RiskProfile:
    try:
        risk_mode = mode if isinstance(mode, RiskMode) else RiskMode[mode.upper()]
    except KeyError as exc:
        choices = ", ".join(risk_mode.value for risk_mode in RiskMode)
        raise ValueError(f"Unknown risk profile '{mode}'. Choose one of: {choices}") from exc

    try:
        return replace(RISK_PROFILES[risk_mode])
    except KeyError as exc:
        choices = ", ".join(risk_mode.value for risk_mode in RiskMode)
        raise ValueError(f"Unknown risk profile '{mode}'. Choose one of: {choices}") from exc
