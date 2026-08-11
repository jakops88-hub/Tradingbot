"""Risk profile presets used by the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RiskProfile:
    name: str
    max_position_fraction: float
    max_trade_risk_fraction: float
    max_daily_loss_fraction: float
    stop_loss_pct: float
    take_profit_pct: float
    min_cash_reserve_fraction: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "max_position_fraction": self.max_position_fraction,
            "max_trade_risk_fraction": self.max_trade_risk_fraction,
            "max_daily_loss_fraction": self.max_daily_loss_fraction,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_cash_reserve_fraction": self.min_cash_reserve_fraction,
        }
        for field_name, value in values.items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.max_position_fraction > 1:
            raise ValueError("max_position_fraction must be <= 1")
        if self.min_cash_reserve_fraction >= 1:
            raise ValueError("min_cash_reserve_fraction must be < 1")


RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="conservative",
        max_position_fraction=0.10,
        max_trade_risk_fraction=0.005,
        max_daily_loss_fraction=0.01,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        min_cash_reserve_fraction=0.20,
    ),
    "balanced": RiskProfile(
        name="balanced",
        max_position_fraction=0.20,
        max_trade_risk_fraction=0.01,
        max_daily_loss_fraction=0.02,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        min_cash_reserve_fraction=0.10,
    ),
    "aggressive": RiskProfile(
        name="aggressive",
        max_position_fraction=0.35,
        max_trade_risk_fraction=0.02,
        max_daily_loss_fraction=0.04,
        stop_loss_pct=0.08,
        take_profit_pct=0.16,
        min_cash_reserve_fraction=0.05,
    ),
}


def get_risk_profile(name: str = "balanced") -> RiskProfile:
    try:
        return replace(RISK_PROFILES[name.lower()])
    except KeyError as exc:
        choices = ", ".join(sorted(RISK_PROFILES))
        raise ValueError(f"Unknown risk profile '{name}'. Choose one of: {choices}") from exc
