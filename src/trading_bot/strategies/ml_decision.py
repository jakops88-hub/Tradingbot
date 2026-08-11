"""Long-only ML probability strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.ml.features import build_latest_feature_row
from trading_bot.strategies.base import Strategy


class ProbabilityModel(Protocol):
    def predict_positive_probability(self, features: Sequence[float]) -> float:
        ...


@dataclass(frozen=True)
class MLDecisionConfig:
    probability_threshold: Decimal = Decimal("0.60")
    stop_loss_pct: Decimal = Decimal("0.05")
    max_holding_period: int = 10

    def __post_init__(self) -> None:
        if not Decimal("0") <= self.probability_threshold <= Decimal("1"):
            raise ValueError("probability_threshold must be between 0 and 1")
        if not Decimal("0") < self.stop_loss_pct < Decimal("1"):
            raise ValueError("stop_loss_pct must be between 0 and 1")
        if self.max_holding_period <= 0:
            raise ValueError("max_holding_period must be positive")


class MLDecisionStrategy(Strategy):
    name = "ml_decision_v1"

    def __init__(self, model: ProbabilityModel, config: MLDecisionConfig | None = None) -> None:
        self.model = model
        self.config = config or MLDecisionConfig()
        self.holding_bars = 0

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        latest = candles[-1]
        if snapshot.open_positions > 0:
            self.holding_bars += 1
            if self.holding_bars >= self.config.max_holding_period:
                return Signal(
                    symbol=latest.symbol,
                    action=SignalAction.SELL,
                    generated_at=latest.timestamp,
                    reason="maximum ML holding period reached",
                )
            return Signal(latest.symbol, SignalAction.HOLD, latest.timestamp, reason="holding ML position")

        self.holding_bars = 0
        feature_row = build_latest_feature_row(candles)
        if feature_row is None:
            return Signal(latest.symbol, SignalAction.HOLD, latest.timestamp, reason="ML feature warm-up")

        probability = Decimal(str(self.model.predict_positive_probability(feature_row.as_float_list())))
        if probability >= self.config.probability_threshold:
            return Signal(
                symbol=latest.symbol,
                action=SignalAction.BUY,
                generated_at=latest.timestamp,
                confidence=probability,
                reason=f"ML probability {probability} >= {self.config.probability_threshold}",
                stop_loss_price=latest.close * (Decimal("1") - self.config.stop_loss_pct),
            )
        return Signal(
            symbol=latest.symbol,
            action=SignalAction.HOLD,
            generated_at=latest.timestamp,
            confidence=probability,
            reason=f"ML probability {probability} below threshold",
        )
