"""Labeled ML dataset construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_bot.data.models import Candle
from trading_bot.ml.features import FeatureRow, build_feature_rows


TARGET_HORIZON_TRADING_DAYS = 10


@dataclass(frozen=True)
class MLSample:
    symbol: str
    feature_time: datetime
    entry_time: datetime
    exit_time: datetime
    features: tuple[Decimal, ...]
    target: int
    target_return: Decimal

    def feature_floats(self) -> list[float]:
        return [float(value) for value in self.features]


def build_labeled_samples(
    candles: Sequence[Candle],
    *,
    horizon: int = TARGET_HORIZON_TRADING_DAYS,
) -> list[MLSample]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
    feature_rows = build_feature_rows(sorted_candles)
    samples: list[MLSample] = []
    for feature_row in feature_rows:
        entry_index = feature_row.candle_index + 1
        exit_index = entry_index + horizon
        if exit_index >= len(sorted_candles):
            continue
        entry_price = sorted_candles[entry_index].open
        exit_price = sorted_candles[exit_index].open
        target_return = (exit_price / entry_price) - Decimal("1")
        samples.append(
            MLSample(
                symbol=feature_row.symbol,
                feature_time=feature_row.timestamp,
                entry_time=sorted_candles[entry_index].timestamp,
                exit_time=sorted_candles[exit_index].timestamp,
                features=feature_row.values,
                target=1 if target_return > 0 else 0,
                target_return=target_return,
            )
        )
    return samples


def feature_row_to_sample_features(feature_row: FeatureRow) -> tuple[Decimal, ...]:
    return feature_row.values
