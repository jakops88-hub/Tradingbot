"""Feature construction for local ML research.

All features are computed from candles available at the evaluated candle close.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_bot.data.models import Candle
from trading_bot.strategies.indicators import exponential_moving_average


FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "ema20_vs_ema50",
    "close_vs_ema20",
    "rsi14",
    "atr14_over_close",
    "volatility_20d",
    "volume_vs_20d_avg",
]


@dataclass(frozen=True)
class FeatureRow:
    symbol: str
    timestamp: datetime
    candle_index: int
    values: tuple[Decimal, ...]

    def as_float_list(self) -> list[float]:
        return [float(value) for value in self.values]


def build_feature_rows(candles: Sequence[Candle]) -> list[FeatureRow]:
    sorted_candles = _validated_candles(candles)
    rows: list[FeatureRow] = []
    for index in range(len(sorted_candles)):
        row = build_feature_row_at(sorted_candles, index)
        if row is not None:
            rows.append(row)
    return rows


def build_latest_feature_row(candles: Sequence[Candle]) -> FeatureRow | None:
    sorted_candles = _validated_candles(candles)
    if not sorted_candles:
        return None
    return build_feature_row_at(sorted_candles, len(sorted_candles) - 1)


def build_feature_row_at(candles: Sequence[Candle], index: int) -> FeatureRow | None:
    sorted_candles = _validated_candles(candles)
    if index < 49 or index >= len(sorted_candles):
        return None

    window = sorted_candles[: index + 1]
    latest = window[-1]
    closes = [candle.close for candle in window]
    volumes = [candle.volume for candle in window]
    ema20 = exponential_moving_average(closes, 20)[-1]
    ema50 = exponential_moving_average(closes, 50)[-1]
    volume_average_20 = sum(volumes[-20:], Decimal("0")) / Decimal("20")

    return FeatureRow(
        symbol=latest.symbol,
        timestamp=latest.timestamp,
        candle_index=index,
        values=(
            _return(closes[-2], closes[-1]),
            _return(closes[-6], closes[-1]),
            _return(closes[-21], closes[-1]),
            _return(ema50, ema20),
            _return(ema20, latest.close),
            _rsi(closes, 14),
            _atr(window, 14) / latest.close,
            _volatility([_return(closes[offset - 1], closes[offset]) for offset in range(index - 19, index + 1)]),
            _return(volume_average_20, latest.volume),
        ),
    )


def _validated_candles(candles: Sequence[Candle]) -> list[Candle]:
    sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
    timestamps = [candle.timestamp for candle in sorted_candles]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("duplicate candle timestamps are not allowed")
    symbols = {candle.symbol for candle in sorted_candles}
    if len(symbols) > 1:
        raise ValueError("feature rows must be built for one symbol at a time")
    return sorted_candles


def _return(start: Decimal, end: Decimal) -> Decimal:
    if start <= 0:
        raise ValueError("return denominator must be positive")
    return (end / start) - Decimal("1")


def _rsi(closes: Sequence[Decimal], period: int) -> Decimal:
    changes = [closes[index] - closes[index - 1] for index in range(len(closes) - period, len(closes))]
    gains = [max(change, Decimal("0")) for change in changes]
    losses = [abs(min(change, Decimal("0"))) for change in changes]
    average_gain = sum(gains, Decimal("0")) / Decimal(period)
    average_loss = sum(losses, Decimal("0")) / Decimal(period)
    if average_gain == 0 and average_loss == 0:
        return Decimal("50")
    if average_loss == 0:
        return Decimal("100")
    relative_strength = average_gain / average_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + relative_strength))


def _atr(candles: Sequence[Candle], period: int) -> Decimal:
    true_ranges: list[Decimal] = []
    start_index = len(candles) - period
    for index in range(start_index, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return sum(true_ranges, Decimal("0")) / Decimal(period)


def _volatility(returns: Sequence[Decimal]) -> Decimal:
    if not returns:
        return Decimal("0")
    mean = sum(returns, Decimal("0")) / Decimal(len(returns))
    variance = sum(((value - mean) ** 2 for value in returns), Decimal("0")) / Decimal(len(returns))
    return variance.sqrt()
