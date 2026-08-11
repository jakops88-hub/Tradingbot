from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.data.models import Candle
from trading_bot.ml.dataset import build_labeled_samples
from trading_bot.ml.features import FEATURE_NAMES, build_feature_row_at, build_feature_rows


START = datetime(2020, 1, 1)


def make_candles(count: int) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        close = Decimal("100") + Decimal(index % 17) + (Decimal(index) / Decimal("10"))
        open_price = close - Decimal("0.25")
        candles.append(
            Candle(
                symbol="ABC",
                timestamp=START + timedelta(days=index),
                open=open_price,
                high=close + Decimal("1"),
                low=open_price - Decimal("1"),
                close=close,
                volume=Decimal("1000") + Decimal(index * 3),
            )
        )
    return candles


def test_ml_feature_schema_has_locked_features() -> None:
    assert FEATURE_NAMES == [
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


def test_features_use_no_future_candles() -> None:
    candles = make_candles(80)
    changed_future = list(candles)
    changed_future[60:] = [
        Candle(
            symbol=candle.symbol,
            timestamp=candle.timestamp,
            open=candle.open * Decimal("10"),
            high=candle.high * Decimal("10"),
            low=candle.low * Decimal("10"),
            close=candle.close * Decimal("10"),
            volume=candle.volume * Decimal("10"),
        )
        for candle in changed_future[60:]
    ]

    original = build_feature_row_at(candles, 55)
    changed = build_feature_row_at(changed_future, 55)

    assert original is not None
    assert changed is not None
    assert original.values == changed.values


def test_labels_use_future_data_only_as_training_targets() -> None:
    candles = make_candles(70)
    samples = build_labeled_samples(candles)
    first_sample = samples[0]
    feature_row = build_feature_rows(candles)[0]

    assert first_sample.feature_time == candles[49].timestamp
    assert first_sample.entry_time == candles[50].timestamp
    assert first_sample.exit_time == candles[60].timestamp
    assert first_sample.features == feature_row.values
    expected_return = (candles[60].open / candles[50].open) - Decimal("1")
    assert first_sample.target_return == expected_return
    assert first_sample.target == (1 if expected_return > 0 else 0)
