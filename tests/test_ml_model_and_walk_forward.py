from datetime import datetime
from decimal import Decimal

from trading_bot.ml.dataset import MLSample
from trading_bot.ml.model import SklearnLogisticDecisionModel
from trading_bot.ml.walk_forward import (
    WalkForwardFold,
    select_test_samples,
    select_training_samples,
    target_window_overlaps_period,
)


def sample(
    *,
    feature_time: datetime,
    entry_time: datetime,
    exit_time: datetime,
    target: int,
    first_feature: str,
) -> MLSample:
    return MLSample(
        symbol="ABC",
        feature_time=feature_time,
        entry_time=entry_time,
        exit_time=exit_time,
        features=(Decimal(first_feature),) + (Decimal("0"),) * 8,
        target=target,
        target_return=Decimal("0.01") if target else Decimal("-0.01"),
    )


def test_preprocessing_fits_training_data_only() -> None:
    training = [
        sample(
            feature_time=datetime(2020, 1, 1),
            entry_time=datetime(2020, 1, 2),
            exit_time=datetime(2020, 1, 12),
            target=0,
            first_feature="0",
        ),
        sample(
            feature_time=datetime(2020, 1, 2),
            entry_time=datetime(2020, 1, 3),
            exit_time=datetime(2020, 1, 13),
            target=1,
            first_feature="2",
        ),
    ]
    test = sample(
        feature_time=datetime(2022, 1, 1),
        entry_time=datetime(2022, 1, 2),
        exit_time=datetime(2022, 1, 12),
        target=1,
        first_feature="100",
    )

    model = SklearnLogisticDecisionModel()
    model.fit(training)

    scaler = model.pipeline.named_steps["scaler"]
    assert Decimal(str(scaler.mean_[0])) == Decimal("1.0")
    assert test.feature_floats()[0] == 100.0


def test_test_year_samples_never_enter_training() -> None:
    fold = WalkForwardFold(2018, 2021, 2022)
    samples = {
        "ABC": [
            sample(
                feature_time=datetime(2021, 6, 1),
                entry_time=datetime(2021, 6, 2),
                exit_time=datetime(2021, 6, 12),
                target=1,
                first_feature="1",
            ),
            sample(
                feature_time=datetime(2022, 6, 1),
                entry_time=datetime(2022, 6, 2),
                exit_time=datetime(2022, 6, 12),
                target=0,
                first_feature="2",
            ),
        ]
    }

    training = select_training_samples(samples, fold)
    testing = select_test_samples(samples, fold)

    assert [sample.feature_time.year for sample in training] == [2021]
    assert [sample.feature_time.year for sample in testing] == [2022]


def test_target_windows_overlapping_test_period_are_purged() -> None:
    fold = WalkForwardFold(2018, 2021, 2022)
    overlapping = sample(
        feature_time=datetime(2021, 12, 20),
        entry_time=datetime(2021, 12, 21),
        exit_time=datetime(2022, 1, 5),
        target=1,
        first_feature="1",
    )
    clean = sample(
        feature_time=datetime(2021, 12, 1),
        entry_time=datetime(2021, 12, 2),
        exit_time=datetime(2021, 12, 15),
        target=0,
        first_feature="2",
    )

    training = select_training_samples({"ABC": [overlapping, clean]}, fold)

    assert target_window_overlaps_period(overlapping, fold.test_start, fold.test_end)
    assert not target_window_overlaps_period(clean, fold.test_start, fold.test_end)
    assert training == [clean]
