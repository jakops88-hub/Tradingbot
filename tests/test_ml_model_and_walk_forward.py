from datetime import datetime
from decimal import Decimal

from trading_bot.ml.dataset import MLSample
from trading_bot.ml.model import SklearnLogisticDecisionModel
from trading_bot.ml.walk_forward import (
    CALIBRATION_CANDIDATE_THRESHOLDS,
    CALIBRATION_FALLBACK_THRESHOLD,
    MIN_VALIDATION_TRADES,
    CandidateThresholdResult,
    WalkForwardFold,
    select_calibrated_threshold,
    select_test_samples,
    select_training_samples,
    target_window_overlaps_period,
    validation_fold_for_outer_fold,
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


def candidate(
    *,
    threshold: Decimal,
    validation_return_pct: str,
    validation_max_drawdown: str,
    validation_trades: int,
) -> CandidateThresholdResult:
    eligible = validation_trades >= MIN_VALIDATION_TRADES
    drawdown = Decimal(validation_max_drawdown)
    return_pct = Decimal(validation_return_pct)
    return CandidateThresholdResult(
        threshold=threshold,
        validation_return_pct=return_pct,
        validation_max_drawdown=drawdown,
        validation_trades=validation_trades,
        weighted_win_rate=Decimal("0.50"),
        score=return_pct - (drawdown * Decimal("100")) if eligible else Decimal("-Infinity"),
        eligible=eligible,
    )


def test_validation_data_occurs_before_outer_test_data() -> None:
    outer_fold = WalkForwardFold(2018, 2024, 2025)
    validation_fold = validation_fold_for_outer_fold(outer_fold)

    assert validation_fold.train_start_year == 2018
    assert validation_fold.train_end_year == 2023
    assert validation_fold.test_year == 2024
    assert validation_fold.test_end < outer_fold.test_start


def test_outer_test_data_is_never_used_for_threshold_selection_samples() -> None:
    outer_fold = WalkForwardFold(2018, 2021, 2022)
    validation_fold = validation_fold_for_outer_fold(outer_fold)
    samples = {
        "ABC": [
            sample(
                feature_time=datetime(2020, 6, 1),
                entry_time=datetime(2020, 6, 2),
                exit_time=datetime(2020, 6, 12),
                target=0,
                first_feature="1",
            ),
            sample(
                feature_time=datetime(2021, 6, 1),
                entry_time=datetime(2021, 6, 2),
                exit_time=datetime(2021, 6, 12),
                target=1,
                first_feature="2",
            ),
            sample(
                feature_time=datetime(2022, 6, 1),
                entry_time=datetime(2022, 6, 2),
                exit_time=datetime(2022, 6, 12),
                target=1,
                first_feature="3",
            ),
        ]
    }

    internal_training = select_training_samples(samples, validation_fold)
    validation = select_test_samples(samples, validation_fold)

    assert [sample.feature_time.year for sample in internal_training] == [2020]
    assert [sample.feature_time.year for sample in validation] == [2021]
    assert all(sample.feature_time.year < outer_fold.test_year for sample in internal_training + validation)


def test_selected_threshold_comes_only_from_fixed_candidate_set() -> None:
    assert CALIBRATION_CANDIDATE_THRESHOLDS == (
        Decimal("0.50"),
        Decimal("0.525"),
        Decimal("0.55"),
        Decimal("0.575"),
        Decimal("0.60"),
    )
    selected = select_calibrated_threshold(
        [
            candidate(
                threshold=Decimal("0.525"),
                validation_return_pct="2",
                validation_max_drawdown="0.01",
                validation_trades=MIN_VALIDATION_TRADES,
            ),
            candidate(
                threshold=Decimal("0.55"),
                validation_return_pct="1",
                validation_max_drawdown="0.00",
                validation_trades=MIN_VALIDATION_TRADES,
            ),
        ]
    )

    assert selected.threshold in CALIBRATION_CANDIDATE_THRESHOLDS


def test_minimum_validation_trade_requirement_blocks_lucky_sparse_threshold() -> None:
    selected = select_calibrated_threshold(
        [
            candidate(
                threshold=Decimal("0.50"),
                validation_return_pct="1",
                validation_max_drawdown="0.01",
                validation_trades=MIN_VALIDATION_TRADES,
            ),
            candidate(
                threshold=Decimal("0.60"),
                validation_return_pct="50",
                validation_max_drawdown="0",
                validation_trades=1,
            ),
        ]
    )

    assert selected.threshold == Decimal("0.50")


def test_calibration_falls_back_to_fixed_threshold_when_no_candidate_has_enough_trades() -> None:
    selected = select_calibrated_threshold(
        [
            candidate(
                threshold=threshold,
                validation_return_pct="10",
                validation_max_drawdown="0",
                validation_trades=1,
            )
            for threshold in CALIBRATION_CANDIDATE_THRESHOLDS
        ]
    )

    assert selected.threshold == CALIBRATION_FALLBACK_THRESHOLD


def test_calibration_tie_breaking_is_deterministic() -> None:
    selected = select_calibrated_threshold(
        [
            candidate(
                threshold=Decimal("0.575"),
                validation_return_pct="3",
                validation_max_drawdown="0.01",
                validation_trades=MIN_VALIDATION_TRADES,
            ),
            candidate(
                threshold=Decimal("0.525"),
                validation_return_pct="3",
                validation_max_drawdown="0.01",
                validation_trades=MIN_VALIDATION_TRADES,
            ),
        ]
    )

    assert selected.threshold == Decimal("0.525")


def test_training_samples_for_refit_include_validation_year_but_not_test_year() -> None:
    outer_fold = WalkForwardFold(2018, 2021, 2022)
    validation_fold = validation_fold_for_outer_fold(outer_fold)
    samples = {
        "ABC": [
            sample(
                feature_time=datetime(2020, 6, 1),
                entry_time=datetime(2020, 6, 2),
                exit_time=datetime(2020, 6, 12),
                target=0,
                first_feature="1",
            ),
            sample(
                feature_time=datetime(2021, 6, 1),
                entry_time=datetime(2021, 6, 2),
                exit_time=datetime(2021, 6, 12),
                target=1,
                first_feature="2",
            ),
            sample(
                feature_time=datetime(2022, 6, 1),
                entry_time=datetime(2022, 6, 2),
                exit_time=datetime(2022, 6, 12),
                target=0,
                first_feature="3",
            ),
        ]
    }

    internal_training = select_training_samples(samples, validation_fold)
    outer_refit_training = select_training_samples(samples, outer_fold)

    assert [sample.feature_time.year for sample in internal_training] == [2020]
    assert [sample.feature_time.year for sample in outer_refit_training] == [2020, 2021]
