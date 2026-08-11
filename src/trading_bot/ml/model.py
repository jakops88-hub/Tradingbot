"""Deterministic scikit-learn probability model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading_bot.ml.dataset import MLSample


@dataclass(frozen=True)
class ProbabilityBucket:
    label: str
    predictions: int
    actual_positive_rate: Decimal
    average_net_trade_return: Decimal


@dataclass(frozen=True)
class PredictionMetrics:
    accuracy: Decimal
    precision: Decimal
    recall: Decimal
    roc_auc: Decimal | None
    predictions: int
    positive_class_rate: Decimal
    predicted_buy_rate: Decimal
    average_predicted_probability: Decimal
    average_probability_for_positive_labels: Decimal
    average_probability_for_negative_labels: Decimal
    probability_buckets: list[ProbabilityBucket]


class ProbabilityDecisionModel(Protocol):
    def fit(self, samples: Sequence[MLSample]) -> None:
        ...

    def predict_positive_probability(self, features: Sequence[float]) -> float:
        ...

    def predict_probabilities(self, samples: Sequence[MLSample]) -> list[float]:
        ...


class SklearnLogisticDecisionModel:
    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        solver="liblinear",
                        random_state=0,
                        max_iter=1000,
                    ),
                ),
            ]
        )

    def fit(self, samples: Sequence[MLSample]) -> None:
        if not samples:
            raise ValueError("training samples cannot be empty")
        targets = [sample.target for sample in samples]
        if len(set(targets)) < 2:
            raise ValueError("training samples must contain both positive and negative targets")
        self.pipeline.fit([sample.feature_floats() for sample in samples], targets)

    def predict_positive_probability(self, features: Sequence[float]) -> float:
        return float(self.pipeline.predict_proba([list(features)])[0][1])

    def predict_probabilities(self, samples: Sequence[MLSample]) -> list[float]:
        if not samples:
            return []
        probabilities = self.pipeline.predict_proba([sample.feature_floats() for sample in samples])
        return [float(row[1]) for row in probabilities]


XGBOOST_FIXED_CONFIG = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 0,
    "n_jobs": 1,
}


class XGBoostDecisionModel:
    def __init__(self) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install XGBoost with `python -m pip install -e .[ml]`") from exc

        self.model = XGBClassifier(**XGBOOST_FIXED_CONFIG)

    def fit(self, samples: Sequence[MLSample]) -> None:
        if not samples:
            raise ValueError("training samples cannot be empty")
        targets = [sample.target for sample in samples]
        if len(set(targets)) < 2:
            raise ValueError("training samples must contain both positive and negative targets")
        self.model.fit([sample.feature_floats() for sample in samples], targets)

    def predict_positive_probability(self, features: Sequence[float]) -> float:
        return float(self.model.predict_proba([list(features)])[0][1])

    def predict_probabilities(self, samples: Sequence[MLSample]) -> list[float]:
        if not samples:
            return []
        probabilities = self.model.predict_proba([sample.feature_floats() for sample in samples])
        return [float(row[1]) for row in probabilities]


def prediction_metrics(samples: Sequence[MLSample], probabilities: Sequence[float], threshold: float) -> PredictionMetrics:
    if len(samples) != len(probabilities):
        raise ValueError("samples and probabilities must have the same length")
    if not samples:
        return PredictionMetrics(
            accuracy=Decimal("0"),
            precision=Decimal("0"),
            recall=Decimal("0"),
            roc_auc=None,
            predictions=0,
            positive_class_rate=Decimal("0"),
            predicted_buy_rate=Decimal("0"),
            average_predicted_probability=Decimal("0"),
            average_probability_for_positive_labels=Decimal("0"),
            average_probability_for_negative_labels=Decimal("0"),
            probability_buckets=_probability_buckets([], []),
        )

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    actual = [sample.target for sample in samples]
    predicted = [1 if probability >= threshold else 0 for probability in probabilities]
    roc_auc: Decimal | None
    try:
        roc_auc = Decimal(str(roc_auc_score(actual, list(probabilities))))
    except ValueError:
        roc_auc = None

    return PredictionMetrics(
        accuracy=Decimal(str(accuracy_score(actual, predicted))),
        precision=Decimal(str(precision_score(actual, predicted, zero_division=0))),
        recall=Decimal(str(recall_score(actual, predicted, zero_division=0))),
        roc_auc=roc_auc,
        predictions=len(samples),
        positive_class_rate=Decimal(sum(actual)) / Decimal(len(actual)),
        predicted_buy_rate=Decimal(sum(predicted)) / Decimal(len(predicted)),
        average_predicted_probability=_average_decimal([Decimal(str(probability)) for probability in probabilities]),
        average_probability_for_positive_labels=_average_decimal(
            [Decimal(str(probability)) for sample, probability in zip(samples, probabilities) if sample.target == 1]
        ),
        average_probability_for_negative_labels=_average_decimal(
            [Decimal(str(probability)) for sample, probability in zip(samples, probabilities) if sample.target == 0]
        ),
        probability_buckets=_probability_buckets(samples, probabilities),
    )


def _probability_buckets(samples: Sequence[MLSample], probabilities: Sequence[float]) -> list[ProbabilityBucket]:
    bucket_definitions = [
        ("50-55%", Decimal("0.50"), Decimal("0.55")),
        ("55-60%", Decimal("0.55"), Decimal("0.60")),
        ("60-65%", Decimal("0.60"), Decimal("0.65")),
        ("65-70%", Decimal("0.65"), Decimal("0.70")),
        ("70%+", Decimal("0.70"), Decimal("Infinity")),
    ]
    rows = [(sample, Decimal(str(probability))) for sample, probability in zip(samples, probabilities)]
    buckets: list[ProbabilityBucket] = []
    for label, lower, upper in bucket_definitions:
        bucket_rows = [
            (sample, probability)
            for sample, probability in rows
            if probability >= lower and (upper == Decimal("Infinity") or probability < upper)
        ]
        actual_positive_rate = (
            Decimal(sum(sample.target for sample, _ in bucket_rows)) / Decimal(len(bucket_rows))
            if bucket_rows
            else Decimal("0")
        )
        average_net_trade_return = _average_decimal(
            [
                sample.net_trade_return if sample.net_trade_return is not None else sample.target_return
                for sample, _ in bucket_rows
            ]
        )
        buckets.append(
            ProbabilityBucket(
                label=label,
                predictions=len(bucket_rows),
                actual_positive_rate=actual_positive_rate,
                average_net_trade_return=average_net_trade_return,
            )
        )
    return buckets


def _average_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, Decimal("0")) / Decimal(len(values))
