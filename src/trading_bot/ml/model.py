"""Deterministic scikit-learn probability model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading_bot.ml.dataset import MLSample


@dataclass(frozen=True)
class PredictionMetrics:
    accuracy: Decimal
    precision: Decimal
    recall: Decimal
    roc_auc: Decimal | None
    predictions: int


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


def prediction_metrics(samples: Sequence[MLSample], probabilities: Sequence[float], threshold: float) -> PredictionMetrics:
    if len(samples) != len(probabilities):
        raise ValueError("samples and probabilities must have the same length")
    if not samples:
        return PredictionMetrics(Decimal("0"), Decimal("0"), Decimal("0"), None, 0)

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
    )
