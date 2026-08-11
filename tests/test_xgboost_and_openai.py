import inspect
import json
from datetime import datetime, timedelta
from decimal import Decimal

import trading_bot.ai.scanner as scanner_module
import trading_bot.ml.walk_forward as walk_forward_module
from trading_bot.ai.openai_analyst import (
    AnalystDecision,
    AnalystRegime,
    AnalystSentiment,
    OpenAIAnalysisRequest,
    OpenAIAnalysisResult,
    OpenAIAnalyst,
)
from trading_bot.ai.scanner import HybridMarketScanner
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle
from trading_bot.ml.dataset import MLSample
from trading_bot.ml.features import build_feature_row_at
from trading_bot.ml.model import XGBOOST_FIXED_CONFIG, XGBoostDecisionModel


START = datetime(2021, 1, 1)


def sample(target: int, first_feature: str) -> MLSample:
    return MLSample(
        symbol="ABC",
        feature_time=START,
        entry_time=START + timedelta(days=1),
        exit_time=START + timedelta(days=11),
        features=(Decimal(first_feature),) + (Decimal("0"),) * 8,
        target=target,
        target_return=Decimal("0.01") if target else Decimal("-0.01"),
    )


def candles(symbol: str, count: int = 80, offset: int = 0) -> list[Candle]:
    rows: list[Candle] = []
    for index in range(count):
        close = Decimal("100") + Decimal((index + offset) % 13) + (Decimal(index) / Decimal("20"))
        open_price = close - Decimal("0.10")
        rows.append(
            Candle(
                symbol=symbol,
                timestamp=START + timedelta(days=index),
                open=open_price,
                high=close + Decimal("1"),
                low=open_price - Decimal("1"),
                close=close,
                volume=Decimal("1000") + Decimal(index * 5 + offset),
            )
        )
    return rows


def analysis_request(symbol: str = "VOLV-B.ST") -> OpenAIAnalysisRequest:
    return OpenAIAnalysisRequest(
        symbol=symbol,
        xgboost_probability=Decimal("0.61"),
        technical_indicators={"return_20d": Decimal("0.03")},
        momentum=Decimal("0.03"),
        volatility=Decimal("0.02"),
        volume_vs_average=Decimal("0.10"),
        current_exposure_pct=Decimal("0"),
        risk_profile="MEDIUM; no leverage",
    )


def test_xgboost_training_and_prediction() -> None:
    model = XGBoostDecisionModel()
    model.fit(
        [
            sample(0, "-2"),
            sample(0, "-1"),
            sample(1, "1"),
            sample(1, "2"),
        ]
    )

    probability = model.predict_positive_probability([2.0] + [0.0] * 8)

    assert 0 <= probability <= 1
    assert XGBOOST_FIXED_CONFIG["max_depth"] == 3
    assert XGBOOST_FIXED_CONFIG["n_estimators"] == 100


def test_xgboost_features_use_no_future_candles() -> None:
    original = candles("ABC", 80)
    changed_future = list(original)
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

    original_features = build_feature_row_at(original, 55)
    changed_features = build_feature_row_at(changed_future, 55)

    assert original_features is not None
    assert changed_features is not None
    assert original_features.values == changed_features.values


def test_historical_xgb_research_never_imports_openai() -> None:
    source = inspect.getsource(walk_forward_module)

    assert "openai" not in source.lower()


def test_openai_structured_response_parsing_and_key_not_logged() -> None:
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return type(
                "Response",
                (),
                {
                    "output_text": json.dumps(
                        {
                            "decision": "APPROVE",
                            "confidence": 0.7,
                            "sentiment": "POSITIVE",
                            "regime": "BULLISH",
                            "positive_factors": ["trend"],
                            "negative_factors": ["valuation"],
                            "risk_flags": ["earnings soon"],
                            "summary": "Constructive but advisory only.",
                        }
                    )
                },
            )()

    class FakeClient:
        responses = FakeResponses()

    analyst = OpenAIAnalyst(api_key="secret-test-key", model="test-model", client_factory=lambda api_key: FakeClient())
    result = analyst.analyze(analysis_request())

    assert result.decision == AnalystDecision.APPROVE
    assert result.sentiment == AnalystSentiment.POSITIVE
    assert result.regime == AnalystRegime.BULLISH
    assert result.positive_factors == ["trend"]
    assert "secret-test-key" not in str(calls)


def test_openai_missing_key_fails_safely(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = OpenAIAnalyst().analyze(analysis_request())

    assert result.safe_failure
    assert result.decision == AnalystDecision.WATCH
    assert result.confidence == Decimal("0")


def test_openai_api_failure_fails_safely() -> None:
    class BrokenResponses:
        def create(self, **kwargs):
            raise RuntimeError("network down")

    class BrokenClient:
        responses = BrokenResponses()

    result = OpenAIAnalyst(api_key="secret-test-key", client_factory=lambda api_key: BrokenClient()).analyze(
        analysis_request()
    )

    assert result.safe_failure
    assert result.error is not None
    assert "network down" in result.error


def test_hybrid_scanner_limits_openai_calls(monkeypatch) -> None:
    class FakeModel:
        def fit(self, samples):
            self.samples = samples

        def predict_positive_probability(self, features):
            return min(0.99, 0.50 + abs(features[0]))

    class CountingAnalyst:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def analyze(self, request):
            self.calls.append(request.symbol)
            return OpenAIAnalysisResult(
                symbol=request.symbol,
                decision=AnalystDecision.WATCH,
                confidence=Decimal("0.5"),
                sentiment=AnalystSentiment.NEUTRAL,
                regime=AnalystRegime.UNCERTAIN,
                positive_factors=[],
                negative_factors=[],
                risk_flags=[],
                summary="mocked",
            )

    monkeypatch.setattr(scanner_module, "XGBoostDecisionModel", lambda: FakeModel())
    monkeypatch.setattr(
        scanner_module,
        "build_labeled_samples",
        lambda *args, **kwargs: [sample(0, "-1"), sample(1, "1")],
    )
    analyst = CountingAnalyst()
    datasets = {f"SYM{index}": candles(f"SYM{index}", offset=index) for index in range(5)}

    report = HybridMarketScanner(
        analyst=analyst,
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
    ).scan(datasets)

    assert len(analyst.calls) == 3
    assert len(report.analyzed_candidates) == 3
    assert report.max_openai_analyses == 3


def test_openai_advisory_layer_cannot_bypass_risk_manager() -> None:
    source = inspect.getsource(scanner_module)

    assert "submit_order" not in source
    assert "RiskManager" not in source
    assert "Order(" not in source
