import inspect
import json
from datetime import datetime, timedelta
from decimal import Decimal

import trading_bot.ai.scanner as scanner_module
import trading_bot.ml.walk_forward as walk_forward_module
from trading_bot.ai.candidate_snapshot import CandidateSnapshot
from trading_bot.ai.openai_analyst import (
    AnalystDecision,
    AnalystRegime,
    AnalystSentiment,
    OpenAIDiagnostics,
    OpenAIAnalysisRequest,
    OpenAIAnalysisResult,
    OpenAISource,
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
        snapshot=CandidateSnapshot(
            symbol=symbol,
            decision_timestamp=START,
            close_price=Decimal("100"),
            data_age_trading_days=0,
            xgboost_probability_pct=Decimal("61"),
            return_1d_pct=Decimal("1"),
            return_5d_pct=Decimal("2"),
            return_20d_pct=Decimal("3"),
            ema20_vs_ema50_pct=Decimal("4"),
            close_vs_ema20_pct=Decimal("5"),
            rsi14=Decimal("55"),
            atr14_over_close_pct=Decimal("1.2"),
            volatility_20d_pct=Decimal("2"),
            volume_vs_20d_pct=Decimal("10"),
            portfolio_exposure_pct=Decimal("0"),
        ),
        risk_profile="MEDIUM; no leverage",
    )


def diagnostics() -> OpenAIDiagnostics:
    return OpenAIDiagnostics(
        model_used="test-model",
        http_status_code=None,
        openai_error_type=None,
        openai_error_code=None,
        failure_phase=None,
        structured_output_used=True,
        web_search_used=True,
        web_search_count=1,
        source_urls=[OpenAISource(title="Example", url="https://example.com")],
        model_request_succeeded=True,
        web_search_failed=False,
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
                    ),
                    "output": [
                        {"type": "web_search_call", "status": "completed"},
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "annotations": [
                                        {
                                            "type": "url_citation",
                                            "title": "Current company news",
                                            "url": "https://example.com/news",
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
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
    assert result.diagnostics.model_used == "test-model"
    assert result.diagnostics.web_search_used
    assert result.diagnostics.web_search_count == 1
    assert result.diagnostics.source_urls[0].url == "https://example.com/news"
    assert "secret-test-key" not in str(calls)
    payload = json.loads(calls[0]["input"][1]["content"])
    snapshot_payload = payload["candidate_snapshot"]
    assert snapshot_payload["values"]["volume_vs_20d_pct"] == "10"
    assert payload["portfolio_currency"] == "SEK"
    assert payload["investor_country"] == "Sweden"
    assert "+8.92 means 8.92% above average" in snapshot_payload["units"]["volume_vs_20d_pct"]
    assert snapshot_payload["units"]["rsi14"] == "RSI index from 0 to 100, not a percent"


def test_openai_model_success_without_web_search_is_reported() -> None:
    class FakeResponses:
        def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "output_text": json.dumps(
                        {
                            "decision": "WATCH",
                            "confidence": 0.4,
                            "sentiment": "NEUTRAL",
                            "regime": "UNCERTAIN",
                            "positive_factors": [],
                            "negative_factors": [],
                            "risk_flags": [],
                            "summary": "No search happened.",
                        }
                    ),
                    "output": [{"type": "message", "content": [{"type": "output_text", "annotations": []}]}],
                },
            )()

    class FakeClient:
        responses = FakeResponses()

    result = OpenAIAnalyst(api_key="secret-test-key", client_factory=lambda api_key: FakeClient()).analyze(
        analysis_request()
    )

    assert not result.safe_failure
    assert result.diagnostics.model_request_succeeded
    assert not result.diagnostics.web_search_used
    assert result.error == "OpenAI web search was not executed"


def test_openai_missing_key_fails_safely(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = OpenAIAnalyst().analyze(analysis_request())

    assert result.safe_failure
    assert result.decision == AnalystDecision.WATCH
    assert result.confidence == Decimal("0")
    assert result.diagnostics.model_used == "gpt-5.6-terra"
    assert result.diagnostics.failure_phase == "configuration"


def test_openai_api_failure_fails_safely() -> None:
    class FakeAPIError(RuntimeError):
        status_code = 400
        body = {"error": {"type": "invalid_request_error", "code": "unsupported_model"}}

    class BrokenResponses:
        def create(self, **kwargs):
            raise FakeAPIError("model rejected")

    class BrokenClient:
        responses = BrokenResponses()

    result = OpenAIAnalyst(api_key="secret-test-key", client_factory=lambda api_key: BrokenClient()).analyze(
        analysis_request()
    )

    assert result.safe_failure
    assert result.error is not None
    assert "HTTP status 400" in result.error
    assert result.diagnostics.http_status_code == 400
    assert result.diagnostics.openai_error_type == "invalid_request_error"
    assert result.diagnostics.openai_error_code == "unsupported_model"
    assert result.diagnostics.failure_phase == "model_request"


def test_openai_structured_output_failure_is_diagnosed() -> None:
    class FakeResponses:
        def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "output_text": "not-json",
                    "output": [{"type": "web_search_call", "status": "completed"}],
                },
            )()

    class FakeClient:
        responses = FakeResponses()

    result = OpenAIAnalyst(api_key="secret-test-key", client_factory=lambda api_key: FakeClient()).analyze(
        analysis_request()
    )

    assert result.safe_failure
    assert result.diagnostics.failure_phase == "structured_output"
    assert result.diagnostics.model_request_succeeded


def test_openai_model_success_with_failed_web_search_is_reported() -> None:
    class FakeResponses:
        def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "output_text": json.dumps(
                        {
                            "decision": "WATCH",
                            "confidence": 0.4,
                            "sentiment": "NEUTRAL",
                            "regime": "UNCERTAIN",
                            "positive_factors": [],
                            "negative_factors": [],
                            "risk_flags": [],
                            "summary": "Search failed.",
                        }
                    ),
                    "output": [{"type": "web_search_call", "status": "failed"}],
                },
            )()

    class FakeClient:
        responses = FakeResponses()

    result = OpenAIAnalyst(api_key="secret-test-key", client_factory=lambda api_key: FakeClient()).analyze(
        analysis_request()
    )

    assert result.error == "OpenAI model succeeded, web search failed"
    assert result.diagnostics.web_search_failed
    assert result.diagnostics.failure_phase == "web_search"


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
            self.calls.append(request.snapshot.symbol)
            return OpenAIAnalysisResult(
                symbol=request.snapshot.symbol,
                decision=AnalystDecision.WATCH,
                confidence=Decimal("0.5"),
                sentiment=AnalystSentiment.NEUTRAL,
                regime=AnalystRegime.UNCERTAIN,
                positive_factors=[],
                negative_factors=[],
                risk_flags=[],
                summary="mocked",
                diagnostics=diagnostics(),
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
    ).scan(datasets, scan_timestamp=START + timedelta(days=79, hours=20))

    assert len(analyst.calls) == 3
    assert len(report.analyzed_candidates) == 3
    assert report.max_openai_analyses == 3


def test_scanner_and_openai_receive_identical_candidate_snapshot(monkeypatch) -> None:
    class FixedModel:
        def fit(self, samples):
            pass

        def predict_positive_probability(self, features):
            return 0.54321

    class CapturingAnalyst:
        def __init__(self) -> None:
            self.requests = []

        def analyze(self, request):
            self.requests.append(request)
            return OpenAIAnalysisResult(
                symbol=request.snapshot.symbol,
                decision=AnalystDecision.WATCH,
                confidence=Decimal("0.5"),
                sentiment=AnalystSentiment.NEUTRAL,
                regime=AnalystRegime.UNCERTAIN,
                positive_factors=[],
                negative_factors=[],
                risk_flags=[],
                summary="mocked",
                diagnostics=diagnostics(),
            )

    monkeypatch.setattr(scanner_module, "XGBoostDecisionModel", lambda: FixedModel())
    monkeypatch.setattr(
        scanner_module,
        "build_labeled_samples",
        lambda *args, **kwargs: [sample(0, "-1"), sample(1, "1")],
    )
    analyst = CapturingAnalyst()
    report = HybridMarketScanner(
        analyst=analyst,
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        max_openai_analyses=1,
    ).scan({"ABC": candles("ABC")}, scan_timestamp=START + timedelta(days=79, hours=20))

    report_snapshot = report.analyzed_candidates[0].candidate.snapshot
    openai_snapshot = analyst.requests[0].snapshot

    assert report_snapshot == openai_snapshot
    assert report_snapshot.xgboost_probability_pct == Decimal("54.32100")
    assert str(report_snapshot.volume_vs_20d_pct) == report_snapshot.as_prompt_payload()["values"]["volume_vs_20d_pct"]
    assert report_snapshot.as_prompt_payload()["values"]["volume_vs_20d_pct"] != str(report_snapshot.volume_vs_20d_pct / Decimal("100"))


def test_openai_advisory_layer_cannot_bypass_risk_manager() -> None:
    source = inspect.getsource(scanner_module)

    assert "submit_order" not in source
    assert "RiskManager" not in source
    assert "Order(" not in source
