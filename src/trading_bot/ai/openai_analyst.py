"""OpenAI advisory analysis for current trade candidates.

This module is intentionally advisory-only. It does not create orders, call a
broker, change risk settings, or participate in historical backtests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class AnalystDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    WATCH = "WATCH"


class AnalystSentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class AnalystRegime(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class OpenAIAnalysisRequest:
    symbol: str
    xgboost_probability: Decimal
    technical_indicators: Mapping[str, Decimal]
    momentum: Decimal
    volatility: Decimal
    volume_vs_average: Decimal
    current_exposure_pct: Decimal
    risk_profile: str


@dataclass(frozen=True)
class OpenAIAnalysisResult:
    symbol: str
    decision: AnalystDecision
    confidence: Decimal
    sentiment: AnalystSentiment
    regime: AnalystRegime
    positive_factors: list[str]
    negative_factors: list[str]
    risk_flags: list[str]
    summary: str
    safe_failure: bool = False
    error: str | None = None


OpenAIClientFactory = Callable[[str], Any]


class OpenAIAnalyst:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client_factory: OpenAIClientFactory | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.client_factory = client_factory or _default_client_factory

    def analyze(self, request: OpenAIAnalysisRequest) -> OpenAIAnalysisResult:
        if not self.api_key:
            return _safe_failure(request.symbol, "OPENAI_API_KEY is not configured")
        try:
            client = self.client_factory(self.api_key)
            response = client.responses.create(
                model=self.model,
                tools=[{"type": "web_search_preview"}],
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are an advisory market analyst. Analyze only the current candidate. "
                            "Do not recommend leverage, do not change risk settings or stop loss, "
                            "do not call brokers, and do not claim certainty."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(_request_payload(request), default=str),
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "trade_candidate_analysis",
                        "strict": True,
                        "schema": _analysis_schema(),
                    }
                },
            )
            return _parse_response(request.symbol, _response_text(response))
        except Exception as exc:
            return _safe_failure(request.symbol, f"OpenAI analysis failed: {exc}")


def _default_client_factory(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install OpenAI support with `python -m pip install -e .[ai]`") from exc
    return OpenAI(api_key=api_key)


def _request_payload(request: OpenAIAnalysisRequest) -> dict[str, Any]:
    return {
        "symbol": request.symbol,
        "xgboost_probability": str(request.xgboost_probability),
        "technical_indicators": {name: str(value) for name, value in request.technical_indicators.items()},
        "momentum": str(request.momentum),
        "volatility": str(request.volatility),
        "volume_vs_average": str(request.volume_vs_average),
        "current_exposure_pct": str(request.current_exposure_pct),
        "risk_profile": request.risk_profile,
        "constraints": [
            "OpenAI is advisory only",
            "No broker calls",
            "No risk setting changes",
            "No leverage",
            "No stop-loss changes",
        ],
    }


def _analysis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision",
            "confidence",
            "sentiment",
            "regime",
            "positive_factors",
            "negative_factors",
            "risk_flags",
            "summary",
        ],
        "properties": {
            "decision": {"type": "string", "enum": [decision.value for decision in AnalystDecision]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sentiment": {"type": "string", "enum": [sentiment.value for sentiment in AnalystSentiment]},
            "regime": {"type": "string", "enum": [regime.value for regime in AnalystRegime]},
            "positive_factors": {"type": "array", "items": {"type": "string"}},
            "negative_factors": {"type": "array", "items": {"type": "string"}},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
    }


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = getattr(response, "output", None)
    if isinstance(output, Sequence):
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, Sequence):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    return text
    raise ValueError("OpenAI response did not contain output_text")


def _parse_response(symbol: str, text: str) -> OpenAIAnalysisResult:
    payload = json.loads(text)
    return OpenAIAnalysisResult(
        symbol=symbol,
        decision=AnalystDecision(payload["decision"]),
        confidence=_confidence(payload["confidence"]),
        sentiment=AnalystSentiment(payload["sentiment"]),
        regime=AnalystRegime(payload["regime"]),
        positive_factors=list(payload["positive_factors"]),
        negative_factors=list(payload["negative_factors"]),
        risk_flags=list(payload["risk_flags"]),
        summary=str(payload["summary"]),
    )


def _confidence(value: Any) -> Decimal:
    confidence = Decimal(str(value))
    if not Decimal("0") <= confidence <= Decimal("1"):
        raise ValueError("OpenAI confidence must be between 0 and 1")
    return confidence


def _safe_failure(symbol: str, message: str) -> OpenAIAnalysisResult:
    return OpenAIAnalysisResult(
        symbol=symbol,
        decision=AnalystDecision.WATCH,
        confidence=Decimal("0"),
        sentiment=AnalystSentiment.NEUTRAL,
        regime=AnalystRegime.UNCERTAIN,
        positive_factors=[],
        negative_factors=[],
        risk_flags=["OpenAI advisory unavailable"],
        summary="No AI advisory recommendation was produced.",
        safe_failure=True,
        error=message,
    )
