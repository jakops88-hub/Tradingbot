"""OpenAI advisory analysis for current trade candidates.

This module is intentionally advisory-only. It does not create orders, call a
broker, change risk settings, or participate in historical backtests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from typing import Any

from trading_bot.ai.candidate_snapshot import CandidateSnapshot


DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"


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
    snapshot: CandidateSnapshot
    risk_profile: str


@dataclass(frozen=True)
class OpenAISource:
    title: str
    url: str


@dataclass(frozen=True)
class OpenAIDiagnostics:
    model_used: str
    http_status_code: int | None
    openai_error_type: str | None
    openai_error_code: str | None
    failure_phase: str | None
    structured_output_used: bool
    web_search_used: bool
    web_search_count: int | None
    source_urls: list[OpenAISource]
    model_request_succeeded: bool
    web_search_failed: bool


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
    diagnostics: OpenAIDiagnostics
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
            return _safe_failure(
                request.snapshot.symbol,
                "OPENAI_API_KEY is not configured",
                model_used=self.model,
                failure_phase="configuration",
            )
        response: Any | None = None
        try:
            client = self.client_factory(self.api_key)
            response = client.responses.create(
                model=self.model,
                tools=[{"type": "web_search", "search_context_size": "low"}],
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are an advisory market analyst. Analyze only the current candidate. "
                            "You must use web search for current public news and market context. "
                            "The portfolio currency is SEK and the investor country is Sweden; "
                            "do not assume a US-based portfolio. "
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
        except Exception as exc:
            return _safe_failure(
                request.snapshot.symbol,
                _format_openai_error(exc),
                model_used=self.model,
                failure_phase="model_request",
                http_status_code=_http_status_code(exc),
                openai_error_type=_openai_error_type(exc),
                openai_error_code=_openai_error_code(exc),
            )

        diagnostics = _response_diagnostics(response, model_used=self.model)
        try:
            result = _parse_response(request.snapshot.symbol, _response_text(response), diagnostics)
        except Exception as exc:
            return _safe_failure(
                request.snapshot.symbol,
                f"OpenAI Structured Output parsing failed: {exc}",
                model_used=self.model,
                failure_phase="structured_output",
                diagnostics=diagnostics,
            )
        if diagnostics.web_search_failed:
            return OpenAIAnalysisResult(
                symbol=result.symbol,
                decision=result.decision,
                confidence=result.confidence,
                sentiment=result.sentiment,
                regime=result.regime,
                positive_factors=result.positive_factors,
                negative_factors=result.negative_factors,
                risk_flags=[*result.risk_flags, "OpenAI model succeeded, web search failed"],
                summary=result.summary,
                diagnostics=diagnostics,
                safe_failure=False,
                error="OpenAI model succeeded, web search failed",
            )
        if not diagnostics.web_search_used:
            return OpenAIAnalysisResult(
                symbol=result.symbol,
                decision=result.decision,
                confidence=result.confidence,
                sentiment=result.sentiment,
                regime=result.regime,
                positive_factors=result.positive_factors,
                negative_factors=result.negative_factors,
                risk_flags=[*result.risk_flags, "OpenAI web search was not executed"],
                summary=result.summary,
                diagnostics=diagnostics,
                safe_failure=False,
                error="OpenAI web search was not executed",
            )
        return result


def _default_client_factory(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install OpenAI support with `python -m pip install -e .[ai]`") from exc
    return OpenAI(api_key=api_key)


def _request_payload(request: OpenAIAnalysisRequest) -> dict[str, Any]:
    return {
        "candidate_snapshot": request.snapshot.as_prompt_payload(),
        "portfolio_currency": "SEK",
        "investor_country": "Sweden",
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


def _parse_response(symbol: str, text: str, diagnostics: OpenAIDiagnostics) -> OpenAIAnalysisResult:
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
        diagnostics=diagnostics,
    )


def _confidence(value: Any) -> Decimal:
    confidence = Decimal(str(value))
    if not Decimal("0") <= confidence <= Decimal("1"):
        raise ValueError("OpenAI confidence must be between 0 and 1")
    return confidence


def _response_diagnostics(response: Any, *, model_used: str) -> OpenAIDiagnostics:
    output = getattr(response, "output", None)
    web_search_count = 0
    web_search_failed = False
    sources: list[OpenAISource] = []
    if isinstance(output, Sequence):
        for item in output:
            item_type = _value(item, "type")
            if item_type in {"web_search_call", "web_search_preview_call"}:
                web_search_count += 1
                status = _value(item, "status")
                if status not in {None, "completed"}:
                    web_search_failed = True
            content = _value(item, "content")
            if isinstance(content, Sequence):
                for part in content:
                    annotations = _value(part, "annotations")
                    if isinstance(annotations, Sequence):
                        sources.extend(_annotation_sources(annotations))
    return OpenAIDiagnostics(
        model_used=model_used,
        http_status_code=None,
        openai_error_type=None,
        openai_error_code=None,
        failure_phase="web_search" if web_search_failed else None,
        structured_output_used=True,
        web_search_used=web_search_count > 0 and not web_search_failed,
        web_search_count=web_search_count,
        source_urls=_dedupe_sources(sources),
        model_request_succeeded=True,
        web_search_failed=web_search_failed,
    )


def _annotation_sources(annotations: Sequence[Any]) -> list[OpenAISource]:
    sources: list[OpenAISource] = []
    for annotation in annotations:
        if _value(annotation, "type") != "url_citation":
            continue
        url = str(_value(annotation, "url") or "")
        title = str(_value(annotation, "title") or url)
        if url:
            sources.append(OpenAISource(title=title, url=url))
    return sources


def _dedupe_sources(sources: Sequence[OpenAISource]) -> list[OpenAISource]:
    seen: set[str] = set()
    unique: list[OpenAISource] = []
    for source in sources:
        if source.url in seen:
            continue
        seen.add(source.url)
        unique.append(source)
    return unique


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _safe_failure(
    symbol: str,
    message: str,
    *,
    model_used: str,
    failure_phase: str,
    http_status_code: int | None = None,
    openai_error_type: str | None = None,
    openai_error_code: str | None = None,
    diagnostics: OpenAIDiagnostics | None = None,
) -> OpenAIAnalysisResult:
    failure_diagnostics = (
        replace(
            diagnostics,
            failure_phase=failure_phase,
            http_status_code=http_status_code,
            openai_error_type=openai_error_type,
            openai_error_code=openai_error_code,
        )
        if diagnostics is not None
        else OpenAIDiagnostics(
        model_used=model_used,
        http_status_code=http_status_code,
        openai_error_type=openai_error_type,
        openai_error_code=openai_error_code,
        failure_phase=failure_phase,
        structured_output_used=failure_phase != "configuration",
        web_search_used=False,
        web_search_count=None,
        source_urls=[],
        model_request_succeeded=False,
        web_search_failed=failure_phase == "web_search",
    )
    )
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
        diagnostics=failure_diagnostics,
        safe_failure=True,
        error=message,
    )


def _format_openai_error(exc: Exception) -> str:
    status_code = _http_status_code(exc)
    error_type = _openai_error_type(exc)
    error_code = _openai_error_code(exc)
    parts = ["OpenAI analysis failed"]
    if status_code is not None:
        parts.append(f"HTTP status {status_code}")
    if error_type:
        parts.append(f"type={error_type}")
    if error_code:
        parts.append(f"code={error_code}")
    message = str(exc)
    if message:
        parts.append(message)
    return "; ".join(parts)


def _http_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _openai_error_type(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("type"):
            return str(error["type"])
    error_type = getattr(exc, "type", None)
    return str(error_type) if error_type else None


def _openai_error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("code"):
            return str(error["code"])
    error_code = getattr(exc, "code", None)
    return str(error_code) if error_code else None
