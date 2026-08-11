"""Current-market XGBoost + OpenAI advisory scanner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading_bot.ai.openai_analyst import OpenAIAnalysisRequest, OpenAIAnalysisResult, OpenAIAnalyst
from trading_bot.config.risk_profiles import RiskProfile
from trading_bot.data.models import Candle
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.ml.dataset import MLTargetMode, build_labeled_samples
from trading_bot.ml.features import FEATURE_NAMES, build_latest_feature_row
from trading_bot.ml.model import XGBoostDecisionModel


MAX_OPENAI_ANALYSES = 3


@dataclass(frozen=True)
class ScanCandidate:
    symbol: str
    xgboost_probability: Decimal
    technical_indicators: Mapping[str, Decimal]
    momentum: Decimal
    volatility: Decimal
    volume_vs_average: Decimal
    current_exposure_pct: Decimal


@dataclass(frozen=True)
class AnalyzedCandidate:
    candidate: ScanCandidate
    openai_analysis: OpenAIAnalysisResult | None


@dataclass(frozen=True)
class HybridScanReport:
    ranked_candidates: list[ScanCandidate]
    analyzed_candidates: list[AnalyzedCandidate]
    max_openai_analyses: int


class HybridMarketScanner:
    def __init__(
        self,
        *,
        analyst: OpenAIAnalyst,
        risk_profile: RiskProfile,
        cost_config: ExecutionCostConfig | None = None,
        max_openai_analyses: int = MAX_OPENAI_ANALYSES,
    ) -> None:
        if max_openai_analyses < 0:
            raise ValueError("max_openai_analyses must be non-negative")
        self.analyst = analyst
        self.risk_profile = risk_profile
        self.cost_config = cost_config or ExecutionCostConfig()
        self.max_openai_analyses = min(max_openai_analyses, MAX_OPENAI_ANALYSES)

    def scan(self, datasets: Mapping[str, Sequence[Candle]]) -> HybridScanReport:
        samples = [
            sample
            for candles in datasets.values()
            for sample in build_labeled_samples(
                candles,
                target_mode=MLTargetMode.TRADE_ALIGNED,
                cost_config=self.cost_config,
                stop_loss_pct=Decimal("0.05"),
            )
        ]
        model = XGBoostDecisionModel()
        model.fit(samples)

        candidates = sorted(
            [
                _score_candidate(
                    symbol=symbol,
                    candles=list(candles),
                    model=model,
                    risk_profile=self.risk_profile,
                )
                for symbol, candles in datasets.items()
            ],
            key=lambda candidate: candidate.xgboost_probability,
            reverse=True,
        )
        analyzed = [
            AnalyzedCandidate(
                candidate=candidate,
                openai_analysis=self.analyst.analyze(_analysis_request(candidate, self.risk_profile)),
            )
            for candidate in candidates[: self.max_openai_analyses]
        ]
        return HybridScanReport(
            ranked_candidates=candidates,
            analyzed_candidates=analyzed,
            max_openai_analyses=self.max_openai_analyses,
        )


def _score_candidate(
    *,
    symbol: str,
    candles: list[Candle],
    model: XGBoostDecisionModel,
    risk_profile: RiskProfile,
) -> ScanCandidate:
    feature_row = build_latest_feature_row(candles)
    if feature_row is None:
        raise ValueError(f"not enough candles to score {symbol}")
    probability = Decimal(str(model.predict_positive_probability(feature_row.as_float_list())))
    indicators = dict(zip(FEATURE_NAMES, feature_row.values, strict=True))
    return ScanCandidate(
        symbol=symbol,
        xgboost_probability=probability,
        technical_indicators=indicators,
        momentum=indicators["return_20d"],
        volatility=indicators["volatility_20d"],
        volume_vs_average=indicators["volume_vs_20d_avg"],
        current_exposure_pct=Decimal("0"),
    )


def _analysis_request(candidate: ScanCandidate, risk_profile: RiskProfile) -> OpenAIAnalysisRequest:
    return OpenAIAnalysisRequest(
        symbol=candidate.symbol,
        xgboost_probability=candidate.xgboost_probability,
        technical_indicators=candidate.technical_indicators,
        momentum=candidate.momentum,
        volatility=candidate.volatility,
        volume_vs_average=candidate.volume_vs_average,
        current_exposure_pct=candidate.current_exposure_pct,
        risk_profile=(
            f"{risk_profile.mode.value}; risk_per_trade={risk_profile.risk_per_trade}; "
            f"max_exposure={risk_profile.max_exposure}; max_open_positions={risk_profile.max_open_positions}; "
            "no leverage"
        ),
    )
