"""Current-market XGBoost + OpenAI advisory scanner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from trading_bot.ai.candidate_snapshot import CandidateSnapshot, ratio_to_pct
from trading_bot.ai.freshness import (
    DEFAULT_STALENESS_TOLERANCE_TRADING_DAYS,
    CandleFreshness,
    filter_completed_candles,
    freshness_for_candles,
    latest_expected_completed_daily_candle,
)
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
    snapshot: CandidateSnapshot

    @property
    def symbol(self) -> str:
        return self.snapshot.symbol

    @property
    def xgboost_probability_pct(self) -> Decimal:
        return self.snapshot.xgboost_probability_pct


@dataclass(frozen=True)
class AnalyzedCandidate:
    candidate: ScanCandidate
    openai_analysis: OpenAIAnalysisResult | None


@dataclass(frozen=True)
class StaleCandidate:
    symbol: str
    latest_candle_timestamp: datetime
    latest_expected_date: date
    data_age_trading_days: int
    current_close: Decimal
    reason: str


@dataclass(frozen=True)
class HybridScanReport:
    scan_timestamp: datetime
    ranked_candidates: list[ScanCandidate]
    analyzed_candidates: list[AnalyzedCandidate]
    stale_candidates: list[StaleCandidate]
    data_issues: list[str]
    max_openai_analyses: int


class HybridMarketScanner:
    def __init__(
        self,
        *,
        analyst: OpenAIAnalyst,
        risk_profile: RiskProfile,
        cost_config: ExecutionCostConfig | None = None,
        max_openai_analyses: int = MAX_OPENAI_ANALYSES,
        staleness_tolerance_trading_days: int = DEFAULT_STALENESS_TOLERANCE_TRADING_DAYS,
    ) -> None:
        if max_openai_analyses < 0:
            raise ValueError("max_openai_analyses must be non-negative")
        self.analyst = analyst
        self.risk_profile = risk_profile
        self.cost_config = cost_config or ExecutionCostConfig()
        self.max_openai_analyses = min(max_openai_analyses, MAX_OPENAI_ANALYSES)
        self.staleness_tolerance_trading_days = staleness_tolerance_trading_days

    def scan(
        self,
        datasets: Mapping[str, Sequence[Candle]],
        *,
        scan_timestamp: datetime | None = None,
        data_issues: Sequence[str] = (),
    ) -> HybridScanReport:
        scan_time = scan_timestamp or datetime.now()
        fresh_datasets, fresh_freshness, stale_candidates = _fresh_datasets(
            datasets,
            scan_timestamp=scan_time,
            tolerance_trading_days=self.staleness_tolerance_trading_days,
        )
        if not fresh_datasets:
            return HybridScanReport(
                scan_timestamp=scan_time,
                ranked_candidates=[],
                analyzed_candidates=[],
                stale_candidates=stale_candidates,
                data_issues=list(data_issues),
                max_openai_analyses=self.max_openai_analyses,
            )

        samples = [
            sample
            for candles in fresh_datasets.values()
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
                    freshness=fresh_freshness[symbol],
                )
                for symbol, candles in fresh_datasets.items()
            ],
            key=lambda candidate: candidate.xgboost_probability_pct,
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
            scan_timestamp=scan_time,
            ranked_candidates=candidates,
            analyzed_candidates=analyzed,
            stale_candidates=stale_candidates,
            data_issues=list(data_issues),
            max_openai_analyses=self.max_openai_analyses,
        )


def _score_candidate(
    *,
    symbol: str,
    candles: list[Candle],
    model: XGBoostDecisionModel,
    risk_profile: RiskProfile,
    freshness: CandleFreshness,
) -> ScanCandidate:
    sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
    feature_row = build_latest_feature_row(sorted_candles)
    if feature_row is None:
        raise ValueError(f"not enough candles to score {symbol}")
    probability_pct = ratio_to_pct(Decimal(str(model.predict_positive_probability(feature_row.as_float_list()))))
    indicators = dict(zip(FEATURE_NAMES, feature_row.values, strict=True))
    return ScanCandidate(
        snapshot=CandidateSnapshot(
            symbol=symbol,
            decision_timestamp=feature_row.timestamp,
            close_price=sorted_candles[-1].close,
            data_age_trading_days=freshness.data_age_trading_days,
            xgboost_probability_pct=probability_pct,
            return_1d_pct=ratio_to_pct(indicators["return_1d"]),
            return_5d_pct=ratio_to_pct(indicators["return_5d"]),
            return_20d_pct=ratio_to_pct(indicators["return_20d"]),
            ema20_vs_ema50_pct=ratio_to_pct(indicators["ema20_vs_ema50"]),
            close_vs_ema20_pct=ratio_to_pct(indicators["close_vs_ema20"]),
            rsi14=indicators["rsi14"],
            atr14_over_close_pct=ratio_to_pct(indicators["atr14_over_close"]),
            volatility_20d_pct=ratio_to_pct(indicators["volatility_20d"]),
            volume_vs_20d_pct=ratio_to_pct(indicators["volume_vs_20d_avg"]),
            portfolio_exposure_pct=Decimal("0"),
        ),
    )


def _analysis_request(candidate: ScanCandidate, risk_profile: RiskProfile) -> OpenAIAnalysisRequest:
    return OpenAIAnalysisRequest(
        snapshot=candidate.snapshot,
        risk_profile=(
            f"{risk_profile.mode.value}; risk_per_trade={risk_profile.risk_per_trade}; "
            f"max_exposure={risk_profile.max_exposure}; max_open_positions={risk_profile.max_open_positions}; "
            "no leverage"
        ),
    )


def _fresh_datasets(
    datasets: Mapping[str, Sequence[Candle]],
    *,
    scan_timestamp: datetime,
    tolerance_trading_days: int,
) -> tuple[dict[str, list[Candle]], dict[str, CandleFreshness], list[StaleCandidate]]:
    fresh: dict[str, list[Candle]] = {}
    fresh_freshness: dict[str, CandleFreshness] = {}
    stale: list[StaleCandidate] = []
    latest_expected = latest_expected_completed_daily_candle(scan_timestamp)
    for symbol, candles in datasets.items():
        completed = filter_completed_candles(list(candles), now=scan_timestamp)
        if not completed:
            stale.append(
                StaleCandidate(
                    symbol=symbol,
                    latest_candle_timestamp=datetime.min,
                    latest_expected_date=latest_expected,
                    data_age_trading_days=0,
                    current_close=Decimal("0"),
                    reason="MARKET DATA STALE - candidate rejected before AI (no completed daily candles available)",
                )
            )
            continue
        freshness = freshness_for_candles(
            completed,
            now=scan_timestamp,
            tolerance_trading_days=tolerance_trading_days,
        )
        if freshness.is_stale:
            stale.append(_stale_candidate(symbol, freshness))
            continue
        fresh[symbol] = sorted(completed, key=lambda candle: candle.timestamp)
        fresh_freshness[symbol] = freshness
    return fresh, fresh_freshness, stale


def _stale_candidate(symbol: str, freshness: CandleFreshness) -> StaleCandidate:
    return StaleCandidate(
        symbol=symbol,
        latest_candle_timestamp=freshness.latest_candle_timestamp,
        latest_expected_date=freshness.latest_expected_date,
        data_age_trading_days=freshness.data_age_trading_days,
        current_close=freshness.current_close,
        reason=freshness.reason or "MARKET DATA STALE - candidate rejected before AI",
    )
