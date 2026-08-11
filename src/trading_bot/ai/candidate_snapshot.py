"""Canonical typed snapshot for current-market candidate analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class CandidateSnapshot:
    symbol: str
    decision_timestamp: datetime
    close_price: Decimal
    data_age_trading_days: int
    xgboost_probability_pct: Decimal
    return_1d_pct: Decimal
    return_5d_pct: Decimal
    return_20d_pct: Decimal
    ema20_vs_ema50_pct: Decimal
    close_vs_ema20_pct: Decimal
    rsi14: Decimal
    atr14_over_close_pct: Decimal
    volatility_20d_pct: Decimal
    volume_vs_20d_pct: Decimal
    portfolio_exposure_pct: Decimal

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "units": {
                "xgboost_probability_pct": "percentage points from 0 to 100",
                "decision_timestamp": "timestamp of the completed candle used for this decision",
                "close_price": "latest completed candle close price in the instrument quote currency",
                "data_age_trading_days": "trading days between the latest candle and expected latest completed candle",
                "return_1d_pct": "percentage points; positive means price is above 1 trading day ago",
                "return_5d_pct": "percentage points; positive means price is above 5 trading days ago",
                "return_20d_pct": "percentage points; positive means price is above 20 trading days ago",
                "ema20_vs_ema50_pct": "percentage points; positive means EMA20 is above EMA50",
                "close_vs_ema20_pct": "percentage points; positive means close is above EMA20",
                "rsi14": "RSI index from 0 to 100, not a percent",
                "atr14_over_close_pct": "percentage points of price",
                "volatility_20d_pct": "standard deviation of daily returns, in percentage points",
                "volume_vs_20d_pct": (
                    "percentage points versus 20-day average volume; +8.92 means 8.92% above average, "
                    "-12.57 means 12.57% below average"
                ),
                "portfolio_exposure_pct": "current portfolio exposure in percentage points",
            },
            "values": {name: str(value) for name, value in asdict(self).items()},
        }

    def report_lines(self) -> list[str]:
        return [
            f"symbol: {self.symbol}",
            f"decision_timestamp: {self.decision_timestamp.isoformat()}",
            f"close_price: {self.close_price}",
            f"data_age_trading_days: {self.data_age_trading_days}",
            f"xgboost_probability_pct: {self.xgboost_probability_pct}",
            f"return_1d_pct: {self.return_1d_pct}",
            f"return_5d_pct: {self.return_5d_pct}",
            f"return_20d_pct: {self.return_20d_pct}",
            f"ema20_vs_ema50_pct: {self.ema20_vs_ema50_pct}",
            f"close_vs_ema20_pct: {self.close_vs_ema20_pct}",
            f"rsi14: {self.rsi14}",
            f"atr14_over_close_pct: {self.atr14_over_close_pct}",
            f"volatility_20d_pct: {self.volatility_20d_pct}",
            f"volume_vs_20d_pct: {self.volume_vs_20d_pct}",
            f"portfolio_exposure_pct: {self.portfolio_exposure_pct}",
        ]


def ratio_to_pct(value: Decimal) -> Decimal:
    return value * Decimal("100")
