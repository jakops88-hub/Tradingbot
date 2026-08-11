"""Freshness checks for current-market daily candle scans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from trading_bot.data.models import Candle


MARKET_CLOSE_BUFFER = time(18, 30)
DEFAULT_STALENESS_TOLERANCE_TRADING_DAYS = 1


@dataclass(frozen=True)
class CandleFreshness:
    latest_expected_date: date
    latest_candle_timestamp: datetime
    data_age_trading_days: int
    current_close: Decimal
    is_stale: bool
    reason: str | None


def latest_expected_completed_daily_candle(now: datetime) -> date:
    candidate = now.date()
    if candidate.weekday() >= 5 or now.time() < MARKET_CLOSE_BUFFER:
        candidate -= timedelta(days=1)
    return previous_weekday(candidate)


def current_scan_download_end(now: datetime) -> datetime:
    return datetime.combine(now.date() + timedelta(days=1), time.min)


def freshness_for_candles(
    candles: list[Candle],
    *,
    now: datetime,
    tolerance_trading_days: int = DEFAULT_STALENESS_TOLERANCE_TRADING_DAYS,
) -> CandleFreshness:
    if not candles:
        raise ValueError("freshness check requires at least one candle")
    if tolerance_trading_days < 0:
        raise ValueError("tolerance_trading_days cannot be negative")

    latest_expected = latest_expected_completed_daily_candle(now)
    latest_candle = max(candles, key=lambda candle: candle.timestamp)
    latest_date = latest_candle.timestamp.date()
    if latest_date > latest_expected:
        return CandleFreshness(
            latest_expected_date=latest_expected,
            latest_candle_timestamp=latest_candle.timestamp,
            data_age_trading_days=0,
            current_close=latest_candle.close,
            is_stale=True,
            reason=(
                "latest candle is after the latest expected completed trading day "
                f"({latest_date.isoformat()} > {latest_expected.isoformat()})"
            ),
        )

    age = trading_day_gap(latest_date, latest_expected)
    is_stale = age > tolerance_trading_days
    return CandleFreshness(
        latest_expected_date=latest_expected,
        latest_candle_timestamp=latest_candle.timestamp,
        data_age_trading_days=age,
        current_close=latest_candle.close,
        is_stale=is_stale,
        reason=(
            "MARKET DATA STALE - candidate rejected before AI "
            f"(latest candle {latest_date.isoformat()}, expected {latest_expected.isoformat()}, "
            f"age {age} trading days, tolerance {tolerance_trading_days})"
            if is_stale
            else None
        ),
    )


def filter_completed_candles(candles: list[Candle], *, now: datetime) -> list[Candle]:
    latest_expected = latest_expected_completed_daily_candle(now)
    return [candle for candle in candles if candle.timestamp.date() <= latest_expected]


def trading_day_gap(start_date: date, end_date: date) -> int:
    if start_date >= end_date:
        return 0
    days = 0
    cursor = start_date + timedelta(days=1)
    while cursor <= end_date:
        if cursor.weekday() < 5:
            days += 1
        cursor += timedelta(days=1)
    return days


def previous_weekday(value: date) -> date:
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value
