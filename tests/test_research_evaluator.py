from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import pytest

from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.research.evaluator import ResearchEvaluator, yearly_periods
from trading_bot.research.models import ResearchPeriod
from trading_bot.strategies.base import Strategy


class BuyFirstSellSecondStrategy(Strategy):
    name = "buy_first_sell_second"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        if len(candles) == 1:
            action = SignalAction.BUY
        elif len(candles) == 2:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
        return Signal(candles[-1].symbol, action, candles[-1].timestamp)


def make_candle(timestamp: datetime, close: str) -> Candle:
    price = Decimal(close)
    return Candle(
        symbol="ABC",
        timestamp=timestamp,
        open=price,
        high=price + Decimal("1"),
        low=price - Decimal("1"),
        close=price,
        volume=Decimal("1000"),
    )


def research_candles() -> list[Candle]:
    return [
        make_candle(datetime(2020, 1, 1), "100"),
        make_candle(datetime(2020, 1, 2), "110"),
        make_candle(datetime(2021, 1, 1), "100"),
        make_candle(datetime(2021, 1, 2), "90"),
    ]


def make_evaluator() -> ResearchEvaluator:
    return ResearchEvaluator(
        strategy_factory=BuyFirstSellSecondStrategy,
        broker_factory=PaperBroker,
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        starting_capital=Decimal("1000"),
        close_open_positions=True,
    )


def test_yearly_periods_cover_dataset_years() -> None:
    periods = yearly_periods(research_candles())

    assert [period.label for period in periods] == ["2020", "2021"]
    assert periods[0].start == datetime(2020, 1, 1)
    assert periods[1].end == datetime(2021, 12, 31, 23, 59, 59, 999999)


def test_independent_yearly_periods_start_with_fresh_capital() -> None:
    report = make_evaluator().evaluate(research_candles(), yearly_periods(research_candles()))

    assert [period.starting_capital for period in report.period_results] == [
        Decimal("1000"),
        Decimal("1000"),
    ]
    assert [period.ending_capital for period in report.period_results] == [
        Decimal("1001.0"),
        Decimal("999.0"),
    ]
    assert [period.total_trades for period in report.period_results] == [2, 2]


def test_research_aggregate_metrics_and_best_worst_periods() -> None:
    report = make_evaluator().evaluate(research_candles(), yearly_periods(research_candles()))
    aggregate = report.aggregate

    assert aggregate.average_strategy_return_pct == Decimal("0.0")
    assert aggregate.median_strategy_return_pct == Decimal("0.0")
    assert aggregate.average_benchmark_return_pct == Decimal("0.0")
    assert aggregate.profitable_periods == 1
    assert aggregate.losing_periods == 1
    assert aggregate.best_period is not None
    assert aggregate.best_period.period.label == "2020"
    assert aggregate.worst_period is not None
    assert aggregate.worst_period.period.label == "2021"
    assert aggregate.average_max_drawdown == Decimal("0.0005")
    assert aggregate.total_trades == 4


def test_full_history_evaluation_is_distinct_from_independent_periods() -> None:
    report = make_evaluator().evaluate(research_candles(), yearly_periods(research_candles()))

    assert report.full_history.period.label == "Full History"
    assert report.full_history.total_trades == 2
    assert report.full_history.ending_capital == Decimal("1001.0")
    assert sum(period.total_trades for period in report.period_results) == 4


def test_empty_filtered_period_is_skipped_clearly() -> None:
    empty_period = ResearchPeriod(
        start=datetime(2022, 1, 1),
        end=datetime(2022, 12, 31, 23, 59, 59),
        label="2022",
    )

    report = make_evaluator().evaluate(research_candles(), [empty_period])

    assert report.period_results == []
    assert len(report.skipped_periods) == 1
    assert report.skipped_periods[0].reason == "no candles in period"


def test_duplicate_timestamps_are_rejected_by_research_evaluator() -> None:
    candles = [
        make_candle(datetime(2020, 1, 1), "100"),
        make_candle(datetime(2020, 1, 1), "101"),
    ]

    with pytest.raises(ValueError, match="duplicate candle timestamps"):
        make_evaluator().evaluate(candles, yearly_periods(candles))
