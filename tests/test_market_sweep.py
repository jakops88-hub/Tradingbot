from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.metadata import DatasetMetadata, save_dataset_metadata
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.research.market_sweep import MarketSweepEvaluator, load_symbol_config
from trading_bot.strategies.base import Strategy


START = datetime(2020, 1, 1)


class BuyFirstSellSecondStrategy(Strategy):
    name = "buy_first_sell_second"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        action = SignalAction.BUY if len(candles) == 1 else SignalAction.SELL if len(candles) == 2 else SignalAction.HOLD
        return Signal(
            symbol=candles[-1].symbol,
            action=action,
            generated_at=candles[-1].timestamp,
            stop_loss_price=candles[-1].close * Decimal("0.95") if action == SignalAction.BUY else None,
        )


def write_dataset(tmp_path: Path, symbol: str, prices: list[str], *, adjustment_policy: str = "adjusted") -> Path:
    csv_path = tmp_path / f"{symbol}_daily.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    for index, price_text in enumerate(prices):
        price = Decimal(price_text)
        timestamp = START + timedelta(days=index)
        rows.append(f"{timestamp.isoformat()},{price},{price + Decimal('1')},{price - Decimal('1')},{price},1000")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    save_dataset_metadata(
        csv_path,
        DatasetMetadata(
            symbol=symbol,
            quote_currency="SEK",
            source="test",
            interval="1d",
            start_date=START.date().isoformat(),
            end_date=(START + timedelta(days=len(prices) - 1)).date().isoformat(),
            adjustment_policy=adjustment_policy,
            auto_adjust=adjustment_policy == "adjusted",
            yfinance_repair=True,
            ohlc_normalization_policy="yahoo_rounding_tolerance",
            repaired_ohlc_rows=2,
            largest_repaired_ohlc_violation_pct="0.00000000000002",
        ),
    )
    return csv_path


def make_evaluator(fetcher) -> MarketSweepEvaluator:
    return MarketSweepEvaluator(
        dataset_fetcher=fetcher,
        strategy_factory=BuyFirstSellSecondStrategy,
        broker_factory=PaperBroker,
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        starting_capital=Decimal("1000"),
        portfolio_currency="SEK",
        expected_adjustment_policy="adjusted",
        start_date=START,
        end_date=START + timedelta(days=10),
    )


def test_portfolio_state_is_independent_between_symbols(tmp_path: Path) -> None:
    paths = {
        "AAA.ST": write_dataset(tmp_path, "AAA.ST", ["100", "110"]),
        "BBB.ST": write_dataset(tmp_path, "BBB.ST", ["100", "110"]),
    }

    report = make_evaluator(lambda symbol: paths[symbol]).evaluate(["AAA.ST", "BBB.ST"])

    assert [result.ending_capital for result in report.results] == [
        Decimal("1020.00000000"),
        Decimal("1020.00000000"),
    ]


def test_failed_symbol_does_not_kill_entire_sweep(tmp_path: Path) -> None:
    paths = {"AAA.ST": write_dataset(tmp_path, "AAA.ST", ["100", "110"])}

    def fetch(symbol: str) -> Path:
        if symbol == "FAIL.ST":
            raise RuntimeError("download failed")
        return paths[symbol]

    report = make_evaluator(fetch).evaluate(["AAA.ST", "FAIL.ST"])

    assert [result.symbol for result in report.results] == ["AAA.ST"]
    assert len(report.failures) == 1
    assert report.failures[0].symbol == "FAIL.ST"
    assert "download failed" in report.failures[0].reason


def test_market_sweep_aggregate_metrics_ranking_and_counts(tmp_path: Path) -> None:
    paths = {
        "WIN.ST": write_dataset(tmp_path, "WIN.ST", ["100", "110"]),
        "LOSE.ST": write_dataset(tmp_path, "LOSE.ST", ["100", "90"]),
        "BEAT.ST": write_dataset(tmp_path, "BEAT.ST", ["100", "80"]),
    }

    report = make_evaluator(lambda symbol: paths[symbol]).evaluate(["WIN.ST", "LOSE.ST", "BEAT.ST"])

    assert report.summary.profitable_instruments == 1
    assert report.summary.losing_instruments == 2
    assert report.summary.average_strategy_return_pct == Decimal("0")
    assert report.summary.median_strategy_return_pct == Decimal("-1.0000000000")
    assert report.summary.average_benchmark_return_pct == Decimal("-6.666666666666666666666666667")
    assert report.summary.average_max_drawdown == Decimal("0.006666666666666666666666666667")
    assert report.summary.best_strategy_instrument is not None
    assert report.summary.best_strategy_instrument.symbol == "WIN.ST"
    assert report.summary.worst_strategy_instrument is not None
    assert report.summary.worst_strategy_instrument.symbol == "LOSE.ST"
    assert report.summary.strategy_beats_buy_and_hold_count == 2
    assert [result.symbol for result in report.ranking] == ["WIN.ST", "LOSE.ST", "BEAT.ST"]
    assert report.results[0].candle_count == 2
    assert report.results[0].repaired_ohlc_rows == 2
    assert report.results[0].largest_repaired_ohlc_violation_pct == Decimal("0.00000000000002")
    assert report.results[0].data_quality_status == "PASS"


def test_adjusted_price_policy_metadata_is_required(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, "RAW.ST", ["100", "110"], adjustment_policy="unadjusted")

    report = make_evaluator(lambda symbol: path).evaluate(["RAW.ST"])

    assert report.results == []
    assert len(report.failures) == 1
    assert "adjustment policy mismatch" in report.failures[0].reason


def test_symbol_config_loader_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    config_path = tmp_path / "symbols.txt"
    config_path.write_text("# comment\n\nVOLV-B.ST\nERIC-B.ST\n", encoding="utf-8")

    assert load_symbol_config(config_path) == ["VOLV-B.ST", "ERIC-B.ST"]
