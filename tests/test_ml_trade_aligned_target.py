from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.ml.dataset import MLTargetMode, build_labeled_samples, simulate_trade_outcome
from trading_bot.strategies.base import Strategy


START = datetime(2024, 1, 1)


class SingleTradeTenDayStrategy(Strategy):
    name = "single_trade_ten_day"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        if len(candles) == 1:
            return Signal(
                symbol=candles[-1].symbol,
                action=SignalAction.BUY,
                generated_at=candles[-1].timestamp,
                stop_loss_price=candles[-1].close * Decimal("0.95"),
                stop_loss_pct=Decimal("0.05"),
            )
        if snapshot.open_positions > 0 and len(candles) == 11:
            return Signal(candles[-1].symbol, SignalAction.SELL, candles[-1].timestamp)
        return Signal(candles[-1].symbol, SignalAction.HOLD, candles[-1].timestamp)


def make_candles(opens: list[str], *, low_override: dict[int, str] | None = None) -> list[Candle]:
    low_override = low_override or {}
    candles: list[Candle] = []
    for index, open_text in enumerate(opens):
        open_price = Decimal(open_text)
        close = open_price
        low = Decimal(low_override[index]) if index in low_override else open_price - Decimal("1")
        candles.append(
            Candle(
                symbol="ABC",
                timestamp=START + timedelta(days=index),
                open=open_price,
                high=open_price + Decimal("2"),
                low=low,
                close=close,
                volume=Decimal("1000") + Decimal(index),
            )
        )
    return candles


def warmup_candles(entry_open: str, exit_open: str, *, low_override: dict[int, str] | None = None) -> list[Candle]:
    opens = ["100"] * 61
    opens[50] = entry_open
    opens[60] = exit_open
    return make_candles(opens, low_override=low_override)


def test_trade_aligned_target_labels_profitable_normal_exit() -> None:
    samples = build_labeled_samples(
        warmup_candles("100", "110"),
        target_mode=MLTargetMode.TRADE_ALIGNED,
    )

    assert samples[0].target == 1
    assert samples[0].exit_reason == "max_hold"
    assert samples[0].entry_time == START + timedelta(days=50)
    assert samples[0].exit_time == START + timedelta(days=60)


def test_trade_aligned_target_labels_losing_normal_exit() -> None:
    samples = build_labeled_samples(
        warmup_candles("100", "99"),
        target_mode=MLTargetMode.TRADE_ALIGNED,
    )

    assert samples[0].target == 0
    assert samples[0].exit_reason == "max_hold"
    assert samples[0].target_return < 0


def test_trade_aligned_target_labels_intraday_stop() -> None:
    candles = make_candles(["100"] * 12, low_override={2: "94"})
    outcome = simulate_trade_outcome(candles, 0)

    assert outcome.exit_reason == "stop_loss"
    assert outcome.exit_time == START + timedelta(days=2)
    assert outcome.exit_price == Decimal("95.00")
    assert outcome.net_pnl < 0


def test_trade_aligned_target_labels_gap_through_stop() -> None:
    candles = make_candles(["100", "100", "90"] + ["100"] * 9, low_override={2: "89"})
    outcome = simulate_trade_outcome(candles, 0)

    assert outcome.exit_reason == "stop_loss"
    assert outcome.exit_time == START + timedelta(days=2)
    assert outcome.exit_price == Decimal("90")
    assert outcome.net_pnl == Decimal("-10")


def test_fees_can_turn_gross_profit_into_net_loss() -> None:
    candles = make_candles(["100"] + ["100"] * 10 + ["100.1"])
    outcome = simulate_trade_outcome(
        candles,
        0,
        cost_config=ExecutionCostConfig(fixed_fee=Decimal("0.20")),
    )

    assert outcome.exit_reason == "max_hold"
    assert outcome.exit_price == Decimal("100.1")
    assert outcome.net_pnl == Decimal("-0.30")
    assert outcome.net_pnl < 0


def test_trade_aligned_target_includes_slippage() -> None:
    candles = make_candles(["100"] + ["100"] * 10 + ["110"])
    outcome = simulate_trade_outcome(
        candles,
        0,
        cost_config=ExecutionCostConfig(slippage_percentage=Decimal("0.01")),
    )

    assert outcome.entry_price == Decimal("101.00")
    assert outcome.exit_price == Decimal("108.90")
    assert outcome.net_pnl == Decimal("7.90")


def test_target_holding_period_matches_ml_strategy_exit_open() -> None:
    candles = make_candles(["100"] + ["100"] * 10 + ["110"])
    outcome = simulate_trade_outcome(candles, 0)

    assert outcome.entry_time == START + timedelta(days=1)
    assert outcome.exit_time == START + timedelta(days=11)
    assert outcome.exit_reason == "max_hold"


def test_target_execution_rules_match_backtest_engine() -> None:
    candles = make_candles(["100"] + ["100"] * 10 + ["110"])
    costs = ExecutionCostConfig(
        percentage_fee=Decimal("0.001"),
        slippage_percentage=Decimal("0.001"),
    )
    outcome = simulate_trade_outcome(candles, 0, cost_config=costs)
    result = BacktestEngine(
        strategy=SingleTradeTenDayStrategy(),
        risk_profile=get_risk_profile(RiskMode.HIGH),
        broker=PaperBroker(costs),
        starting_cash=Decimal("1000"),
    ).run(candles)

    engine_entry = result.trade_log[0]
    engine_exit = result.trade_log[-1]
    engine_net_pnl_per_unit = (
        engine_exit.cash_effect + engine_entry.cash_effect
    ) / engine_entry.quantity

    assert outcome.entry_price == engine_entry.price
    assert outcome.exit_price == engine_exit.price
    assert outcome.net_pnl == engine_net_pnl_per_unit
