from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.strategies.base import Strategy


START = datetime(2024, 1, 1)


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
            stop_loss_price=Decimal("95") if action == SignalAction.BUY else None,
        )


class BuyFirstHoldStrategy(Strategy):
    name = "buy_first_hold"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        action = SignalAction.BUY if len(candles) == 1 else SignalAction.HOLD
        return Signal(
            symbol=candles[-1].symbol,
            action=action,
            generated_at=candles[-1].timestamp,
            stop_loss_price=Decimal("95") if action == SignalAction.BUY else None,
        )


class HoldStrategy(Strategy):
    name = "hold"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        return Signal(candles[-1].symbol, SignalAction.HOLD, candles[-1].timestamp)


class RecordingStrategy(Strategy):
    name = "recording"

    def __init__(self) -> None:
        self.observed_windows: list[list[datetime]] = []

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        self.observed_windows.append([candle.timestamp for candle in candles])
        action = SignalAction.BUY if len(candles) == 1 else SignalAction.HOLD
        return Signal(
            symbol=candles[-1].symbol,
            action=action,
            generated_at=candles[-1].timestamp,
            stop_loss_price=Decimal("95") if action == SignalAction.BUY else None,
        )


def make_candle(index: int, *, open_price: str, high: str, low: str, close: str) -> Candle:
    return Candle(
        symbol="ABC",
        timestamp=START + timedelta(days=index),
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def test_signal_on_candle_n_executes_on_candle_n_plus_one_open() -> None:
    result = BacktestEngine(
        strategy=BuyFirstHoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="90", high="101", low="89", close="100"),
            make_candle(1, open_price="111", high="120", low="100", close="120"),
        ]
    )

    assert result.total_trades == 1
    assert result.trade_log[0].executed_at == START + timedelta(days=1)
    assert result.trade_log[0].market_price == Decimal("111")
    assert result.trade_log[0].price == Decimal("111")


def test_no_same_bar_execution_at_signal_candle_close() -> None:
    result = BacktestEngine(
        strategy=BuyFirstHoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="90", high="101", low="89", close="100"),
            make_candle(1, open_price="110", high="112", low="100", close="105"),
        ]
    )

    assert result.trade_log[0].price != Decimal("100")
    assert result.equity_curve[0] == Decimal("1000")


def test_final_candle_signal_does_not_execute_without_next_candle() -> None:
    result = BacktestEngine(
        strategy=BuyFirstHoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run([make_candle(0, open_price="100", high="101", low="99", close="100")])

    assert result.total_trades == 0
    assert result.open_positions == 0


def test_gap_below_stop_executes_stop_at_open_price() -> None:
    result = BacktestEngine(
        strategy=BuyFirstHoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="100", high="101", low="99", close="100"),
            make_candle(1, open_price="100", high="101", low="99", close="100"),
            make_candle(2, open_price="90", high="92", low="89", close="91"),
        ]
    )

    assert result.stop_loss_exits == 1
    assert result.trade_log[-1].exit_reason == "stop_loss"
    assert result.trade_log[-1].market_price == Decimal("90")
    assert result.trade_log[-1].price == Decimal("90")


def test_intraday_stop_executes_at_stop_price() -> None:
    result = BacktestEngine(
        strategy=BuyFirstHoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="100", high="101", low="99", close="100"),
            make_candle(1, open_price="100", high="101", low="99", close="100"),
            make_candle(2, open_price="100", high="101", low="94", close="98"),
        ]
    )

    assert result.stop_loss_exits == 1
    assert result.trade_log[-1].market_price == Decimal("95")
    assert result.trade_log[-1].price == Decimal("95")


def test_fees_and_slippage_apply_to_stopped_trades() -> None:
    result = BacktestEngine(
        strategy=BuyFirstHoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(
            ExecutionCostConfig(
                percentage_fee=Decimal("0.01"),
                fixed_fee=Decimal("1"),
                slippage_percentage=Decimal("0.01"),
            )
        ),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="100", high="101", low="99", close="100"),
            make_candle(1, open_price="100", high="101", low="99", close="100"),
            make_candle(2, open_price="90", high="92", low="89", close="91"),
        ]
    )

    stopped_trade = result.trade_log[-1]
    assert stopped_trade.exit_reason == "stop_loss"
    assert stopped_trade.price == Decimal("89.10")
    assert stopped_trade.commission > 0
    assert stopped_trade.slippage_cost > 0
    assert result.total_execution_costs > result.total_fees_paid


def test_strategy_never_receives_future_candles() -> None:
    strategy = RecordingStrategy()

    result = BacktestEngine(
        strategy=strategy,
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="90", high="101", low="89", close="100"),
            make_candle(1, open_price="150", high="151", low="149", close="150"),
        ]
    )

    assert strategy.observed_windows == [
        [START],
        [START, START + timedelta(days=1)],
    ]
    assert result.trade_log[0].quantity == Decimal("2.00000000")


def test_benchmark_max_drawdown_uses_buy_and_hold_equity_curve() -> None:
    result = BacktestEngine(
        strategy=HoldStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(
        [
            make_candle(0, open_price="100", high="101", low="99", close="100"),
            make_candle(1, open_price="120", high="121", low="119", close="120"),
            make_candle(2, open_price="90", high="91", low="89", close="90"),
            make_candle(3, open_price="110", high="111", low="109", close="110"),
        ]
    )

    assert result.benchmark_return_pct == Decimal("10.0")
    assert result.benchmark_max_drawdown == Decimal("0.25")
