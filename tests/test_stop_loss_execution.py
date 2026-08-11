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


class BuyAndHoldWithStopStrategy(Strategy):
    name = "buy_and_hold_with_stop"

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


def make_candle(index: int, close: str, low: str) -> Candle:
    price = Decimal(close)
    return Candle(
        symbol="ABC",
        timestamp=START + timedelta(days=index),
        open=price,
        high=price + Decimal("2"),
        low=Decimal(low),
        close=price,
        volume=Decimal("1000"),
    )


def test_stop_loss_execution_closes_position_when_candle_crosses_stop() -> None:
    engine = BacktestEngine(
        strategy=BuyAndHoldWithStopStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    )

    result = engine.run([
        make_candle(0, "100", "99"),
        make_candle(1, "98", "94"),
    ])

    assert result.total_trades == 2
    assert result.stop_loss_exits == 1
    assert result.open_positions == 0
    assert result.trade_log[-1].exit_reason == "stop_loss"
    assert result.trade_log[-1].price == Decimal("95")


def test_stop_loss_execution_applies_fees_and_sell_slippage() -> None:
    engine = BacktestEngine(
        strategy=BuyAndHoldWithStopStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(
            ExecutionCostConfig(
                percentage_fee=Decimal("0.01"),
                fixed_fee=Decimal("1"),
                slippage_percentage=Decimal("0.01"),
            )
        ),
        starting_cash=Decimal("1000"),
    )

    result = engine.run([
        make_candle(0, "100", "99"),
        make_candle(1, "98", "94"),
    ])

    assert result.stop_loss_exits == 1
    assert result.trade_log[-1].price == Decimal("94.05")
    assert result.trade_log[-1].commission > 0
    assert result.total_execution_costs > result.total_fees_paid


def test_position_cannot_exceed_configured_exposure() -> None:
    engine = BacktestEngine(
        strategy=BuyAndHoldWithStopStrategy(),
        risk_profile=get_risk_profile(RiskMode.LOW),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    )

    result = engine.run([make_candle(0, "100", "99")])

    assert result.largest_position_value <= Decimal("300")
    assert result.maximum_portfolio_exposure_pct <= Decimal("30")
