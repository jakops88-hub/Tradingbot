from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.strategies.base import Strategy


class ScriptedStrategy(Strategy):
    name = "scripted"

    def __init__(self, actions: list[SignalAction]) -> None:
        self.actions = actions

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        index = len(candles) - 1
        action = self.actions[index] if index < len(self.actions) else SignalAction.HOLD
        return Signal(
            symbol=candles[-1].symbol,
            action=action,
            generated_at=candles[-1].timestamp,
            stop_loss_price=candles[-1].close * Decimal("0.95")
            if action == SignalAction.BUY
            else None,
        )


def make_candles(prices: list[str]) -> list[Candle]:
    start = datetime(2024, 1, 1)
    return [
        Candle(
            symbol="ABC",
            timestamp=start + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price) + Decimal("1"),
            low=Decimal(price) - Decimal("1"),
            close=Decimal(price),
            volume=Decimal("1000"),
        )
        for index, price in enumerate(prices)
    ]


def test_backtest_preserves_signal_risk_broker_portfolio_flow() -> None:
    engine = BacktestEngine(
        strategy=ScriptedStrategy([SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL, SignalAction.HOLD]),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    )

    result = engine.run(make_candles(["100", "110", "120", "130"]))

    assert result.trades == 2
    assert len(result.equity_curve) == 4
    assert result.ending_equity == Decimal("1040.00000000")
