from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.strategies.ema_trend import EMATrendConfig, EMATrendStrategy


START = datetime(2024, 1, 1)


def make_candles(prices: list[str]) -> list[Candle]:
    return [
        Candle(
            symbol="ABC",
            timestamp=START + timedelta(days=index),
            open=Decimal(price),
            high=Decimal(price) + Decimal("1"),
            low=Decimal(price) - Decimal("1"),
            close=Decimal(price),
            volume=Decimal("1000"),
        )
        for index, price in enumerate(prices)
    ]


def test_complete_ema_strategy_backtest_uses_existing_flow() -> None:
    engine = BacktestEngine(
        strategy=EMATrendStrategy(EMATrendConfig(fast_period=2, slow_period=3)),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(ExecutionCostConfig(fixed_fee=Decimal("0.01"))),
        starting_cash=Decimal("1000"),
        close_open_positions=True,
    )

    result = engine.run(make_candles(["10", "9", "8", "12", "13", "12", "8"]))

    assert result.total_trades >= 2
    assert result.total_fees_paid > 0
    assert result.open_positions == 0
    assert result.unrealized_pnl == Decimal("0")
    assert len(result.equity_curve) == 7


def test_open_positions_are_marked_to_market_by_default() -> None:
    engine = BacktestEngine(
        strategy=EMATrendStrategy(EMATrendConfig(fast_period=2, slow_period=3)),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    )

    result = engine.run(make_candles(["10", "9", "8", "12", "13", "14"]))

    assert result.total_trades == 1
    assert result.open_positions == 1
    assert result.positions_value > 0
    assert result.unrealized_pnl > 0


def test_open_positions_can_be_closed_on_final_candle() -> None:
    engine = BacktestEngine(
        strategy=EMATrendStrategy(EMATrendConfig(fast_period=2, slow_period=3)),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
        close_open_positions=True,
    )

    result = engine.run(make_candles(["10", "9", "8", "12", "13"]))

    assert result.total_trades == 2
    assert result.open_positions == 0
    assert result.positions_value == Decimal("0")
    assert result.unrealized_pnl == Decimal("0")
