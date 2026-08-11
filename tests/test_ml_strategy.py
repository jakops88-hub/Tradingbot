from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.models import Candle
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.strategies.ml_decision import MLDecisionStrategy


START = datetime(2024, 1, 1)


class AlwaysBuyModel:
    def predict_positive_probability(self, features) -> float:
        return 0.9


def make_candles(count: int) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        close = Decimal("100") + Decimal(index % 5)
        open_price = close
        if index == 50:
            open_price = Decimal("123")
            close = Decimal("123")
        candles.append(
            Candle(
                symbol="ABC",
                timestamp=START + timedelta(days=index),
                open=open_price,
                high=max(open_price, close) + Decimal("2"),
                low=min(open_price, close) - Decimal("2"),
                close=close,
                volume=Decimal("1000") + Decimal(index),
            )
        )
    return candles


def test_ml_signal_executes_on_next_candle_open() -> None:
    result = BacktestEngine(
        strategy=MLDecisionStrategy(AlwaysBuyModel()),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    ).run(make_candles(52))

    assert result.total_trades == 1
    assert result.trade_log[0].executed_at == START + timedelta(days=50)
    assert result.trade_log[0].market_price == Decimal("123")
    assert result.trade_log[0].price == Decimal("123")
