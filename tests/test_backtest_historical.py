from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.risk_profiles import RiskMode, get_risk_profile
from trading_bot.data.market_data import HistoricalDataProvider
from trading_bot.data.models import Candle, PortfolioSnapshot, Signal, SignalAction
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.strategies.base import Strategy


FIXTURES = Path(__file__).parent / "fixtures"


class ThresholdFixtureStrategy(Strategy):
    name = "threshold_fixture"

    def generate_signal(
        self,
        candles: Sequence[Candle],
        snapshot: PortfolioSnapshot,
    ) -> Signal:
        latest = candles[-1]
        if snapshot.open_positions == 0 and latest.close == Decimal("100"):
            action = SignalAction.BUY
        elif snapshot.open_positions > 0 and latest.close in {Decimal("110"), Decimal("90")}:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        return Signal(
            symbol=latest.symbol,
            action=action,
            generated_at=latest.timestamp,
            reason="fixture threshold",
        )


def test_complete_historical_backtest_flow_records_expected_metrics() -> None:
    provider = HistoricalDataProvider(FIXTURES / "sample_ohlcv.csv")
    candles = list(
        provider.historical_candles(
            "ABC",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )
    )
    engine = BacktestEngine(
        strategy=ThresholdFixtureStrategy(),
        risk_profile=get_risk_profile(RiskMode.MEDIUM),
        broker=PaperBroker(),
        starting_cash=Decimal("1000"),
    )

    result = engine.run(candles)

    assert result.starting_capital == Decimal("1000")
    assert result.ending_capital == Decimal("999.99900000")
    assert result.total_return == Decimal("-0.00000100")
    assert result.total_return_pct == Decimal("-0.00010000")
    assert result.total_trades == 4
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert result.win_rate == Decimal("0.5")
    assert result.realized_pnl == Decimal("-0.00100000")
    assert result.profit_factor == Decimal("0.9990009990009990009990009990")
    assert result.max_drawdown == Decimal("-0.001")
    assert result.equity_curve == [
        Decimal("1000.00000000"),
        Decimal("1000.50000000"),
        Decimal("1001.00000000"),
        Decimal("1001.00000000"),
        Decimal("999.99900000"),
    ]
