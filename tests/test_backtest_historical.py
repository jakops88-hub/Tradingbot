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
            stop_loss_price=latest.close * Decimal("0.95")
            if action == SignalAction.BUY
            else None,
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
    assert result.ending_capital == Decimal("990.00000000")
    assert result.total_return == Decimal("-0.01000000")
    assert result.total_return_pct == Decimal("-1.00000000")
    assert result.strategy_return_pct == Decimal("-1.00000000")
    assert result.benchmark_return_pct == Decimal("-10.0")
    assert result.difference_vs_benchmark_pct == Decimal("9.00000000")
    assert result.benchmark_max_drawdown == Decimal("0.1818181818181818181818181818")
    assert result.total_trades == 4
    assert result.winning_trades == 0
    assert result.losing_trades == 1
    assert result.win_rate == Decimal("0")
    assert result.realized_pnl == Decimal("-10.00000000")
    assert result.gross_pnl == Decimal("-10.00000000")
    assert result.net_pnl == Decimal("-10.00000000")
    assert result.total_fees_paid == Decimal("0E-8")
    assert result.total_execution_costs == Decimal("0E-8")
    assert result.profit_factor == Decimal("0E+8")
    assert result.max_drawdown == Decimal("0.0198019801980198019801980198")
    assert result.average_position_value == Decimal("194.10000000")
    assert result.average_portfolio_exposure_pct == Decimal("19.500")
    assert result.largest_position_value == Decimal("210.00000000")
    assert result.maximum_portfolio_exposure_pct == Decimal("21.00")
    assert result.stop_loss_exits == 1
    assert result.average_monetary_risk_at_entry == Decimal("14.9500000000")
    assert result.equity_curve == [
        Decimal("1000"),
        Decimal("1000.00000000"),
        Decimal("1010.00000000"),
        Decimal("990.00000000"),
        Decimal("990.00000000"),
    ]
