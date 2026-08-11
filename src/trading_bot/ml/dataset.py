"""Labeled ML dataset construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from trading_bot.data.models import Candle, Order, OrderSide
from trading_bot.execution.costs import ExecutionCostConfig
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.execution.simulation import long_stop_market_price
from trading_bot.ml.features import FeatureRow, build_feature_rows


TARGET_HORIZON_TRADING_DAYS = 10
TARGET_STOP_LOSS_PCT = Decimal("0.05")


class MLTargetMode(str, Enum):
    RAW_RETURN = "raw_return"
    TRADE_ALIGNED = "trade_aligned"


@dataclass(frozen=True)
class TradeOutcome:
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    net_pnl: Decimal
    net_return: Decimal
    exit_reason: str


@dataclass(frozen=True)
class MLSample:
    symbol: str
    feature_time: datetime
    entry_time: datetime
    exit_time: datetime
    features: tuple[Decimal, ...]
    target: int
    target_return: Decimal
    net_trade_return: Decimal | None = None
    exit_reason: str = "raw_horizon"
    actual_exit_time: datetime | None = None

    def feature_floats(self) -> list[float]:
        return [float(value) for value in self.features]


def build_labeled_samples(
    candles: Sequence[Candle],
    *,
    horizon: int = TARGET_HORIZON_TRADING_DAYS,
    target_mode: MLTargetMode = MLTargetMode.RAW_RETURN,
    cost_config: ExecutionCostConfig | None = None,
    stop_loss_pct: Decimal = TARGET_STOP_LOSS_PCT,
) -> list[MLSample]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not Decimal("0") < stop_loss_pct < Decimal("1"):
        raise ValueError("stop_loss_pct must be between 0 and 1")
    sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
    feature_rows = build_feature_rows(sorted_candles)
    samples: list[MLSample] = []
    for feature_row in feature_rows:
        entry_index = feature_row.candle_index + 1
        exit_index = entry_index + horizon
        if exit_index >= len(sorted_candles):
            continue
        entry_price = sorted_candles[entry_index].open
        exit_price = sorted_candles[exit_index].open
        raw_return = (exit_price / entry_price) - Decimal("1")
        trade_outcome = simulate_trade_outcome(
            sorted_candles,
            feature_row.candle_index,
            horizon=horizon,
            cost_config=cost_config,
            stop_loss_pct=stop_loss_pct,
        )
        if target_mode == MLTargetMode.RAW_RETURN:
            target = 1 if raw_return > 0 else 0
            target_return = raw_return
        elif target_mode == MLTargetMode.TRADE_ALIGNED:
            target = 1 if trade_outcome.net_pnl > 0 else 0
            target_return = trade_outcome.net_return
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
        samples.append(
            MLSample(
                symbol=feature_row.symbol,
                feature_time=feature_row.timestamp,
                entry_time=sorted_candles[entry_index].timestamp,
                exit_time=sorted_candles[exit_index].timestamp,
                features=feature_row.values,
                target=target,
                target_return=target_return,
                net_trade_return=trade_outcome.net_return,
                exit_reason=trade_outcome.exit_reason,
                actual_exit_time=trade_outcome.exit_time,
            )
        )
    return samples


def simulate_trade_outcome(
    candles: Sequence[Candle],
    feature_index: int,
    *,
    horizon: int = TARGET_HORIZON_TRADING_DAYS,
    cost_config: ExecutionCostConfig | None = None,
    stop_loss_pct: Decimal = TARGET_STOP_LOSS_PCT,
) -> TradeOutcome:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    sorted_candles = sorted(candles, key=lambda candle: candle.timestamp)
    entry_index = feature_index + 1
    max_exit_index = entry_index + horizon
    if feature_index < 0 or max_exit_index >= len(sorted_candles):
        raise ValueError("not enough future candles for trade outcome")

    broker = PaperBroker(cost_config or ExecutionCostConfig())
    entry_candle = sorted_candles[entry_index]
    entry_order = Order(
        symbol=entry_candle.symbol,
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        created_at=entry_candle.timestamp,
        stop_loss_pct=stop_loss_pct,
    )
    entry_trade = broker.submit_order(entry_order, entry_candle.open)
    if entry_trade.stop_loss_price is None:
        raise ValueError("entry trade did not produce a stop loss")

    exit_market_price: Decimal | None = None
    exit_candle = sorted_candles[max_exit_index]
    exit_reason = "max_hold"
    for candle in sorted_candles[entry_index:max_exit_index]:
        stop_market_price = long_stop_market_price(candle, entry_trade.stop_loss_price)
        if stop_market_price is None:
            continue
        exit_market_price = stop_market_price
        exit_candle = candle
        exit_reason = "stop_loss"
        break
    if exit_market_price is None:
        exit_market_price = exit_candle.open

    exit_order = Order(
        symbol=exit_candle.symbol,
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        created_at=exit_candle.timestamp,
        exit_reason=exit_reason,
    )
    exit_trade = broker.submit_order(exit_order, exit_market_price)
    net_pnl = exit_trade.cash_effect + entry_trade.cash_effect
    return TradeOutcome(
        entry_time=entry_trade.executed_at,
        exit_time=exit_trade.executed_at,
        entry_price=entry_trade.price,
        exit_price=exit_trade.price,
        net_pnl=net_pnl,
        net_return=net_pnl / entry_trade.gross_value,
        exit_reason=exit_reason,
    )


def feature_row_to_sample_features(feature_row: FeatureRow) -> tuple[Decimal, ...]:
    return feature_row.values
