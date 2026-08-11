# TradingBot

TradingBot is a modular Python foundation for an automated trading platform. The current project intentionally stops at offline architecture: typed domain models, strategy and broker interfaces, risk gates, portfolio accounting, paper execution primitives, metrics, historical CSV loading, and backtest orchestration.

The current version cannot trade real money. It has no live broker integration, no API-key requirements, and no credential storage.

## Current Status

Implemented:

- `MarketDataProvider`, `Strategy`, and `Broker` abstractions
- Typed models for candles, signals, orders, trades, positions, and portfolio snapshots
- BUY, SELL, and HOLD signal support
- LOW, MEDIUM, and HIGH risk profile presets
- Decimal-based accounting for cash, prices, quantities, and PnL
- Paper broker that returns simulated trade records only
- Portfolio accounting and offline historical backtest orchestration
- CSV-based historical OHLCV loading with validation
- Backtest results with return, drawdown, trade counts, win rate, realized PnL, profit factor, and equity curve
- Configurable offline execution costs: percentage fee, fixed fee, and slippage
- Buy-and-hold benchmark return comparison for the same historical period
- Offline pytest coverage for foundational components

Not implemented yet:

- Live trading
- Real broker integrations
- A profitable strategy
- Dashboard or web UI

## Architecture

```text
Market Data
-> Strategy
-> Signal
-> Risk Manager
-> Order
-> Broker
-> Portfolio
```

Strategies only produce signals. The risk manager is responsible for converting approved signals into orders. Brokers execute orders and return trade records. Portfolio accounting applies those trades.

## Risk Modes

| Mode | Risk/Trade | Max Exposure | Max Drawdown | Max Open Positions |
| --- | ---: | ---: | ---: | ---: |
| LOW | 0.5% | 30% | 8% | 2 |
| MEDIUM | 1% | 60% | 15% | 4 |
| HIGH | 2% | 100% | 25% | 6 |

Leverage is not allowed. Future portfolio goals, such as tracking progress from 1000 SEK to 10000 SEK, are for metrics only and must not increase risk limits.

## Windows Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `py -3.11` is unavailable, use a Python 3.11+ interpreter already installed on your PATH:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run Tests

```powershell
python -m pytest
```

## CLI Check

```powershell
trading-bot
```

The CLI runs a deterministic offline demo using `tests/fixtures/sample_ohlcv.csv`, starting capital of `1000 SEK`, the `MEDIUM` risk profile, and a simple threshold strategy used only to verify the pipeline.
