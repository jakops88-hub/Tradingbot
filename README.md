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
- Long-only EMA crossover strategy for historical simulation
- Optional final-candle closing of open backtest positions
- Historical research reports across independent yearly periods and full-history runs
- Optional Yahoo Finance historical data download adapter
- Normalized CSV cache files with sidecar dataset metadata
- Risk-based position sizing using entry price, stop-loss price, and risk per trade
- Stop-loss execution in historical backtests with fees and slippage applied
- Research diagnostics for position value, exposure, stop-loss exits, and monetary risk at entry
- One-click locked EMA 20/50 multi-market sweep for Swedish large-cap symbols
- Local ML Decision Engine v1 research using `StandardScaler -> LogisticRegression`
- Offline pytest coverage for foundational components

Not implemented yet:

- Live trading
- Real broker integrations
- Strategy optimization or profitability claims
- Neural networks, LLM APIs, or automatic parameter tuning
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

Install the optional market-data adapter when you want to download Yahoo Finance history:

```powershell
python -m pip install -e ".[market-data]"
```

## Run Tests

```powershell
python -m pytest
```

## CLI Check

```powershell
trading-bot
```

The CLI runs a deterministic offline EMA 20/50 historical simulation with starting capital of `1000 SEK`, the `MEDIUM` risk profile, and example execution costs. It is not evidence of future profitability.

## One-Click Research

On Windows, double-click:

```text
run_research.bat
```

The script creates or reuses `.venv`, installs required market-data dependencies, downloads `VOLV-B.ST` daily data from `2018-01-01` to `2026-01-01`, runs EMA 20/50 yearly and full-history research, and saves the report to:

```text
reports/latest_research.txt
```

The terminal window stays open when the run completes or fails, so errors can be read.

Run the locked Swedish market sweep by double-clicking:

```text
run_market_sweep.bat
```

The sweep uses `config/swedish_large_caps.txt`, downloads adjusted daily OHLCV data, applies the same EMA 20/50 strategy to every instrument independently, and saves the report to:

```text
reports/market_sweep.txt
```

Locked sweep settings are `1000 SEK`, `MEDIUM` risk, `0.1%` percentage fee, `0.1%` slippage, `5%` initial stop loss, adjusted prices, and daily candles from `2018-01-01` to `2026-01-01`.

Yahoo Finance data uses `auto_adjust=True` and explicitly sets `repair=False` because yfinance's repair path may require optional SciPy support. TradingBot applies its own Yahoo-only OHLC ordering repair for tiny adjusted-price rounding artifacts up to `0.000001%` of price. Core `Candle` validation and local CSV validation remain strict. The market sweep report includes a Data Quality section with candle counts, repaired OHLC rows, largest repaired violation, and pass/fail status per symbol.

Run the locked ML Decision Engine v1 walk-forward research by double-clicking:

```text
run_ml_research.bat
```

The ML runner downloads or refreshes the same ten Swedish symbols, pools them for training, evaluates out-of-sample yearly folds, and saves the report to:

```text
reports/ml_research.txt
```

ML v1 uses only local scikit-learn components: `StandardScaler -> LogisticRegression`. Features are built from data available at candle close: 1-day, 5-day, and 20-day returns; EMA20 vs EMA50; close vs EMA20; RSI14; ATR14 divided by close; 20-day volatility; and volume versus 20-day average. The target is positive return for a trade entered at candle `N+1` open and exited at candle `N+11` open. Symbol identity is not used as a feature.

## Backtest Execution Timing

Daily strategy signals are generated only after a candle has completed. If candle `N` creates a BUY or SELL signal, the approved order executes no earlier than candle `N+1` open. A signal on the final candle is not executed because no next open exists.

Long stop-loss exits are conservative for OHLC data: if the next candle opens below the stop, the stop exits at that opening price; otherwise, if the candle low reaches the stop, the stop exits at the stop price. Existing fees and slippage are still applied by the paper broker.

Run EMA 20/50 research against a local OHLCV CSV:

```powershell
python -m trading_bot.app research .\data.csv --symbol ABC --risk MEDIUM
```

The research command starts each yearly period with a fresh `1000 SEK` portfolio and also prints a separate continuous full-history result.

Download normalized daily OHLCV data from Yahoo Finance:

```powershell
python -m trading_bot.app download VOLV-B.ST --start 2018-01-01 --end 2026-01-01
```

Downloaded files are cached under `data/`, for example `data/VOLV-B.ST_daily.csv`, with metadata stored beside it as `data/VOLV-B.ST_daily.csv.metadata.json`.

Run research against downloaded data:

```powershell
python -m trading_bot.app research .\data\VOLV-B.ST_daily.csv --symbol VOLV-B.ST --risk MEDIUM
```

If dataset metadata says the instrument quote currency differs from the portfolio currency, the command fails clearly. FX conversion is not implemented.
