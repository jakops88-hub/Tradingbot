# TradingBot Agent Instructions

## Project Purpose

TradingBot is a Python 3.11+ foundation for an automated trading platform. It currently supports offline domain modeling, historical CSV loading, paper execution with configurable simulated costs, risk gates, portfolio accounting, metrics, backtest orchestration, local ML research, and advisory current-market scanning. It must not place real trades.

## Architecture

Preserve this trading flow:

```text
Market Data -> Strategy -> Signal -> Risk Manager -> Order -> Broker -> Portfolio
```

Core boundaries:

- `data`: market-data provider interfaces, historical CSV loading, optional external historical adapters, dataset metadata, and typed market/domain models.
- `strategies`: strategy interface and offline strategies; strategies emit signals and never send orders.
- `risk`: risk profiles, stop-loss based position sizing, and signal-to-order approval.
- `execution`: broker interface and paper broker implementation.
- `portfolio`: cash, positions, realized PnL, and snapshots.
- `backtest`: orchestration of the full offline trading flow.
- `research`: independent multi-period and full-history historical evaluation.
- `metrics`: performance and goal tracking metrics.
- `ml`: local deterministic feature, target, model, and walk-forward research code.
- `ai`: advisory current-market OpenAI analysis only; never part of historical backtests or execution.

## Coding Conventions

- Use strong typing and keep modules dependency-light.
- Use `Decimal` for money, prices, quantities, and percentage calculations where precision matters.
- Raise clear exceptions for invalid state; do not silently swallow errors.
- Keep live broker integrations, credentials, API keys, and real financial transactions out of the codebase.
- Keep external data adapters isolated from core backtest and research layers.
- Keep Yahoo-specific OHLC repairs inside `YahooFinanceDataProvider`; core `Candle` and CSV validation must remain strict.
- Backtest signals generated from candle `N` must execute no earlier than candle `N+1` open; do not reintroduce same-bar close fills.
- Historical ML research must stay local and deterministic; do not use OpenAI, neural networks, automatic tuning, or symbol identity as a predictive feature.
- OpenAI is allowed only for current-market advisory scans, must read `OPENAI_API_KEY` from the environment, and must never log credentials.
- OpenAI must not call brokers, create orders, bypass `RiskManager`, change stop-losses, change risk settings, increase exposure, or use leverage.
- Keep ML trade-aligned target generation tied to shared paper execution helpers so stop, fee, and slippage assumptions do not drift from backtests.
- Risk must never increase because a portfolio is behind a user goal; goals are tracking only.

## Test Command

```powershell
python -m pytest
```

## Definition Of Done

- Relevant offline tests are added or updated.
- `python -m pytest` passes after changes.
- Documentation remains accurate.
- The strategy-to-broker separation remains intact.
- Live trading is not added unless explicitly requested by the user.
