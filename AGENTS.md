# TradingBot Agent Instructions

## Project Purpose

TradingBot is a Python 3.11+ foundation for an automated trading platform. It currently supports offline domain modeling, historical CSV loading, paper execution with configurable simulated costs, risk gates, portfolio accounting, metrics, and backtest orchestration. It must not place real trades.

## Architecture

Preserve this trading flow:

```text
Market Data -> Strategy -> Signal -> Risk Manager -> Order -> Broker -> Portfolio
```

Core boundaries:

- `data`: market-data provider interfaces, historical CSV loading, and typed market/domain models.
- `strategies`: strategy interface and offline strategies; strategies emit signals and never send orders.
- `risk`: risk profiles, position sizing, and signal-to-order approval.
- `execution`: broker interface and paper broker implementation.
- `portfolio`: cash, positions, realized PnL, and snapshots.
- `backtest`: orchestration of the full offline trading flow.
- `research`: independent multi-period and full-history historical evaluation.
- `metrics`: performance and goal tracking metrics.

## Coding Conventions

- Use strong typing and keep modules dependency-light.
- Use `Decimal` for money, prices, quantities, and percentage calculations where precision matters.
- Raise clear exceptions for invalid state; do not silently swallow errors.
- Keep live broker integrations, credentials, API keys, and real financial transactions out of the codebase.
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
