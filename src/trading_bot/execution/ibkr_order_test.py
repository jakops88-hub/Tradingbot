"""Strictly controlled IBKR paper order round-trip test.

This module is intentionally separate from the read-only broker adapter,
research, AI scanning, and dashboard layers. It can submit at most one
BUY and one SELL in explicit test mode against the local IBKR paper endpoint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from trading_bot.execution.ibkr_broker import (
    DEFAULT_CONTRACT_CONFIG,
    DEFAULT_IBKR_CLIENT_ID,
    DEFAULT_IBKR_HOST,
    DEFAULT_IBKR_PAPER_PORT,
    IbkrAccountSnapshot,
    IbkrConnectionConfig,
    IbkrContractSpec,
    IbkrOpenOrder,
    IbkrPosition,
    IbkrResolvedContract,
    ReconciliationResult,
    _account_snapshot,
    _decimal_or_none,
    _normalized_local_symbol,
    _open_order,
    _position,
    _resolve_contract,
    _value,
    account_id_hash,
    load_contract_specs,
    mask_account_id,
    reconcile_positions,
)
from trading_bot.persistence.sqlite_store import DEFAULT_DATABASE_PATH, TradingBotSQLiteStore


DEFAULT_ORDER_TEST_SYMBOL = "ERIC-B.ST"
MAX_ORDER_TEST_QUANTITY = Decimal("1")
ORDER_TEST_ORDER_TYPE = "MKT"
ORDER_TEST_TIF = "DAY"


@dataclass(frozen=True)
class IbkrOrderTestConfig:
    host: str = DEFAULT_IBKR_HOST
    port: int = DEFAULT_IBKR_PAPER_PORT
    client_id: int = DEFAULT_IBKR_CLIENT_ID
    timeout_seconds: int = 60
    symbol: str = DEFAULT_ORDER_TEST_SYMBOL
    quantity: Decimal = MAX_ORDER_TEST_QUANTITY
    explicit_test_mode: bool = False
    contracts_path: Path = DEFAULT_CONTRACT_CONFIG
    database_path: Path = DEFAULT_DATABASE_PATH


@dataclass(frozen=True)
class IbkrOrderStatusChange:
    timestamp: datetime
    status: str
    filled_quantity: Decimal
    remaining_quantity: Decimal | None
    average_fill_price: Decimal | None


@dataclass(frozen=True)
class IbkrOrderFillResult:
    order_id: int
    final_status: str
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    status_changes: list[IbkrOrderStatusChange]
    error_messages: list[str] = field(default_factory=list)
    timed_out: bool = False


@dataclass(frozen=True)
class IbkrOrderRoundTripResult:
    passed: bool
    started_at: datetime
    endpoint: str
    client_id: int
    account_environment: str = "UNKNOWN"
    account_id_masked: str = "n/a"
    contract: IbkrResolvedContract | None = None
    buy_order_type: str = ORDER_TEST_ORDER_TYPE
    buy_tif: str = ORDER_TEST_TIF
    buy_order_id: int | None = None
    buy_status: str | None = None
    buy_fill_quantity: Decimal = Decimal("0")
    buy_average_fill_price: Decimal | None = None
    buy_status_changes: list[IbkrOrderStatusChange] = field(default_factory=list)
    buy_error_messages: list[str] = field(default_factory=list)
    sell_order_type: str = ORDER_TEST_ORDER_TYPE
    sell_tif: str = ORDER_TEST_TIF
    sell_order_id: int | None = None
    sell_status: str | None = None
    sell_fill_quantity: Decimal = Decimal("0")
    sell_average_fill_price: Decimal | None = None
    sell_status_changes: list[IbkrOrderStatusChange] = field(default_factory=list)
    sell_error_messages: list[str] = field(default_factory=list)
    paper_pnl: Decimal | None = None
    final_position_quantity: Decimal | None = None
    reconciliation: ReconciliationResult | None = None
    cancelled_buy_order: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class IbkrOrderTestClientBoundary(Protocol):
    def connect(self, config: IbkrConnectionConfig) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def account_values(self) -> list[Any]: ...
    def positions(self) -> list[Any]: ...
    def open_orders(self) -> list[Any]: ...
    def resolve_contract(self, spec: IbkrContractSpec) -> Any: ...
    def place_market_order(
        self,
        contract: Any,
        action: str,
        quantity: Decimal,
        *,
        order_type: str = ORDER_TEST_ORDER_TYPE,
        tif: str = ORDER_TEST_TIF,
    ) -> int: ...
    def wait_for_fill(self, order_id: int, timeout_seconds: int) -> IbkrOrderFillResult: ...
    def cancel_order(self, order_id: int) -> None: ...


class IbInsyncPaperOrderClient:
    def __init__(self) -> None:
        self._ib: Any | None = None
        self._trades_by_order_id: dict[int, Any] = {}

    def connect(self, config: IbkrConnectionConfig) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError("Install IBKR support with `python -m pip install -e .[ibkr]`") from exc
        self._ib = IB()
        self._ib.connect(
            config.host,
            config.port,
            clientId=config.client_id,
            readonly=False,
            timeout=config.timeout_seconds,
        )

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self._ib is not None and self._ib.isConnected())

    def account_values(self) -> list[Any]:
        return list(self._ib.accountValues()) if self._ib is not None else []

    def positions(self) -> list[Any]:
        return list(self._ib.positions()) if self._ib is not None else []

    def open_orders(self) -> list[Any]:
        return list(self._ib.openOrders()) if self._ib is not None else []

    def resolve_contract(self, spec: IbkrContractSpec) -> Any:
        if self._ib is None:
            raise RuntimeError("IBKR client is not connected")
        from ib_insync import Stock

        contract = Stock(
            spec.local_symbol,
            spec.exchange,
            spec.currency,
            primaryExchange=spec.primary_exchange or "",
        )
        details = self._ib.reqContractDetails(contract)
        if not details:
            raise ValueError(f"No IBKR contract details returned for {spec.tradingbot_symbol}")
        return details[0]

    def place_market_order(
        self,
        contract: Any,
        action: str,
        quantity: Decimal,
        *,
        order_type: str = ORDER_TEST_ORDER_TYPE,
        tif: str = ORDER_TEST_TIF,
    ) -> int:
        if self._ib is None:
            raise RuntimeError("IBKR client is not connected")
        from ib_insync import MarketOrder

        if order_type != ORDER_TEST_ORDER_TYPE:
            raise RuntimeError(f"IBKR paper order test only supports {ORDER_TEST_ORDER_TYPE} orders")
        order = MarketOrder(action.upper(), float(quantity))
        order.tif = tif
        trade = self._ib.placeOrder(contract, order)
        order_id = int(_value(_value(trade, "order"), "orderId"))
        self._trades_by_order_id[order_id] = trade
        return order_id

    def wait_for_fill(self, order_id: int, timeout_seconds: int) -> IbkrOrderFillResult:
        if self._ib is None:
            raise RuntimeError("IBKR client is not connected")
        trade = self._trades_by_order_id.get(order_id)
        if trade is None:
            raise RuntimeError(f"Unknown IBKR order id {order_id}")
        changes: list[IbkrOrderStatusChange] = []
        seen: set[tuple[str, Decimal, Decimal | None, Decimal | None]] = set()
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = _order_status_text(trade)
            filled = _order_filled_quantity(trade)
            remaining = _order_remaining_quantity(trade)
            average_price = _order_average_fill_price(trade)
            key = (status, filled, remaining, average_price)
            if key not in seen:
                seen.add(key)
                changes.append(
                    IbkrOrderStatusChange(
                        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                        status=status,
                        filled_quantity=filled,
                        remaining_quantity=remaining,
                        average_fill_price=average_price,
                    )
                )
            if status.upper() in {"FILLED", "CANCELLED", "INACTIVE"}:
                return IbkrOrderFillResult(
                    order_id=order_id,
                    final_status=status,
                    filled_quantity=filled,
                    average_fill_price=average_price,
                    status_changes=changes,
                    error_messages=_trade_error_messages(trade),
                    timed_out=False,
                )
            if time.monotonic() >= deadline:
                return IbkrOrderFillResult(
                    order_id=order_id,
                    final_status=status,
                    filled_quantity=filled,
                    average_fill_price=average_price,
                    status_changes=changes,
                    error_messages=_trade_error_messages(trade),
                    timed_out=True,
                )
            self._ib.waitOnUpdate(timeout=1)

    def cancel_order(self, order_id: int) -> None:
        if self._ib is None:
            raise RuntimeError("IBKR client is not connected")
        trade = self._trades_by_order_id.get(order_id)
        if trade is None:
            raise RuntimeError(f"Unknown IBKR order id {order_id}")
        self._ib.cancelOrder(_value(trade, "order"))


def run_ibkr_order_round_trip(
    config: IbkrOrderTestConfig,
    *,
    client: IbkrOrderTestClientBoundary | None = None,
    store: TradingBotSQLiteStore | None = None,
    event_logger: Callable[[str], None] | None = None,
) -> IbkrOrderRoundTripResult:
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    result_base = {
        "started_at": started_at,
        "endpoint": f"{config.host}:{config.port}",
        "client_id": config.client_id,
    }
    try:
        _validate_static_guards(config)
        active_store = store or TradingBotSQLiteStore(config.database_path)
        active_client = client or IbInsyncPaperOrderClient()
        connection_config = IbkrConnectionConfig(
            host=config.host,
            port=config.port,
            client_id=config.client_id,
            readonly=False,
            timeout_seconds=min(config.timeout_seconds, 30),
        )
        try:
            active_client.connect(connection_config)
            account = _read_account(active_client)
            _validate_account(account, active_store)
            open_orders = [_open_order(value) for value in active_client.open_orders()]
            if open_orders:
                raise RuntimeError(_unexpected_open_orders_message(open_orders))
            specs = load_contract_specs(config.contracts_path)
            spec = _find_contract_spec(specs, config.symbol)
            resolved = _resolve_contract(active_client, spec)
            _validate_contract_against_verified_store(resolved, active_store)
            broker_positions_before = [_position(value) for value in active_client.positions()]
            if _position_quantity(broker_positions_before, resolved) != 0:
                raise RuntimeError(f"Existing {config.symbol} IBKR position detected; refusing round-trip test")
            contract_object = _contract_object(active_client.resolve_contract(spec))

            _log_pre_submission(event_logger, "BUY", config.quantity)
            buy_order_id = active_client.place_market_order(
                contract_object,
                "BUY",
                config.quantity,
                order_type=ORDER_TEST_ORDER_TYPE,
                tif=ORDER_TEST_TIF,
            )
            buy_fill = active_client.wait_for_fill(buy_order_id, config.timeout_seconds)
            warnings: list[str] = []
            cancelled_buy_order = False
            if buy_fill.timed_out:
                active_client.cancel_order(buy_order_id)
                cancelled_buy_order = True
                if buy_fill.filled_quantity == 0:
                    return IbkrOrderRoundTripResult(
                        **result_base,
                        passed=False,
                        account_environment=account.environment,
                        account_id_masked=mask_account_id(account.account_id),
                        contract=resolved,
                        buy_order_id=buy_order_id,
                        buy_status=buy_fill.final_status,
                        buy_fill_quantity=buy_fill.filled_quantity,
                        buy_average_fill_price=buy_fill.average_fill_price,
                        buy_status_changes=buy_fill.status_changes,
                        buy_error_messages=buy_fill.error_messages,
                        cancelled_buy_order=True,
                        error="BUY did not fill before timeout; only the test BUY order was cancelled",
                    )
                warnings.append("BUY partially filled before timeout; unfilled remainder was cancelled")
            if buy_fill.final_status.upper() in {"CANCELLED", "INACTIVE", "REJECTED"} and buy_fill.filled_quantity <= 0:
                return IbkrOrderRoundTripResult(
                    **result_base,
                    passed=False,
                    account_environment=account.environment,
                    account_id_masked=mask_account_id(account.account_id),
                    contract=resolved,
                    buy_order_id=buy_order_id,
                    buy_status=buy_fill.final_status,
                    buy_fill_quantity=buy_fill.filled_quantity,
                    buy_average_fill_price=buy_fill.average_fill_price,
                    buy_status_changes=buy_fill.status_changes,
                    buy_error_messages=buy_fill.error_messages,
                    cancelled_buy_order=cancelled_buy_order,
                    error=f"BUY was {buy_fill.final_status} before any fill; stopping safely",
                )
            if buy_fill.filled_quantity <= 0:
                return IbkrOrderRoundTripResult(
                    **result_base,
                    passed=False,
                    account_environment=account.environment,
                    account_id_masked=mask_account_id(account.account_id),
                    contract=resolved,
                    buy_order_id=buy_order_id,
                    buy_status=buy_fill.final_status,
                    buy_fill_quantity=buy_fill.filled_quantity,
                    buy_average_fill_price=buy_fill.average_fill_price,
                    buy_status_changes=buy_fill.status_changes,
                    buy_error_messages=buy_fill.error_messages,
                    cancelled_buy_order=cancelled_buy_order,
                    error="BUY order produced no filled quantity",
                )

            filled_quantity = buy_fill.filled_quantity
            positions_after_buy = [_position(value) for value in active_client.positions()]
            buy_position_quantity = _position_quantity(positions_after_buy, resolved)
            if buy_position_quantity < filled_quantity:
                warnings.append(
                    f"IBKR position after BUY is {buy_position_quantity}; expected at least filled quantity {filled_quantity}"
                )

            _log_pre_submission(event_logger, "SELL", filled_quantity)
            sell_order_id = active_client.place_market_order(
                contract_object,
                "SELL",
                filled_quantity,
                order_type=ORDER_TEST_ORDER_TYPE,
                tif=ORDER_TEST_TIF,
            )
            sell_fill = active_client.wait_for_fill(sell_order_id, config.timeout_seconds)
            if sell_fill.timed_out or sell_fill.filled_quantity < filled_quantity:
                remaining = filled_quantity - sell_fill.filled_quantity
                return IbkrOrderRoundTripResult(
                    **result_base,
                    passed=False,
                    account_environment=account.environment,
                    account_id_masked=mask_account_id(account.account_id),
                    contract=resolved,
                    buy_order_id=buy_order_id,
                    buy_status=buy_fill.final_status,
                    buy_fill_quantity=buy_fill.filled_quantity,
                    buy_average_fill_price=buy_fill.average_fill_price,
                    buy_status_changes=buy_fill.status_changes,
                    buy_error_messages=buy_fill.error_messages,
                    sell_order_id=sell_order_id,
                    sell_status=sell_fill.final_status,
                    sell_fill_quantity=sell_fill.filled_quantity,
                    sell_average_fill_price=sell_fill.average_fill_price,
                    sell_status_changes=sell_fill.status_changes,
                    sell_error_messages=sell_fill.error_messages,
                    cancelled_buy_order=cancelled_buy_order,
                    warnings=warnings
                    + [
                        (
                            "SELL did not fully fill. Stop all automation; an open PAPER position may remain: "
                            f"{config.symbol} quantity {remaining}"
                        )
                    ],
                    error="SELL did not fully close the paper test position",
                )

            positions_after_sell = [_position(value) for value in active_client.positions()]
            final_quantity = _position_quantity(positions_after_sell, resolved)
            reconciliation = reconcile_positions({}, positions_after_sell)
            paper_pnl = _paper_pnl(
                filled_quantity,
                buy_fill.average_fill_price,
                sell_fill.average_fill_price,
            )
            passed = final_quantity == 0 and not reconciliation.mismatches
            return IbkrOrderRoundTripResult(
                **result_base,
                passed=passed,
                account_environment=account.environment,
                account_id_masked=mask_account_id(account.account_id),
                contract=resolved,
                buy_order_id=buy_order_id,
                buy_status=buy_fill.final_status,
                buy_fill_quantity=buy_fill.filled_quantity,
                buy_average_fill_price=buy_fill.average_fill_price,
                buy_status_changes=buy_fill.status_changes,
                buy_error_messages=buy_fill.error_messages,
                sell_order_id=sell_order_id,
                sell_status=sell_fill.final_status,
                sell_fill_quantity=sell_fill.filled_quantity,
                sell_average_fill_price=sell_fill.average_fill_price,
                sell_status_changes=sell_fill.status_changes,
                sell_error_messages=sell_fill.error_messages,
                paper_pnl=paper_pnl,
                final_position_quantity=final_quantity,
                reconciliation=reconciliation,
                cancelled_buy_order=cancelled_buy_order,
                warnings=warnings,
                error=None if passed else "Final broker position or reconciliation did not return to flat",
            )
        finally:
            active_client.disconnect()
    except Exception as exc:
        return IbkrOrderRoundTripResult(**result_base, passed=False, error=str(exc))


def _validate_static_guards(config: IbkrOrderTestConfig) -> None:
    if not config.explicit_test_mode:
        raise RuntimeError("Explicit IBKR paper order test mode is not enabled")
    if config.host != DEFAULT_IBKR_HOST or config.port != DEFAULT_IBKR_PAPER_PORT:
        raise RuntimeError("IBKR paper order test only allows endpoint 127.0.0.1:7497")
    if config.quantity != MAX_ORDER_TEST_QUANTITY:
        raise RuntimeError("IBKR paper order test maximum quantity is exactly 1 share")
    if config.symbol != DEFAULT_ORDER_TEST_SYMBOL:
        raise RuntimeError("IBKR paper order test is locked to ERIC-B.ST")
    if config.client_id != DEFAULT_IBKR_CLIENT_ID:
        raise RuntimeError("IBKR paper order test is locked to client ID 15")


def _log_pre_submission(
    event_logger: Callable[[str], None] | None,
    action: str,
    quantity: Decimal,
) -> None:
    if event_logger is None:
        return
    event_logger(
        f"Preparing IBKR PAPER {action} before submission: "
        f"order_type={ORDER_TEST_ORDER_TYPE}, tif={ORDER_TEST_TIF}, quantity={quantity}"
    )


def _read_account(client: IbkrOrderTestClientBoundary) -> IbkrAccountSnapshot:
    captured_at = datetime.now(timezone.utc).replace(tzinfo=None)
    return _account_snapshot(
        connected=client.is_connected(),
        values=client.account_values(),
        captured_at=captured_at,
    )


def _validate_account(account: IbkrAccountSnapshot, store: TradingBotSQLiteStore) -> None:
    if not account.connected:
        raise RuntimeError("IBKR connection is not healthy")
    if account.environment != "PAPER":
        raise RuntimeError(f"IBKR order test requires PAPER account; connected environment is {account.environment}")
    latest = store.latest_ibkr_snapshot()
    if latest is None:
        raise RuntimeError("No previously verified IBKR paper account snapshot found")
    expected_hash = latest["account_id_hash"] if "account_id_hash" in latest.keys() else None
    actual_hash = account_id_hash(account.account_id)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError("Connected IBKR account does not match the previously verified paper account")
    if not expected_hash and latest["account_id_masked"] != mask_account_id(account.account_id):
        raise RuntimeError("Connected IBKR account does not match the previously verified paper account")


def _find_contract_spec(specs: list[IbkrContractSpec], symbol: str) -> IbkrContractSpec:
    for spec in specs:
        if spec.tradingbot_symbol == symbol:
            return spec
    raise RuntimeError(f"No configured IBKR contract spec found for {symbol}")


def _validate_contract_against_verified_store(
    resolved: IbkrResolvedContract,
    store: TradingBotSQLiteStore,
) -> None:
    if not resolved.verified:
        raise RuntimeError(f"IBKR contract did not verify for {resolved.tradingbot_symbol}: {resolved.error}")
    rows = [row for row in store.latest_ibkr_contracts() if row["tradingbot_symbol"] == resolved.tradingbot_symbol]
    if not rows:
        raise RuntimeError(f"No previously verified IBKR contract stored for {resolved.tradingbot_symbol}")
    stored = rows[0]
    if int(stored["verified"]) != 1:
        raise RuntimeError(f"Previously stored IBKR contract is not verified for {resolved.tradingbot_symbol}")
    if stored["con_id"] is None or resolved.con_id != int(stored["con_id"]):
        raise RuntimeError(
            f"Resolved {resolved.tradingbot_symbol} conId {resolved.con_id} does not match stored conId {stored['con_id']}"
        )


def _position_quantity(positions: list[IbkrPosition], contract: IbkrResolvedContract) -> Decimal:
    for position in positions:
        if contract.con_id is not None and position.con_id == contract.con_id:
            return position.quantity
        if _normalized_local_symbol(position.local_symbol) == _normalized_local_symbol(contract.local_symbol):
            return position.quantity
    return Decimal("0")


def _contract_object(details: Any) -> Any:
    return _value(details, "contract") or details


def _unexpected_open_orders_message(open_orders: list[IbkrOpenOrder]) -> str:
    descriptions = [
        f"orderId={order.order_id}, symbol={order.symbol}, action={order.action}, status={order.status}"
        for order in open_orders
    ]
    return "Unexpected open IBKR orders exist; refusing test: " + "; ".join(descriptions)


def _paper_pnl(
    quantity: Decimal,
    buy_average_price: Decimal | None,
    sell_average_price: Decimal | None,
) -> Decimal | None:
    if buy_average_price is None or sell_average_price is None:
        return None
    return (sell_average_price - buy_average_price) * quantity


def _order_status_text(trade: Any) -> str:
    status = _value(trade, "orderStatus")
    return str(_value(status, "status") or "UNKNOWN")


def _order_filled_quantity(trade: Any) -> Decimal:
    status = _value(trade, "orderStatus")
    return _decimal_or_none(_value(status, "filled")) or Decimal("0")


def _order_remaining_quantity(trade: Any) -> Decimal | None:
    status = _value(trade, "orderStatus")
    return _decimal_or_none(_value(status, "remaining"))


def _order_average_fill_price(trade: Any) -> Decimal | None:
    status = _value(trade, "orderStatus")
    return _decimal_or_none(_value(status, "avgFillPrice"))


def _trade_error_messages(trade: Any) -> list[str]:
    messages: list[str] = []
    for entry in _value(trade, "log") or []:
        status = _value(entry, "status")
        message = _value(entry, "message")
        error_code = _value(entry, "errorCode") or _value(entry, "code")
        if error_code or message:
            prefix = f"IBKR error {error_code}: " if error_code else "IBKR message: "
            messages.append(f"{prefix}{message or status or ''}".strip())
        elif status:
            messages.append(f"IBKR status: {status}")
    return messages
