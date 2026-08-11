"""Read-only Interactive Brokers paper broker adapter.

This adapter can read account, position, order, execution, and contract
metadata from a locally running TWS/Gateway paper session. It intentionally
refuses order submission.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from trading_bot.data.models import Order, Trade
from trading_bot.execution.broker import Broker


DEFAULT_IBKR_HOST = "127.0.0.1"
DEFAULT_IBKR_PAPER_PORT = 7497
DEFAULT_IBKR_CLIENT_ID = 15
DEFAULT_CONTRACT_CONFIG = Path("config/ibkr_swedish_contracts.json")
DEFAULT_RESOLVED_CONTRACTS_PATH = Path("data/ibkr_resolved_contracts.json")


@dataclass(frozen=True)
class IbkrConnectionConfig:
    host: str = DEFAULT_IBKR_HOST
    port: int = DEFAULT_IBKR_PAPER_PORT
    client_id: int = DEFAULT_IBKR_CLIENT_ID
    readonly: bool = True
    timeout_seconds: int = 10


@dataclass(frozen=True)
class IbkrAccountSnapshot:
    connected: bool
    environment: str
    account_id: str | None
    base_currency: str | None
    cash_balance: Decimal | None
    net_liquidation_value: Decimal | None
    buying_power: Decimal | None
    captured_at: datetime
    error: str | None = None


@dataclass(frozen=True)
class IbkrPosition:
    account_id: str
    symbol: str
    local_symbol: str
    con_id: int | None
    exchange: str | None
    currency: str | None
    security_type: str | None
    quantity: Decimal
    average_cost: Decimal | None


@dataclass(frozen=True)
class IbkrOpenOrder:
    order_id: int | None
    symbol: str
    action: str | None
    order_type: str | None
    total_quantity: Decimal | None
    status: str | None


@dataclass(frozen=True)
class IbkrExecution:
    execution_id: str | None
    symbol: str
    side: str | None
    quantity: Decimal | None
    price: Decimal | None
    time: str | None


@dataclass(frozen=True)
class IbkrContractSpec:
    tradingbot_symbol: str
    local_symbol: str
    exchange: str
    primary_exchange: str | None
    currency: str
    security_type: str


@dataclass(frozen=True)
class IbkrResolvedContract:
    tradingbot_symbol: str
    intended_local_symbol: str
    verified: bool
    con_id: int | None
    local_symbol: str | None
    exchange: str | None
    primary_exchange: str | None
    currency: str | None
    security_type: str | None
    long_name: str | None
    error: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    local_positions_count: int
    ibkr_positions_count: int
    mismatches: list[str]


@dataclass(frozen=True)
class IbkrReadOnlySnapshot:
    connection: IbkrAccountSnapshot
    positions: list[IbkrPosition]
    open_orders: list[IbkrOpenOrder]
    recent_executions: list[IbkrExecution]
    contracts: list[IbkrResolvedContract]
    reconciliation: ReconciliationResult


class IbkrClientBoundary(Protocol):
    def connect(self, config: IbkrConnectionConfig) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def account_values(self) -> list[Any]: ...
    def positions(self) -> list[Any]: ...
    def open_orders(self) -> list[Any]: ...
    def executions(self) -> list[Any]: ...
    def resolve_contract(self, spec: IbkrContractSpec) -> Any: ...


class IBKRReadOnlyBroker(Broker):
    def __init__(
        self,
        config: IbkrConnectionConfig | None = None,
        *,
        client: IbkrClientBoundary | None = None,
    ) -> None:
        self.config = config or IbkrConnectionConfig()
        if not self.config.readonly:
            raise ValueError("IBKR adapter only supports read-only integration mode")
        self.client = client or IbInsyncClientBoundary()

    def submit_order(self, order: Order, market_price: Decimal) -> Trade:
        raise RuntimeError("IBKR integration is read-only; order submission is disabled")

    def read_snapshot(
        self,
        *,
        contract_specs: list[IbkrContractSpec],
        local_positions: dict[str, Decimal] | None = None,
    ) -> IbkrReadOnlySnapshot:
        captured_at = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            self.client.connect(self.config)
            account = _account_snapshot(
                connected=self.client.is_connected(),
                values=self.client.account_values(),
                captured_at=captured_at,
            )
            positions = [_position(value) for value in self.client.positions()]
            open_orders = [_open_order(value) for value in self.client.open_orders()]
            executions = [_execution(value) for value in self.client.executions()]
            contracts = [_resolve_contract(self.client, spec) for spec in contract_specs]
            return IbkrReadOnlySnapshot(
                connection=account,
                positions=positions,
                open_orders=open_orders,
                recent_executions=executions,
                contracts=contracts,
                reconciliation=reconcile_positions(local_positions or {}, positions),
            )
        except Exception as exc:
            return IbkrReadOnlySnapshot(
                connection=IbkrAccountSnapshot(
                    connected=False,
                    environment="UNKNOWN",
                    account_id=None,
                    base_currency=None,
                    cash_balance=None,
                    net_liquidation_value=None,
                    buying_power=None,
                    captured_at=captured_at,
                    error=str(exc),
                ),
                positions=[],
                open_orders=[],
                recent_executions=[],
                contracts=[],
                reconciliation=reconcile_positions(local_positions or {}, []),
            )
        finally:
            try:
                self.client.disconnect()
            except Exception:
                pass


class IbInsyncClientBoundary:
    def __init__(self) -> None:
        self._ib: Any | None = None

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
            readonly=True,
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

    def executions(self) -> list[Any]:
        return list(self._ib.executions()) if self._ib is not None else []

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


def load_contract_specs(path: str | Path = DEFAULT_CONTRACT_CONFIG) -> list[IbkrContractSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: list[IbkrContractSpec] = []
    for symbol, value in payload.items():
        specs.append(
            IbkrContractSpec(
                tradingbot_symbol=symbol,
                local_symbol=str(value["local_symbol"]),
                exchange=str(value["exchange"]),
                primary_exchange=value.get("primary_exchange"),
                currency=str(value["currency"]),
                security_type=str(value["security_type"]),
            )
        )
    return specs


def save_resolved_contracts(
    contracts: list[IbkrResolvedContract],
    path: str | Path = DEFAULT_RESOLVED_CONTRACTS_PATH,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([contract.__dict__ for contract in contracts], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def reconcile_positions(
    local_positions: dict[str, Decimal],
    ibkr_positions: list[IbkrPosition],
) -> ReconciliationResult:
    mismatches: list[str] = []
    ibkr_by_symbol = {position.symbol: position.quantity for position in ibkr_positions if position.quantity != 0}
    for symbol, local_quantity in local_positions.items():
        ibkr_quantity = ibkr_by_symbol.get(symbol, Decimal("0"))
        if local_quantity != ibkr_quantity:
            mismatches.append(f"{symbol}: local quantity {local_quantity} != IBKR quantity {ibkr_quantity}")
    for symbol, ibkr_quantity in ibkr_by_symbol.items():
        if symbol not in local_positions:
            mismatches.append(f"{symbol}: IBKR has quantity {ibkr_quantity}, local TradingBot has no open position")
    return ReconciliationResult(
        local_positions_count=sum(1 for quantity in local_positions.values() if quantity != 0),
        ibkr_positions_count=len(ibkr_by_symbol),
        mismatches=mismatches,
    )


def mask_account_id(account_id: str | None) -> str:
    if not account_id:
        return "n/a"
    if len(account_id) <= 4:
        return "*" * len(account_id)
    return f"{account_id[:2]}***{account_id[-2:]}"


def account_id_hash(account_id: str | None) -> str | None:
    if not account_id:
        return None
    return hashlib.sha256(account_id.encode("utf-8")).hexdigest()


def _account_snapshot(
    *,
    connected: bool,
    values: list[Any],
    captured_at: datetime,
) -> IbkrAccountSnapshot:
    tags = {(str(_value(item, "tag")), str(_value(item, "currency") or "")): item for item in values}
    account_id = _first_account(values)
    account_currency = _first_value(values, "AccountCurrency")
    base_currency = account_currency if account_currency and account_currency != "BASE" else _first_currency(values)
    return IbkrAccountSnapshot(
        connected=connected,
        environment=_environment(account_id),
        account_id=account_id,
        base_currency=base_currency,
        cash_balance=_account_decimal(tags, "TotalCashValue", base_currency)
        or _account_decimal(tags, "CashBalance", base_currency),
        net_liquidation_value=_account_decimal(tags, "NetLiquidation", base_currency),
        buying_power=_account_decimal(tags, "BuyingPower", base_currency),
        captured_at=captured_at,
    )


def _resolve_contract(client: IbkrClientBoundary, spec: IbkrContractSpec) -> IbkrResolvedContract:
    try:
        details = client.resolve_contract(spec)
        contract = _value(details, "contract") or details
        con_id = _int_or_none(_value(contract, "conId"))
        local_symbol = _string_or_none(_value(contract, "localSymbol")) or _string_or_none(_value(contract, "symbol"))
        exchange = _string_or_none(_value(contract, "exchange"))
        primary_exchange = _string_or_none(_value(contract, "primaryExchange"))
        currency = _string_or_none(_value(contract, "currency"))
        security_type = _string_or_none(_value(contract, "secType"))
        long_name = _string_or_none(_value(details, "longName"))
        verified = (
            con_id is not None
            and _normalized_local_symbol(local_symbol) == _normalized_local_symbol(spec.local_symbol)
            and currency == spec.currency
            and security_type == spec.security_type
            and (spec.primary_exchange is None or primary_exchange in {spec.primary_exchange, None, ""})
        )
        return IbkrResolvedContract(
            tradingbot_symbol=spec.tradingbot_symbol,
            intended_local_symbol=spec.local_symbol,
            verified=verified,
            con_id=con_id,
            local_symbol=local_symbol,
            exchange=exchange,
            primary_exchange=primary_exchange,
            currency=currency,
            security_type=security_type,
            long_name=long_name,
            error=None if verified else "Resolved contract did not match configured Swedish stock spec",
        )
    except Exception as exc:
        return IbkrResolvedContract(
            tradingbot_symbol=spec.tradingbot_symbol,
            intended_local_symbol=spec.local_symbol,
            verified=False,
            con_id=None,
            local_symbol=None,
            exchange=None,
            primary_exchange=None,
            currency=None,
            security_type=None,
            long_name=None,
            error=str(exc),
        )


def _position(value: Any) -> IbkrPosition:
    contract = _value(value, "contract")
    return IbkrPosition(
        account_id=str(_value(value, "account") or ""),
        symbol=str(_value(contract, "symbol") or _value(contract, "localSymbol") or ""),
        local_symbol=str(_value(contract, "localSymbol") or _value(contract, "symbol") or ""),
        con_id=_int_or_none(_value(contract, "conId")),
        exchange=_string_or_none(_value(contract, "exchange")),
        currency=_string_or_none(_value(contract, "currency")),
        security_type=_string_or_none(_value(contract, "secType")),
        quantity=_decimal(_value(value, "position")),
        average_cost=_decimal_or_none(_value(value, "avgCost")),
    )


def _open_order(value: Any) -> IbkrOpenOrder:
    contract = _value(value, "contract")
    order = _value(value, "order") or value
    status = _value(value, "orderStatus")
    return IbkrOpenOrder(
        order_id=_int_or_none(_value(order, "orderId")),
        symbol=str(_value(contract, "localSymbol") or _value(contract, "symbol") or ""),
        action=_string_or_none(_value(order, "action")),
        order_type=_string_or_none(_value(order, "orderType")),
        total_quantity=_decimal_or_none(_value(order, "totalQuantity")),
        status=_string_or_none(_value(status, "status")),
    )


def _execution(value: Any) -> IbkrExecution:
    contract = _value(value, "contract")
    execution = _value(value, "execution") or value
    return IbkrExecution(
        execution_id=_string_or_none(_value(execution, "execId")),
        symbol=str(_value(contract, "localSymbol") or _value(contract, "symbol") or ""),
        side=_string_or_none(_value(execution, "side")),
        quantity=_decimal_or_none(_value(execution, "shares")),
        price=_decimal_or_none(_value(execution, "price")),
        time=_string_or_none(_value(execution, "time")),
    )


def _first_account(values: list[Any]) -> str | None:
    for item in values:
        account = _string_or_none(_value(item, "account"))
        if account:
            return account
    return None


def _first_currency(values: list[Any]) -> str | None:
    for item in values:
        currency = _string_or_none(_value(item, "currency"))
        if currency and currency != "BASE":
            return currency
    return None


def _first_value(values: list[Any], tag: str) -> str | None:
    for item in values:
        if _value(item, "tag") == tag:
            return _string_or_none(_value(item, "value"))
    return None


def _account_decimal(tags: dict[tuple[str, str], Any], tag: str, currency: str | None) -> Decimal | None:
    keys = [(tag, currency or ""), (tag, "")]
    for key in keys:
        item = tags.get(key)
        if item is not None:
            return _decimal_or_none(_value(item, "value"))
    for (item_tag, _), item in tags.items():
        if item_tag == tag:
            return _decimal_or_none(_value(item, "value"))
    return None


def _environment(account_id: str | None) -> str:
    if account_id is None:
        return "UNKNOWN"
    if account_id.startswith("DU"):
        return "PAPER"
    if account_id.startswith("U"):
        return "LIVE"
    return "UNKNOWN"


def _value(item: Any, name: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _decimal(value: Any) -> Decimal:
    parsed = _decimal_or_none(value)
    return parsed if parsed is not None else Decimal("0")


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalized_local_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace(".", " ").strip().upper()
