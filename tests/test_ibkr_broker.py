from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.execution.ibkr_broker import (
    DEFAULT_IBKR_CLIENT_ID,
    DEFAULT_IBKR_PAPER_PORT,
    IBKRReadOnlyBroker,
    IbkrConnectionConfig,
    IbkrContractSpec,
    IbkrPosition,
    account_id_hash,
    load_contract_specs,
    reconcile_positions,
    save_resolved_contracts,
)
from trading_bot.persistence.sqlite_store import TradingBotSQLiteStore


class FakeClient:
    def __init__(self, *, fail_connect: bool = False) -> None:
        self.fail_connect = fail_connect
        self.connected = False
        self.disconnected = False
        self.connected_config = None

    def connect(self, config):
        self.connected_config = config
        if self.fail_connect:
            raise RuntimeError("TWS offline")
        self.connected = True

    def disconnect(self):
        self.disconnected = True
        self.connected = False

    def is_connected(self):
        return self.connected

    def account_values(self):
        return [
            {"tag": "AccountCurrency", "value": "SEK", "currency": "", "account": "DU1234567"},
            {"tag": "TotalCashValue", "value": "1000.50", "currency": "SEK", "account": "DU1234567"},
            {"tag": "NetLiquidation", "value": "1100.25", "currency": "SEK", "account": "DU1234567"},
            {"tag": "BuyingPower", "value": "900.00", "currency": "SEK", "account": "DU1234567"},
        ]

    def positions(self):
        return [
            {
                "account": "DU1234567",
                "contract": {
                    "symbol": "SAND",
                    "localSymbol": "SAND",
                    "conId": 123,
                    "exchange": "SFB",
                    "currency": "SEK",
                    "secType": "STK",
                },
                "position": "2",
                "avgCost": "220.10",
            }
        ]

    def open_orders(self):
        return [
            {
                "contract": {"localSymbol": "SAND"},
                "order": {"orderId": 7, "action": "BUY", "orderType": "LMT", "totalQuantity": "1"},
                "orderStatus": {"status": "Submitted"},
            }
        ]

    def executions(self):
        return [
            {
                "contract": {"localSymbol": "SAND"},
                "execution": {
                    "execId": "abc",
                    "side": "BOT",
                    "shares": "1",
                    "price": "220",
                    "time": "20260811 12:00:00",
                },
            }
        ]

    def resolve_contract(self, spec):
        return {
            "contract": {
                "conId": 123,
                "localSymbol": spec.local_symbol,
                "exchange": spec.exchange,
                "primaryExchange": spec.primary_exchange,
                "currency": spec.currency,
                "secType": spec.security_type,
            },
            "longName": "Sandvik AB",
        }


def spec(symbol: str = "SAND.ST") -> IbkrContractSpec:
    return IbkrContractSpec(
        tradingbot_symbol=symbol,
        local_symbol="SAND",
        exchange="SMART",
        primary_exchange="SFB",
        currency="SEK",
        security_type="STK",
    )


def test_ibkr_connection_and_account_parsing() -> None:
    client = FakeClient()
    broker = IBKRReadOnlyBroker(IbkrConnectionConfig(client_id=22), client=client)

    snapshot = broker.read_snapshot(contract_specs=[spec()], local_positions={"SAND": Decimal("2")})

    assert client.connected_config.port == DEFAULT_IBKR_PAPER_PORT
    assert client.connected_config.client_id == 22
    assert client.connected_config.readonly
    assert client.disconnected
    assert snapshot.connection.connected
    assert snapshot.connection.environment == "PAPER"
    assert snapshot.connection.base_currency == "SEK"
    assert snapshot.connection.cash_balance == Decimal("1000.50")
    assert snapshot.connection.net_liquidation_value == Decimal("1100.25")
    assert snapshot.connection.buying_power == Decimal("900.00")
    assert len(snapshot.positions) == 1
    assert len(snapshot.open_orders) == 1
    assert len(snapshot.recent_executions) == 1


def test_ibkr_offline_fails_safely_and_disconnects() -> None:
    client = FakeClient(fail_connect=True)
    broker = IBKRReadOnlyBroker(client=client)

    snapshot = broker.read_snapshot(contract_specs=[spec()])

    assert not snapshot.connection.connected
    assert snapshot.connection.error == "TWS offline"
    assert snapshot.positions == []
    assert client.disconnected


def test_contract_mapping_is_verified_and_saved(tmp_path: Path) -> None:
    broker = IBKRReadOnlyBroker(client=FakeClient())

    snapshot = broker.read_snapshot(contract_specs=[spec()])
    output_path = save_resolved_contracts(snapshot.contracts, tmp_path / "contracts.json")

    assert snapshot.contracts[0].verified
    assert snapshot.contracts[0].con_id == 123
    assert "Sandvik AB" in output_path.read_text(encoding="utf-8")


def test_contract_mapping_rejects_wrong_contract() -> None:
    class WrongContractClient(FakeClient):
        def resolve_contract(self, spec):
            result = super().resolve_contract(spec)
            result["contract"]["currency"] = "USD"
            return result

    broker = IBKRReadOnlyBroker(client=WrongContractClient())

    snapshot = broker.read_snapshot(contract_specs=[spec()])

    assert not snapshot.contracts[0].verified
    assert snapshot.contracts[0].error == "Resolved contract did not match configured Swedish stock spec"


def test_contract_mapping_accepts_ibkr_dot_space_class_equivalence() -> None:
    class SpaceClassClient(FakeClient):
        def resolve_contract(self, spec):
            result = super().resolve_contract(spec)
            result["contract"]["localSymbol"] = "VOLV B"
            return result

    broker = IBKRReadOnlyBroker(client=SpaceClassClient())
    contract_spec = IbkrContractSpec(
        tradingbot_symbol="VOLV-B.ST",
        local_symbol="VOLV.B",
        exchange="SMART",
        primary_exchange="SFB",
        currency="SEK",
        security_type="STK",
    )

    snapshot = broker.read_snapshot(contract_specs=[contract_spec])

    assert snapshot.contracts[0].verified
    assert snapshot.contracts[0].local_symbol == "VOLV B"


def test_reconciliation_reports_position_mismatches() -> None:
    result = reconcile_positions(
        {"SAND": Decimal("1"), "ERIC B": Decimal("3")},
        [
            IbkrPosition(
                account_id="DU123",
                symbol="SAND",
                local_symbol="SAND",
                con_id=1,
                exchange="SFB",
                currency="SEK",
                security_type="STK",
                quantity=Decimal("2"),
                average_cost=Decimal("100"),
            )
        ],
    )

    assert result.local_positions_count == 2
    assert result.ibkr_positions_count == 1
    assert "SAND: local quantity 1 != IBKR quantity 2" in result.mismatches
    assert "ERIC B: local quantity 3 != IBKR quantity 0" in result.mismatches


def test_ibkr_adapter_refuses_order_submission() -> None:
    broker = IBKRReadOnlyBroker(client=FakeClient())

    with pytest.raises(RuntimeError, match="read-only"):
        broker.submit_order(None, Decimal("100"))  # type: ignore[arg-type]


def test_non_readonly_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="read-only"):
        IBKRReadOnlyBroker(IbkrConnectionConfig(readonly=False), client=FakeClient())


def test_configured_swedish_contract_specs_are_explicit() -> None:
    specs = load_contract_specs()
    symbols = {item.tradingbot_symbol for item in specs}

    assert len(specs) == 10
    assert {"SAND.ST", "ERIC-B.ST", "SHB-A.ST"}.issubset(symbols)
    assert all(item.currency == "SEK" for item in specs)
    assert all(item.security_type == "STK" for item in specs)
    assert DEFAULT_IBKR_CLIENT_ID == 15


def test_ibkr_snapshot_persistence_masks_account(tmp_path: Path) -> None:
    broker = IBKRReadOnlyBroker(client=FakeClient())
    snapshot = broker.read_snapshot(contract_specs=[spec()])
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")

    store.save_ibkr_snapshot(snapshot)
    stored = store.latest_ibkr_snapshot()
    contracts = store.latest_ibkr_contracts()

    assert stored is not None
    assert stored["account_id_masked"] == "DU***67"
    assert stored["account_id_hash"] == account_id_hash("DU1234567")
    assert stored["connected"] == 1
    assert contracts[0]["tradingbot_symbol"] == "SAND.ST"
    assert contracts[0]["verified"] == 1
