from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_bot.execution.ibkr_broker import (
    IbkrAccountSnapshot,
    IbkrReadOnlySnapshot,
    IbkrResolvedContract,
    ReconciliationResult,
)
from trading_bot.execution.ibkr_order_test import (
    DEFAULT_ORDER_TEST_SYMBOL,
    IbkrOrderFillResult,
    IbkrOrderStatusChange,
    IbkrOrderTestConfig,
    run_ibkr_order_round_trip,
)
from trading_bot.persistence.sqlite_store import TradingBotSQLiteStore


ACCOUNT_ID = "DU1234567"
ERIC_CON_ID = 917845


class FakeOrderClient:
    def __init__(
        self,
        *,
        account_id: str = ACCOUNT_ID,
        open_orders=None,
        initial_position: Decimal = Decimal("0"),
        fill_plans=None,
    ) -> None:
        self.account_id = account_id
        self.open_order_values = open_orders or []
        self.position_quantity = initial_position
        self.fill_plans = fill_plans or [
            ("Filled", Decimal("1"), Decimal("101"), False),
            ("Filled", Decimal("1"), Decimal("102"), False),
        ]
        self.connected = False
        self.disconnected = False
        self.connected_config = None
        self.order_id = 1000
        self.actions: dict[int, str] = {}
        self.order_quantities: dict[int, Decimal] = {}
        self.order_types: dict[int, str] = {}
        self.order_tifs: dict[int, str] = {}
        self.cancelled_order_ids: list[int] = []

    def connect(self, config):
        self.connected_config = config
        self.connected = True

    def disconnect(self):
        self.disconnected = True
        self.connected = False

    def is_connected(self):
        return self.connected

    def account_values(self):
        return [
            {"tag": "AccountCurrency", "value": "SEK", "currency": "", "account": self.account_id},
            {"tag": "TotalCashValue", "value": "1000", "currency": "SEK", "account": self.account_id},
            {"tag": "NetLiquidation", "value": "1000", "currency": "SEK", "account": self.account_id},
        ]

    def positions(self):
        if self.position_quantity == 0:
            return []
        return [
            {
                "account": self.account_id,
                "contract": self._contract_payload(),
                "position": str(self.position_quantity),
                "avgCost": "101",
            }
        ]

    def open_orders(self):
        return self.open_order_values

    def resolve_contract(self, spec):
        return {
            "contract": self._contract_payload(),
            "longName": "Telefonaktiebolaget LM Ericsson",
        }

    def place_market_order(
        self,
        contract,
        action: str,
        quantity: Decimal,
        *,
        order_type: str = "MKT",
        tif: str = "DAY",
    ) -> int:
        self.order_id += 1
        self.actions[self.order_id] = action
        self.order_quantities[self.order_id] = quantity
        self.order_types[self.order_id] = order_type
        self.order_tifs[self.order_id] = tif
        return self.order_id

    def wait_for_fill(self, order_id: int, timeout_seconds: int) -> IbkrOrderFillResult:
        plan = self.fill_plans.pop(0)
        status, filled, price, timed_out = plan[:4]
        errors = plan[4] if len(plan) > 4 else []
        action = self.actions[order_id]
        if action == "BUY":
            self.position_quantity += filled
        else:
            self.position_quantity -= filled
        return _fill_result(order_id, status, filled, price, timed_out, errors=errors)

    def cancel_order(self, order_id: int) -> None:
        self.cancelled_order_ids.append(order_id)

    def _contract_payload(self):
        return {
            "conId": ERIC_CON_ID,
            "symbol": "ERIC",
            "localSymbol": "ERIC B",
            "exchange": "SMART",
            "primaryExchange": "SFB",
            "currency": "SEK",
            "secType": "STK",
        }


def test_wrong_live_account_is_rejected(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path, account_id=ACCOUNT_ID)
    client = FakeOrderClient(account_id="U1234567")

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert not result.passed
    assert "requires PAPER account" in result.error
    assert client.actions == {}


def test_wrong_account_is_rejected(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path, account_id=ACCOUNT_ID)
    client = FakeOrderClient(account_id="DU9999999")

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert not result.passed
    assert "does not match" in result.error
    assert client.actions == {}


def test_wrong_port_is_rejected_before_connect(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient()

    result = run_ibkr_order_round_trip(_config(tmp_path, port=4002), client=client, store=store)

    assert not result.passed
    assert "127.0.0.1:7497" in result.error
    assert not client.connected


def test_max_quantity_guard_blocks_larger_orders(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient()

    result = run_ibkr_order_round_trip(_config(tmp_path, quantity=Decimal("2")), client=client, store=store)

    assert not result.passed
    assert "maximum quantity" in result.error
    assert not client.connected


def test_duplicate_open_order_guard_blocks_test(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient(
        open_orders=[
            {
                "contract": {"localSymbol": "ERIC B"},
                "order": {"orderId": 9, "action": "BUY", "orderType": "MKT", "totalQuantity": "1"},
                "orderStatus": {"status": "Submitted"},
            }
        ]
    )

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert not result.passed
    assert "Unexpected open IBKR orders" in result.error
    assert client.actions == {}


def test_successful_buy_sell_round_trip_and_final_reconciliation(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient()
    events: list[str] = []

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store, event_logger=events.append)

    assert result.passed
    assert result.buy_order_id == 1001
    assert result.buy_status == "Filled"
    assert result.buy_fill_quantity == Decimal("1")
    assert result.sell_order_id == 1002
    assert result.sell_fill_quantity == Decimal("1")
    assert result.final_position_quantity == Decimal("0")
    assert result.reconciliation is not None
    assert result.reconciliation.mismatches == []
    assert result.paper_pnl == Decimal("1")
    assert client.order_quantities[1002] == Decimal("1")
    assert client.order_types == {1001: "MKT", 1002: "MKT"}
    assert client.order_tifs == {1001: "DAY", 1002: "DAY"}
    assert result.buy_order_type == "MKT"
    assert result.buy_tif == "DAY"
    assert result.sell_order_type == "MKT"
    assert result.sell_tif == "DAY"
    assert events == [
        "Preparing IBKR PAPER BUY before submission: order_type=MKT, tif=DAY, quantity=1",
        "Preparing IBKR PAPER SELL before submission: order_type=MKT, tif=DAY, quantity=1",
    ]


def test_error_10349_missing_tif_regression_uses_explicit_day_tif(tmp_path: Path) -> None:
    class StrictTifClient(FakeOrderClient):
        def place_market_order(
            self,
            contract,
            action: str,
            quantity: Decimal,
            *,
            order_type: str = "MKT",
            tif: str = "",
        ) -> int:
            if tif != "DAY":
                self.order_id += 1
                self.actions[self.order_id] = action
                self.order_quantities[self.order_id] = quantity
                self.order_types[self.order_id] = order_type
                self.order_tifs[self.order_id] = tif
                return self.order_id
            return super().place_market_order(
                contract,
                action,
                quantity,
                order_type=order_type,
                tif=tif,
            )

        def wait_for_fill(self, order_id: int, timeout_seconds: int) -> IbkrOrderFillResult:
            if self.order_tifs[order_id] != "DAY":
                return _fill_result(
                    order_id,
                    "Cancelled",
                    Decimal("0"),
                    Decimal("0"),
                    False,
                    errors=['IBKR error 10349: Order TIF was set to DAY based on order preset.'],
                )
            return super().wait_for_fill(order_id, timeout_seconds)

    store = _seed_verified_store(tmp_path)
    client = StrictTifClient()

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert result.passed
    assert client.order_tifs == {1001: "DAY", 1002: "DAY"}
    assert result.buy_error_messages == []
    assert result.sell_error_messages == []


def test_buy_cancelled_by_error_10349_stops_without_sell_and_logs_error(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient(
        fill_plans=[
            (
                "Cancelled",
                Decimal("0"),
                Decimal("0"),
                False,
                ["IBKR error 10349: Order TIF was set to DAY based on order preset."],
            )
        ]
    )

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert not result.passed
    assert "BUY was Cancelled before any fill" in result.error
    assert result.buy_error_messages == ["IBKR error 10349: Order TIF was set to DAY based on order preset."]
    assert result.sell_order_id is None
    assert len(client.actions) == 1


def test_partial_buy_fill_sells_exact_filled_quantity(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient(
        fill_plans=[
            ("Submitted", Decimal("0.5"), Decimal("101"), True),
            ("Filled", Decimal("0.5"), Decimal("102"), False),
        ]
    )

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert result.passed
    assert result.cancelled_buy_order
    assert client.cancelled_order_ids == [1001]
    assert client.order_quantities[1002] == Decimal("0.5")
    assert result.sell_fill_quantity == Decimal("0.5")
    assert result.final_position_quantity == Decimal("0.0")


def test_buy_timeout_cancels_only_test_buy_and_stops(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient(fill_plans=[("Submitted", Decimal("0"), Decimal("0"), True)])

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert not result.passed
    assert result.cancelled_buy_order
    assert client.cancelled_order_ids == [1001]
    assert len(client.actions) == 1
    assert "BUY did not fill" in result.error


def test_failed_sell_leaves_explicit_manual_close_warning(tmp_path: Path) -> None:
    store = _seed_verified_store(tmp_path)
    client = FakeOrderClient(
        fill_plans=[
            ("Filled", Decimal("1"), Decimal("101"), False),
            ("Submitted", Decimal("0"), Decimal("0"), True),
        ]
    )

    result = run_ibkr_order_round_trip(_config(tmp_path), client=client, store=store)

    assert not result.passed
    assert "SELL did not fully close" in result.error
    assert any("open PAPER position may remain: ERIC-B.ST quantity 1" in warning for warning in result.warnings)


def test_ai_and_dashboard_cannot_trigger_order_test() -> None:
    ai_source = (Path("src/trading_bot/ai/scanner.py").read_text() + Path("src/trading_bot/ai/openai_analyst.py").read_text())
    dashboard_source = (
        Path("src/trading_bot/dashboard/web.py").read_text()
        + Path("src/trading_bot/dashboard/repository.py").read_text()
    )

    assert "ibkr_order_test" not in ai_source
    assert "run_ibkr_order_round_trip" not in ai_source
    assert "ibkr-order-test" not in dashboard_source
    assert "run_ibkr_order_round_trip" not in dashboard_source


def _config(
    tmp_path: Path,
    *,
    port: int = 7497,
    quantity: Decimal = Decimal("1"),
) -> IbkrOrderTestConfig:
    return IbkrOrderTestConfig(
        port=port,
        quantity=quantity,
        explicit_test_mode=True,
        contracts_path=Path("config/ibkr_swedish_contracts.json"),
        database_path=tmp_path / "tradingbot.sqlite3",
    )


def _fill_result(
    order_id: int,
    status: str,
    filled: Decimal,
    price: Decimal,
    timed_out: bool,
    *,
    errors: list[str] | None = None,
) -> IbkrOrderFillResult:
    return IbkrOrderFillResult(
        order_id=order_id,
        final_status=status,
        filled_quantity=filled,
        average_fill_price=price if price > 0 else None,
        error_messages=errors or [],
        timed_out=timed_out,
        status_changes=[
            IbkrOrderStatusChange(
                timestamp=datetime(2026, 8, 11, 12, 0),
                status=status,
                filled_quantity=filled,
                remaining_quantity=Decimal("1") - filled,
                average_fill_price=price if price > 0 else None,
            )
        ],
    )


def _seed_verified_store(tmp_path: Path, *, account_id: str = ACCOUNT_ID) -> TradingBotSQLiteStore:
    store = TradingBotSQLiteStore(tmp_path / "tradingbot.sqlite3")
    snapshot = IbkrReadOnlySnapshot(
        connection=IbkrAccountSnapshot(
            connected=True,
            environment="PAPER",
            account_id=account_id,
            base_currency="SEK",
            cash_balance=Decimal("1000"),
            net_liquidation_value=Decimal("1000"),
            buying_power=Decimal("1000"),
            captured_at=datetime(2026, 8, 11, 11, 0),
        ),
        positions=[],
        open_orders=[],
        recent_executions=[],
        contracts=[
            IbkrResolvedContract(
                tradingbot_symbol=DEFAULT_ORDER_TEST_SYMBOL,
                intended_local_symbol="ERIC.B",
                verified=True,
                con_id=ERIC_CON_ID,
                local_symbol="ERIC B",
                exchange="SMART",
                primary_exchange="SFB",
                currency="SEK",
                security_type="STK",
                long_name="Telefonaktiebolaget LM Ericsson",
            )
        ],
        reconciliation=ReconciliationResult(local_positions_count=0, ibkr_positions_count=0, mismatches=[]),
    )
    store.save_ibkr_snapshot(snapshot)
    return store
