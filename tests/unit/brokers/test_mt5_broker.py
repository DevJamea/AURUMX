"""MT5Broker tests against the FakeMT5 module (no Windows terminal needed)."""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.config import AppConfig
from app.core.enums import (
    AccountTradeMode,
    Direction,
    OrderType,
    TimeFrame,
)
from app.core.exceptions import (
    ExecutionNotImplementedError,
    MarketDataError,
    MT5ConnectionError,
    MT5NotConnectedError,
    MT5UnavailableError,
)
from app.core.models import MarketOrderRequest, PendingOrderRequest
from tests.conftest import REF_TIME
from tests.fakes.mt5_fake import (
    ACCOUNT_TRADE_MODE_REAL,
    ORDER_TYPE_BUY_LIMIT,
    POSITION_TYPE_BUY,
    POSITION_TYPE_SELL,
    FakeOrder,
    FakePosition,
    default_account,
)


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        from app.core.logging import JsonFormatter

        self.lines.append(JsonFormatter().format(record))


@pytest.fixture()
def log_capture():
    handler = CaptureHandler()
    root = logging.getLogger("aurumx")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    yield handler
    root.removeHandler(handler)


class TestConnection:
    def test_missing_package_fails_closed(self):
        broker = MT5Broker(import_name="DefinitelyNotInstalledModule")
        with pytest.raises(MT5UnavailableError, match="Windows-only"):
            broker.connect()

    def test_connect_and_disconnect(self, fake_mt5):
        broker = MT5Broker(mt5_module=fake_mt5)
        assert not broker.is_connected
        broker.connect()
        assert broker.is_connected
        assert fake_mt5.initialize_calls == [{"timeout": 60000}]
        broker.disconnect()
        assert not broker.is_connected
        assert fake_mt5.shutdown_called
        broker.disconnect()  # idempotent

    def test_failed_initialize_raises_with_last_error(self):
        from tests.fakes.mt5_fake import FakeMT5

        fake = FakeMT5(init_result=False, last_error_value=(-6, "terminal not found"))
        broker = MT5Broker(mt5_module=fake)
        with pytest.raises(MT5ConnectionError, match="terminal not found"):
            broker.connect()

    def test_login_without_password_refused(self):
        from tests.fakes.mt5_fake import FakeMT5

        fake = FakeMT5()
        broker = MT5Broker(mt5_module=fake, login=12345)
        with pytest.raises(MT5ConnectionError, match="MT5_PASSWORD"):
            broker.connect()

    def test_credentials_passed_to_initialize(self):
        from tests.fakes.mt5_fake import FakeMT5

        fake = FakeMT5()
        broker = MT5Broker(
            mt5_module=fake, login=12345, password="secret-pw", server="Broker-Demo"
        )
        broker.connect()
        call = fake.initialize_calls[0]
        assert call["login"] == 12345
        assert call["password"] == "secret-pw"
        assert call["server"] == "Broker-Demo"

    def test_password_never_appears_in_logs(self, log_capture):
        from tests.fakes.mt5_fake import FakeMT5

        fake = FakeMT5()
        broker = MT5Broker(mt5_module=fake, login=12345, password="s3cr3t-p4ssw0rd")
        broker.connect()
        broker.disconnect()

        joined = "\n".join(log_capture.lines)
        assert "s3cr3t-p4ssw0rd" not in joined

    def test_operations_require_connection(self, fake_mt5):
        broker = MT5Broker(mt5_module=fake_mt5)
        with pytest.raises(MT5NotConnectedError):
            broker.get_tick("XAUUSD")
        with pytest.raises(MT5NotConnectedError):
            broker.get_account()
        with pytest.raises(MT5NotConnectedError):
            broker.get_candles("XAUUSD", TimeFrame.M15, 10)

    def test_from_config_attaches_injected_module(self, fake_mt5):
        config = AppConfig(
            _env_file=None,
            mt5_login=999,
            mt5_password="from-config",
            mt5_server="Cfg-Server",
            mt5_terminal_path="C:\\MT5\\terminal64.exe",
        )
        broker = MT5Broker.from_config(config, mt5_module=fake_mt5)
        broker.connect()
        assert broker.is_connected
        call = fake_mt5.initialize_calls[0]
        assert call["login"] == 999
        assert call["server"] == "Cfg-Server"
        assert call["path"] == "C:\\MT5\\terminal64.exe"
        assert call["password"] == "from-config"


class TestAccount:
    def test_account_mapping_demo(self, broker: MT5Broker):
        account = broker.get_account()
        assert account.login == 12345678
        assert account.trade_mode is AccountTradeMode.DEMO
        assert account.is_demo
        assert account.currency == "USD"
        assert account.equity == 10_000.0

    def test_real_account_logs_warning(self, fake_mt5, log_capture):
        fake_mt5.account = default_account(trade_mode=ACCOUNT_TRADE_MODE_REAL)
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        joined = "\n".join(log_capture.lines)
        assert "MT5_REAL_ACCOUNT" in joined

    def test_no_account_raises(self, fake_mt5):
        fake_mt5.account = None
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()
        with pytest.raises(MT5ConnectionError, match="no data"):
            broker.get_account()


class TestSymbols:
    def test_list_symbols(self, broker: MT5Broker):
        names = broker.list_symbols()
        assert "XAUUSD" in names
        assert "XAGUSD" in names

    def test_get_symbol_metadata(self, broker: MT5Broker):
        spec = broker.get_symbol("XAUUSD")
        assert spec is not None
        assert spec.name == "XAUUSD"
        assert spec.point == 0.01
        assert spec.contract_size == 100.0
        assert spec.volume_step == 0.01
        assert spec.stops_level_points == 20
        assert spec.currency_profit == "USD"

    def test_get_unknown_symbol_returns_none(self, broker: MT5Broker):
        assert broker.get_symbol("NOPE") is None

    def test_select_symbol(self, broker: MT5Broker, fake_mt5):
        assert broker.select_symbol("XAUUSD") is True
        assert fake_mt5.symbols["XAUUSD"].select is True
        assert broker.select_symbol("NOPE") is False


class TestMarketData:
    def test_get_tick_uses_millisecond_time(self, broker: MT5Broker):
        tick = broker.get_tick("XAUUSD")
        assert tick is not None
        assert tick.symbol == "XAUUSD"
        assert tick.bid == 2650.0
        assert tick.ask == 2650.2
        expected = REF_TIME - timedelta(seconds=2)
        assert tick.time == expected

    def test_get_tick_none_when_unavailable(self, broker: MT5Broker):
        broker.disconnect()
        broker.connect()
        assert broker.get_tick("XAUUSD") is not None
        assert broker.get_symbol("NOPE") is None

    def test_get_candles_drops_forming_bar(self, broker: MT5Broker):
        result = broker.get_candles("XAUUSD", TimeFrame.M15, 20)
        assert len(result.candles) == 20
        forming_open = REF_TIME.replace(minute=0, second=0)
        assert result.candles[-1].time < forming_open

        for candle in result.candles:
            assert candle.high >= candle.low
            assert candle.high >= max(candle.open, candle.close)

    def test_get_candles_unknown_symbol_raises(self, broker: MT5Broker):
        with pytest.raises(MarketDataError):
            broker.get_candles("NOPE", TimeFrame.M15, 10)

    def test_get_candles_no_history_raises(self, fake_mt5):
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()
        with pytest.raises(MarketDataError):
            broker.get_candles("GOLD", TimeFrame.M15, 10)  # GOLD has no rates wired


class TestPortfolio:
    def test_get_positions(self, fake_mt5):
        fake_mt5.positions = [
            FakePosition(
                ticket=101, symbol="XAUUSD", type=POSITION_TYPE_BUY, volume=0.5,
                price_open=2600.0, price_current=2650.0, sl=2590.0, tp=2700.0,
                profit=25.0, swap=0.0, time=int(REF_TIME.timestamp()) - 3600,
                comment="aurumx", magic=777,
            ),
            FakePosition(
                ticket=102, symbol="XAUUSD", type=POSITION_TYPE_SELL, volume=0.2,
                price_open=2660.0, price_current=2650.0, sl=0.0, tp=0.0,
                profit=2.0, swap=0.0, time=int(REF_TIME.timestamp()) - 1800,
                comment="", magic=0,
            ),
        ]
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        positions = broker.get_positions("XAUUSD")

        assert len(positions) == 2
        first, second = positions
        assert first.direction is Direction.LONG
        assert first.price_sl == 2590.0
        assert first.magic == 777
        assert second.direction is Direction.SHORT
        assert second.price_sl is None  # MT5 zero -> None
        assert second.has_sl is False

    def test_get_orders(self, fake_mt5):
        fake_mt5.orders = [
            FakeOrder(
                ticket=201, symbol="XAUUSD", type=ORDER_TYPE_BUY_LIMIT, volume_current=0.1,
                price_open=2600.0, sl=2590.0, tp=2700.0,
                time_setup=int(REF_TIME.timestamp()), time_expiration=0,
                comment="aurumx-pending", magic=777,
            ),
        ]
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        orders = broker.get_orders()

        assert len(orders) == 1
        assert orders[0].order_type is OrderType.BUY_LIMIT
        assert orders[0].price_open == 2600.0
        assert orders[0].time_expiration is None


class TestExecutionSurface:
    """Phase 5: market orders are implemented; position management stays
    stubbed (later phase).  The fake module's execution functions remain
    unreachable unless a test explicitly opts in."""

    def test_position_management_still_raises(self, broker: MT5Broker):
        pending = PendingOrderRequest(
            symbol="XAUUSD", order_type=OrderType.BUY_LIMIT, volume=0.1, price_open=2600.0
        )
        with pytest.raises(ExecutionNotImplementedError):
            broker.place_pending_order(pending)
        with pytest.raises(ExecutionNotImplementedError):
            broker.modify_position(101, sl=2600.0)
        with pytest.raises(ExecutionNotImplementedError):
            broker.close_position(101)
        with pytest.raises(ExecutionNotImplementedError):
            broker.partial_close(101, 0.05)
        with pytest.raises(ExecutionNotImplementedError):
            broker.cancel_order(201)

    def test_market_order_fails_closed_on_read_only_fake(self, fake_mt5):
        """The fake's read-only safety net must never turn into a success:
        order_check raising -> structured fail-closed result, nothing sent."""
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()
        result = broker.place_market_order(
            MarketOrderRequest(symbol="XAUUSD", direction=Direction.LONG, volume=0.1)
        )
        assert result.accepted is False
        assert result.phase == "check"
        assert result.category == "error"
        assert fake_mt5.order_sends == []  # order_send never reached
