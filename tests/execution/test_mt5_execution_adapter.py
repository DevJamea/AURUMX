"""MT5 execution adapter tests — the ONLY order_check/order_send call site.

Verifies the request dict built for MetaTrader5, the check-then-send
ordering, retcode classification against the *module's* constants, and the
fail-closed behavior on every degenerate broker response.
"""

from __future__ import annotations

import pytest

from app.brokers.mt5 import MT5Broker, _classify_retcode
from app.core.enums import Direction
from app.core.models import MarketOrderRequest
from tests.execution.conftest import make_exec_broker, make_exec_fake
from tests.fakes.mt5_fake import (
    TRADE_ACTION_DEAL,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_DONE_PARTIAL,
    TRADE_RETCODE_INVALID,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_NO_MONEY,
    TRADE_RETCODE_REQUOTE,
    TRADE_RETCODE_TIMEOUT,
)


def make_order_request(**overrides) -> MarketOrderRequest:
    base = dict(
        symbol="XAUUSD", direction=Direction.LONG, volume=0.08,
        sl=2645.2, tp=2660.2, deviation_points=25,
        comment="AURUMX test", magic=0x41555258,
    )
    base.update(overrides)
    return MarketOrderRequest(**base)


class TestOrderConstruction:
    def test_market_order_dict_shape(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        broker.place_market_order(make_order_request())
        sent = exec_fake.order_sends[0]
        assert sent["action"] == TRADE_ACTION_DEAL
        assert sent["symbol"] == "XAUUSD"
        assert sent["volume"] == 0.08
        assert sent["type"] == 0  # ORDER_TYPE_BUY for LONG
        assert sent["sl"] == 2645.2
        assert sent["tp"] == 2660.2
        assert sent["deviation"] == 25
        assert sent["magic"] == 0x41555258
        assert sent["comment"] == "AURUMX test"
        assert "type_filling" not in sent  # omitted unless explicitly set

    def test_buy_uses_ask_sell_uses_bid(self, exec_fake):
        exec_fake.set_tick("XAUUSD", epoch=1_767_225_600, bid=2650.00, ask=2650.20)
        broker = make_exec_broker(exec_fake)
        broker.place_market_order(make_order_request())
        assert exec_fake.order_sends[0]["price"] == 2650.20  # BUY -> ask

        exec_fake.order_sends.clear()
        broker.place_market_order(
            make_order_request(direction=Direction.SHORT)
        )
        assert exec_fake.order_sends[0]["price"] == 2650.00  # SELL -> bid
        assert exec_fake.order_sends[0]["type"] == 1  # ORDER_TYPE_SELL

    def test_explicit_filling_mode_is_passed_through(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        broker.place_market_order(make_order_request(type_filling=2))
        assert exec_fake.order_sends[0]["type_filling"] == 2

    def test_none_sl_tp_sent_as_zero(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        broker.place_market_order(make_order_request(sl=None, tp=None))
        assert exec_fake.order_sends[0]["sl"] == 0.0
        assert exec_fake.order_sends[0]["tp"] == 0.0


class TestCheckBeforeSend:
    def test_check_runs_first_and_send_follows(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted
        assert len(exec_fake.order_checks) == 1
        assert len(exec_fake.order_sends) == 1
        # the exact same order dict is checked and sent
        assert exec_fake.order_checks[0] == exec_fake.order_sends[0]

    @pytest.mark.parametrize("retcode", [TRADE_RETCODE_INVALID, TRADE_RETCODE_NO_MONEY, TRADE_RETCODE_MARKET_CLOSED])
    def test_failed_check_prevents_send(self, retcode):
        fake = make_exec_fake(check_retcode=retcode)
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.phase == "check"
        assert result.retcode == retcode
        assert len(fake.order_checks) == 1
        assert len(fake.order_sends) == 0  # THE invariant


class TestRetcodeClassification:
    def test_success_verdict(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is True
        assert result.retcode == TRADE_RETCODE_DONE
        assert result.retcode_description == "TRADE_RETCODE_DONE"
        assert result.category == "accepted"
        assert result.ticket is not None and result.ticket > 0
        assert result.deal_ticket is not None and result.deal_ticket > 0
        assert result.price == 2650.20
        assert result.volume == 0.08

    def test_partial_is_accepted_with_lower_volume(self):
        fake = make_exec_fake(send_retcode=TRADE_RETCODE_DONE_PARTIAL)
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is True  # broker confirmed (partial) execution
        assert result.category == "partial"
        assert result.volume < 0.08

    @pytest.mark.parametrize(
        "retcode,category",
        [
            (TRADE_RETCODE_REQUOTE, "requote"),
            (TRADE_RETCODE_INVALID, "invalid_request"),
            (TRADE_RETCODE_MARKET_CLOSED, "market_closed"),
            (TRADE_RETCODE_NO_MONEY, "no_money"),
        ],
    )
    def test_definite_rejections(self, retcode, category):
        fake = make_exec_fake(send_retcode=retcode)
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.phase == "send"
        assert result.category == category
        assert result.ticket is None  # never fabricate identifiers

    def test_timeout_is_not_accepted(self):
        fake = make_exec_fake(send_retcode=TRADE_RETCODE_TIMEOUT)
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.category == "timeout"

    def test_unknown_retcode_fails_closed(self):
        fake = make_exec_fake(send_retcode=99001)
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.category == "unknown"
        assert "UNKNOWN_99001" in result.retcode_description

    def test_classification_uses_module_constants(self, exec_fake):
        """The table is resolved from the module's own constants when
        present (spec §7A.5) — retie one and watch the mapping follow."""
        name, category = _classify_retcode(exec_fake, TRADE_RETCODE_DONE)
        assert (name, category) == ("TRADE_RETCODE_DONE", "accepted")
        # a module that exposes no constants still classifies via fallbacks
        class BareModule:
            TRADE_RETCODE_DONE = TRADE_RETCODE_DONE

        name, category = _classify_retcode(BareModule, TRADE_RETCODE_REQUOTE)
        assert (name, category) == ("TRADE_RETCODE_REQUOTE", "requote")


class TestFailClosedResponses:
    def test_no_price_fails_closed_without_sending(self):
        fake = make_exec_fake()
        fake.tick = None  # terminal serving no quotes
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.phase == "check"
        assert result.category == "no_price"
        assert len(fake.order_sends) == 0
        assert len(fake.order_checks) == 0

    def test_check_returning_none_fails_closed(self):
        fake = make_exec_fake()
        fake.order_check = lambda order: None  # type: ignore[method-assign]
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.category == "unknown"
        assert len(fake.order_sends) == 0

    def test_send_returning_none_is_unknown(self):
        fake = make_exec_fake()
        fake.order_send = lambda order: None  # type: ignore[method-assign]
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.phase == "send"
        assert result.category == "unknown"

    def test_send_exception_is_structured_failure(self):
        fake = make_exec_fake(send_exception=OSError("ipc broken"))
        broker = make_exec_broker(fake)
        result = broker.place_market_order(make_order_request())
        assert result.accepted is False
        assert result.phase == "send"
        assert result.category == "error"
        assert "ipc broken" in result.message

    def test_not_connected_raises(self):
        from app.core.exceptions import MT5NotConnectedError

        broker = MT5Broker(mt5_module=make_exec_fake())  # never connected
        with pytest.raises(MT5NotConnectedError):
            broker.place_market_order(make_order_request())

    def test_position_management_remains_stubbed(self, exec_fake):
        from app.core.exceptions import ExecutionNotImplementedError

        broker = make_exec_broker(exec_fake)
        with pytest.raises(ExecutionNotImplementedError):
            broker.modify_position(1, sl=1.0)
        with pytest.raises(ExecutionNotImplementedError):
            broker.close_position(1)


class TestIsTradingAllowed:
    def test_reflects_terminal_flag(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        assert broker.is_trading_allowed() is True
        exec_fake.terminal = exec_fake.terminal._replace(trade_allowed=False)
        assert broker.is_trading_allowed() is False

    def test_none_when_terminal_unavailable(self, exec_fake):
        broker = make_exec_broker(exec_fake)
        exec_fake.terminal = None  # type: ignore[assignment]
        assert broker.is_trading_allowed() is None
