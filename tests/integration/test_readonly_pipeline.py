"""End-to-end read-only pipeline: FakeMT5 -> MT5Broker -> MarketDataService.

Validates the Phase-1 definition of done: symbol discovery, multi-timeframe
data, tick validation, freshness gating and session inference — all without a
real terminal and all deterministically.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.config import AppConfig
from app.core.enums import SessionState, TimeFrame
from app.core.exceptions import (
    MT5NotConnectedError,
    SymbolNotGoldError,
)
from app.market.data_service import MarketDataService
from tests.conftest import REF_TIME, make_fake_mt5


def make_service(fake, *, clock=None, **service_kwargs) -> MarketDataService:
    broker = MT5Broker(mt5_module=fake)
    broker.connect()
    defaults = dict(
        symbol="AUTO",
        candle_count=300,
        max_tick_age_seconds=60,
        candle_freshness_multiplier=3.0,
        clock=clock or (lambda: REF_TIME),
    )
    defaults.update(service_kwargs)
    return MarketDataService(broker, **defaults)


class TestHealthyPipeline:
    def test_snapshot_is_trading_grade(self, fake_mt5):
        service = make_service(fake_mt5)
        service.connect()

        snapshot = service.get_snapshot()

        assert snapshot.symbol is not None
        assert snapshot.symbol.name == "XAUUSD"
        assert snapshot.trading_data_ok is True
        assert snapshot.session_state is SessionState.OPEN
        assert snapshot.tick.valid and snapshot.tick.fresh
        assert snapshot.tick.spread_points == pytest.approx(20.0)
        assert set(snapshot.series) == {TimeFrame.M15, TimeFrame.H1, TimeFrame.H4}
        for tf, check in snapshot.series.items():
            assert check.valid, f"{tf}: {check.report.summary()}"
            assert check.fresh
            assert check.series is not None
            assert len(check.series.candles) == 300
            # closed candles only, newest closed candle is the one before NOW
            assert check.series.include_forming is False

        # M15: forming bar opened 12:00 -> last closed candle opened 11:45
        last_m15 = snapshot.series[TimeFrame.M15].series.last_closed
        assert last_m15.time == REF_TIME - timedelta(minutes=15)

    def test_symbol_discovery_is_cached(self, fake_mt5):
        service = make_service(fake_mt5)
        spec_1 = service.resolve_symbol()
        spec_2 = service.resolve_symbol()
        assert spec_1 is spec_2
        service.resolve_symbol(force=True)  # re-discovery works
        assert service.symbol_spec.name == "XAUUSD"

    def test_explicit_symbol_setting(self, fake_mt5):
        service = make_service(fake_mt5, symbol="GOLD")
        assert service.resolve_symbol().name == "GOLD"

    def test_snapshot_bid_ask(self, fake_mt5):
        snapshot = make_service(fake_mt5).get_snapshot()
        assert snapshot.bid == 2650.0
        assert snapshot.ask == 2650.2
        assert snapshot.summary()["symbol"] == "XAUUSD"


class TestDataQualityGates:
    def test_stale_market_blocks_trading(self):
        fake = make_fake_mt5()  # market anchored at REF_TIME
        # the service clock runs two days ahead -> everything is stale
        service = make_service(fake, clock=lambda: REF_TIME + timedelta(days=2))

        snapshot = service.get_snapshot()

        assert snapshot.trading_data_ok is False
        assert snapshot.session_state is SessionState.CLOSED
        assert snapshot.tick.fresh is False
        assert all(not check.fresh for check in snapshot.series.values())

    def test_stale_tick_alone_blocks_trading(self, fake_mt5):
        # tick 10 minutes old, candles current (mixed state)
        fake_mt5.set_tick("XAUUSD", epoch=REF_TIME.timestamp() - 600, bid=2650.0, ask=2650.2)
        service = make_service(fake_mt5)

        snapshot = service.get_snapshot()

        assert snapshot.tick.valid is True  # sane data...
        assert snapshot.tick.fresh is False  # ...but not fresh
        assert snapshot.trading_data_ok is False
        assert snapshot.session_state is SessionState.UNKNOWN

    def test_zero_tick_blocks_trading(self, fake_mt5):
        fake_mt5.set_tick("XAUUSD", epoch=REF_TIME.timestamp() - 2, bid=0.0, ask=0.0)
        service = make_service(fake_mt5)

        snapshot = service.get_snapshot()

        assert snapshot.trading_data_ok is False
        assert snapshot.tick.valid is False

    def test_inverted_quote_blocks_trading(self, fake_mt5):
        fake_mt5.set_tick("XAUUSD", epoch=REF_TIME.timestamp() - 2, bid=2650.5, ask=2650.0)
        service = make_service(fake_mt5)

        snapshot = service.get_snapshot()

        assert snapshot.trading_data_ok is False

    def test_missing_timeframe_blocks_trading(self, fake_mt5):
        from tests.fakes.mt5_fake import TIMEFRAME_H1

        del fake_mt5.rates[("XAUUSD", TIMEFRAME_H1)]
        service = make_service(fake_mt5)

        snapshot = service.get_snapshot()

        assert snapshot.trading_data_ok is False
        h1 = snapshot.series[TimeFrame.H1]
        assert h1.series is None
        assert not h1.valid

    def test_broken_symbol_falls_back_to_next_best(self, fake_mt5):
        fake_mt5.symbols["XAUUSD"] = fake_mt5.symbols["XAUUSD"]._replace(point=0.0)
        service = make_service(fake_mt5)

        # XAUUSD is broken -> discovery picks XAUUSDm, which has no rates ->
        # snapshot reports fetch failure instead of silently using bad data.
        assert service.resolve_symbol().name == "XAUUSDm"
        snapshot = service.get_snapshot()
        assert snapshot.trading_data_ok is False


class TestClockOffset:
    def test_server_ahead_two_hours_still_fresh(self):
        # Broker server clock is UTC+2: bars and ticks are stamped 2h ahead.
        server_now = REF_TIME + timedelta(hours=2)
        fake = make_fake_mt5(now=server_now)
        service = make_service(fake, clock=lambda: REF_TIME)

        snapshot = service.get_snapshot()

        assert service.clock_offset_seconds == pytest.approx(7200.0, abs=5.0)
        assert snapshot.trading_data_ok is True
        assert snapshot.session_state is SessionState.OPEN
        assert snapshot.tick.age_seconds == pytest.approx(2.0, abs=5.0)


class TestConnectionLifecycle:
    def test_disconnected_broker_raises(self, fake_mt5):
        service = make_service(fake_mt5)
        service.disconnect()
        with pytest.raises(MT5NotConnectedError):
            service.get_snapshot()

    def test_never_connected_broker_raises(self, fake_mt5):
        broker = MT5Broker(mt5_module=fake_mt5)
        service = MarketDataService(broker, clock=lambda: REF_TIME)
        with pytest.raises(MT5NotConnectedError):
            service.get_snapshot()


class TestGoldOnlyProtection:
    def test_non_gold_configured_symbol_is_rejected(self, fake_mt5):
        service = make_service(fake_mt5, symbol="EURUSD")
        with pytest.raises(SymbolNotGoldError):
            service.resolve_symbol()

    def test_silver_configured_symbol_is_rejected(self, fake_mt5):
        service = make_service(fake_mt5, symbol="XAGUSD")
        with pytest.raises(SymbolNotGoldError):
            service.resolve_symbol()


class TestFromConfig:
    def test_service_from_config(self, fake_mt5):
        config = AppConfig(_env_file=None, symbol="AUTO", candle_count=150)
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()
        service = MarketDataService.from_config(config, broker, clock=lambda: REF_TIME)

        snapshot = service.get_snapshot()

        assert snapshot.trading_data_ok is True
        for check in snapshot.series.values():
            assert len(check.series.candles) == 150
