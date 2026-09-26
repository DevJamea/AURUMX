"""Tick validator tests — the spec §53 edge cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.enums import SymbolTradeMode
from app.core.models import MarketTick, SymbolSpec
from app.market.tick import TickValidator

NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def spec(point: float = 0.01) -> SymbolSpec:
    return SymbolSpec(
        name="XAUUSD",
        visible=True,
        trade_mode=SymbolTradeMode.FULL,
        digits=2,
        point=point,
        tick_size=point,
        tick_value=1.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )


def tick(seconds_ago: float, bid: float = 2650.0, ask: float = 2650.2) -> MarketTick:
    return MarketTick(
        symbol="XAUUSD",
        time=NOW - timedelta(seconds=seconds_ago),
        bid=bid,
        ask=ask,
    )


def validator(max_age: float = 60.0) -> TickValidator:
    return TickValidator(max_age_seconds=max_age, clock=lambda: NOW)


class TestTickValidation:
    def test_valid_fresh_tick(self):
        check = validator().validate(tick(2), symbol_spec=spec())
        assert check.valid
        assert check.fresh
        assert check.report.ok
        assert check.spread == pytest.approx(0.2, abs=1e-6)
        assert check.spread_points == pytest.approx(20.0, abs=1e-3)
        assert check.age_seconds == pytest_approx(2.0)

    def test_missing_tick(self):
        check = validator().validate(None)
        assert not check.valid
        assert not check.fresh
        assert any(i.code == "TICK_MISSING" for i in check.report.errors)

    def test_zero_tick(self):
        check = validator().validate(tick(2, bid=0.0, ask=0.0))
        assert not check.valid
        assert any(i.code == "TICK_ZERO_PRICE" for i in check.report.errors)

    def test_ask_below_bid(self):
        check = validator().validate(tick(2, bid=2650.5, ask=2650.0))
        assert not check.valid
        assert any(i.code == "TICK_INVERTED" for i in check.report.errors)
        assert check.spread is None  # spread meaningless when inverted

    def test_stale_tick_is_valid_data_but_not_fresh(self):
        check = validator(max_age=60).validate(tick(300))
        assert check.valid  # the tick itself is sane
        assert not check.fresh
        assert any(i.code == "TICK_STALE" for i in check.report.errors)

    def test_future_tick_beyond_tolerance_is_error(self):
        check = validator().validate(tick(-600))  # 10 minutes in the future
        assert not check.fresh
        assert not check.report.ok  # recorded as a data-quality error
        assert any(i.code == "TICK_FUTURE_TIMESTAMP" for i in check.report.errors)

    def test_small_future_skew_is_tolerated(self):
        check = validator().validate(tick(-5))  # 5s ahead — clock noise
        assert check.fresh

    def test_clock_offset_corrects_server_time(self):
        # Server clock is 2h ahead: tick timestamp is now+2h-2s in server time.
        server_tick = MarketTick(
            symbol="XAUUSD",
            time=NOW + timedelta(hours=2) - timedelta(seconds=2),
            bid=2650.0,
            ask=2650.2,
        )
        check = validator().validate(server_tick, clock_offset_seconds=7200.0)
        assert check.valid
        assert check.fresh
        assert check.age_seconds == pytest_approx(2.0)

    def test_spread_points_requires_point(self):
        check = validator().validate(tick(2), symbol_spec=None)
        assert check.spread_points is None
        assert check.spread == pytest.approx(0.2, abs=1e-6)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-6)
