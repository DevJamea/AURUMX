"""Domain model tests: construction constraints, helpers, symbol verification."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.core.enums import AccountTradeMode, Direction, OrderType, SymbolTradeMode, TimeFrame
from app.core.models import (
    AccountSnapshot,
    Candle,
    CandleSeries,
    MarketTick,
    PendingOrder,
    Position,
    SymbolSpec,
    ValidationReport,
)

TS = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def make_tick(**kwargs) -> MarketTick:
    params = {"symbol": "XAUUSD", "time": TS, "bid": 2650.0, "ask": 2650.2}
    params.update(kwargs)
    return MarketTick(**params)


def make_candle(**kwargs) -> Candle:
    params = {
        "time": TS,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "tick_volume": 100,
    }
    params.update(kwargs)
    return Candle(**params)


class TestMarketTick:
    def test_spread_and_mid(self):
        tick = make_tick(bid=100.0, ask=100.4)
        assert tick.spread == pytest.approx(0.4)
        assert tick.mid == pytest.approx(100.2)

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            make_tick(bid=float("nan"))

    def test_naive_datetime_is_assumed_utc(self):
        tick = make_tick(time=datetime(2026, 1, 5, 12, 0))
        assert tick.time.tzinfo is UTC


class TestCandleSeries:
    def test_last_and_last_closed_without_forming(self):
        series = CandleSeries(
            symbol="XAUUSD",
            timeframe=TimeFrame.M15,
            candles=[
                make_candle(time=TS),
                make_candle(time=TS.replace(minute=15)),
            ],
        )
        assert series.last is series.candles[-1]
        assert series.last_closed is series.candles[-1]  # all closed

    def test_last_closed_excludes_forming_bar(self):
        series = CandleSeries(
            symbol="XAUUSD",
            timeframe=TimeFrame.M15,
            candles=[make_candle(time=TS), make_candle(time=TS.replace(minute=15))],
            include_forming=True,
        )
        assert series.last is series.candles[-1]
        assert series.last_closed is series.candles[0]

    def test_empty_series_last_is_none(self):
        series = CandleSeries(symbol="XAUUSD", timeframe=TimeFrame.M15)
        assert series.last is None
        assert series.last_closed is None
        assert series.is_empty

    def test_to_dataframe_shape(self):
        series = CandleSeries(
            symbol="XAUUSD",
            timeframe=TimeFrame.M15,
            candles=[make_candle(time=TS), make_candle(time=TS.replace(minute=15))],
        )
        df = series.to_dataframe()
        assert list(df.columns) == [
            "time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread_points",
        ]
        assert len(df) == 2


class TestSymbolSpec:
    def test_valid_spec_passes(self):
        from tests.fakes.mt5_fake import default_symbol

        spec = SymbolSpec(
            name="XAUUSD",
            visible=True,
            selected=True,
            trade_mode=SymbolTradeMode.FULL,
            digits=2,
            point=0.01,
            tick_size=0.01,
            tick_value=1.0,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            stops_level_points=20,
            freeze_level_points=10,
            currency_profit="USD",
        )
        report = spec.validate()
        assert report.ok, report.summary()
        assert not report.has_warnings
        assert default_symbol().name == "XAUUSD"

    def test_broken_metadata_fails(self):
        spec = SymbolSpec(name="XAUUSD.bad", digits=2, trade_mode=SymbolTradeMode.FULL)
        report = spec.validate()
        assert not report.ok
        codes = {issue.code for issue in report.errors}
        assert "POINT_NOT_POSITIVE" in codes
        assert "TICK_SIZE_NOT_POSITIVE" in codes
        assert "CONTRACT_SIZE_NOT_POSITIVE" in codes
        assert "VOLUME_MIN_NOT_POSITIVE" in codes
        assert "NOT_VISIBLE" in codes

    def test_disabled_trading_fails(self):
        spec = SymbolSpec(
            name="XAUUSD",
            visible=True,
            trade_mode=SymbolTradeMode.DISABLED,
            digits=2,
            point=0.01,
            tick_size=0.01,
            tick_value=1.0,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        assert not spec.validate().ok

    def test_unusual_digits_is_warning_not_error(self):
        spec = SymbolSpec(
            name="XAUUSD",
            visible=True,
            trade_mode=SymbolTradeMode.FULL,
            digits=5,
            point=0.00001,
            tick_size=0.00001,
            tick_value=0.001,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        report = spec.validate()
        assert report.ok  # warning only
        assert report.has_warnings

    def test_normalize_volume(self):
        spec = SymbolSpec(
            name="XAUUSD",
            trade_mode=SymbolTradeMode.FULL,
            digits=2,
            point=0.01,
            tick_size=0.01,
            tick_value=1.0,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        assert spec.normalize_volume(0.015) == 0.01   # rounds down to step
        assert spec.normalize_volume(0.10) == 0.10
        assert spec.normalize_volume(0.005) == 0.01   # clamped up to min
        assert spec.normalize_volume(150.0) == 100.0  # clamped down to max


class TestAccountSnapshot:
    def test_trade_mode_helpers(self):
        demo = AccountSnapshot(login=1, trade_mode=AccountTradeMode.DEMO)
        real = AccountSnapshot(login=2, trade_mode=AccountTradeMode.REAL)
        assert demo.is_demo and not demo.is_real
        assert real.is_real and not real.is_demo


class TestPortfolioModels:
    def test_position_defaults_and_has_sl(self):
        position = Position(
            ticket=1,
            symbol="XAUUSD",
            direction=Direction.LONG,
            volume=0.5,
            price_open=2650.0,
            time=TS,
        )
        assert position.has_sl is False
        assert position.price_sl is None

        with_sl = position.model_copy(update={"price_sl": 2640.0})
        assert with_sl.has_sl is True

    def test_pending_order(self):
        order = PendingOrder(
            ticket=2,
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            price_open=2600.0,
            time_setup=TS,
        )
        assert order.order_type is OrderType.BUY_LIMIT
        assert order.time_expiration is None


class TestValidationReport:
    def test_ok_with_warnings(self):
        report = ValidationReport()
        report.warning("W", "careful")
        assert report.ok
        assert report.has_warnings
        assert "WARNING W: careful" in report.summary()

    def test_errors_block(self):
        report = ValidationReport()
        report.error("E1", "bad")
        report.warning("W", "careful")
        assert not report.ok
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
