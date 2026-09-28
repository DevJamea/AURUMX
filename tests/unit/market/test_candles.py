"""Candle validator + multi-timeframe fetch tests (no look-ahead, freshness)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.brokers.mt5 import MT5Broker
from app.core.enums import TimeFrame
from app.core.models import Candle, CandleSeries
from app.market.candles import CandleValidator, fetch_timeframes
from tests.fakes.mt5_fake import TIMEFRAME_H4, TIMEFRAME_M15

NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def candle_at(time: datetime, *, o=100.0, h=101.0, low=99.0, c=100.5, v=10) -> Candle:
    return Candle(time=time, open=o, high=h, low=low, close=c, tick_volume=v)


def series(candles: list[Candle], timeframe: TimeFrame = TimeFrame.M15) -> CandleSeries:
    return CandleSeries(symbol="XAUUSD", timeframe=timeframe, candles=candles)


def validator() -> CandleValidator:
    return CandleValidator(freshness_multiplier=3.0, clock=lambda: NOW)


def closed_m15_series(count: int = 10) -> CandleSeries:
    """Closed M15 candles ending at 11:45 (a forming bar opened at 12:00)."""
    start = NOW - timedelta(minutes=15 * count)
    times = [start + timedelta(minutes=15 * i) for i in range(count)]
    return series([candle_at(t) for t in times])


class TestCandleValidation:
    def test_valid_series_is_fresh(self):
        check = validator().validate(closed_m15_series())
        assert check.valid
        assert check.fresh
        # last closed candle closed exactly at NOW -> age 0
        assert check.age_seconds == 0

    def test_empty_series_is_error(self):
        check = validator().validate(series([]))
        assert not check.valid
        assert any(i.code == "SERIES_EMPTY" for i in check.report.errors)

    def test_high_below_low_is_error(self):
        bad = closed_m15_series().candles[-1].model_copy(update={"high": 90.0, "low": 99.0})
        check = validator().validate(series(list(closed_m15_series().candles[:-1]) + [bad]))
        assert not check.valid
        assert any(i.code == "CANDLE_HIGH_BELOW_LOW" for i in check.report.errors)

    def test_ohlc_not_bracketed_is_error(self):
        bad = closed_m15_series().candles[-1].model_copy(update={"open": 150.0})
        check = validator().validate(series(list(closed_m15_series().candles[:-1]) + [bad]))
        assert not check.valid
        assert any(i.code == "CANDLE_OHLC_INCONSISTENT" for i in check.report.errors)

    def test_non_positive_price_is_error(self):
        bad = closed_m15_series().candles[-1].model_copy(update={"close": 0.0})
        check = validator().validate(series(list(closed_m15_series().candles[:-1]) + [bad]))
        assert any(i.code == "CANDLE_NON_POSITIVE_PRICE" for i in check.report.errors)

    def test_duplicate_timestamp_is_error(self):
        candles = closed_m15_series().candles
        duplicated = candles + [candles[-1].model_copy()]
        check = validator().validate(series(duplicated))
        assert any(i.code == "CANDLE_DUPLICATE_TIME" for i in check.report.errors)

    def test_unordered_timestamps_is_error(self):
        candles = closed_m15_series().candles
        shuffled = list(candles)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        check = validator().validate(series(shuffled))
        assert any(i.code == "CANDLE_UNORDERED" for i in check.report.errors)

    def test_stale_series_blocks(self):
        # last closed candle is hours old -> stale
        old = closed_m15_series()
        shifted = [c.model_copy(update={"time": c.time - timedelta(hours=6)}) for c in old.candles]
        check = validator().validate(series(shifted))
        assert check.valid  # OHLC sane
        assert not check.fresh
        assert any(i.code == "CANDLES_STALE" for i in check.report.errors)

    def test_future_series_is_error(self):
        future = closed_m15_series()
        shifted = [c.model_copy(update={"time": c.time + timedelta(hours=6)}) for c in future.candles]
        check = validator().validate(series(shifted))
        assert not check.fresh
        assert any(i.code == "CANDLE_FUTURE_TIMESTAMP" for i in check.report.errors)

    def test_age_uses_last_closed_not_forming(self):
        # With include_forming=True the forming bar (opened at 12:00) must not
        # hide a stale last closed candle.
        stale_closed = [c.model_copy(update={"time": c.time - timedelta(hours=6)}) for c in closed_m15_series().candles]
        with_forming = CandleSeries(
            symbol="XAUUSD",
            timeframe=TimeFrame.M15,
            candles=stale_closed + [candle_at(NOW)],
            include_forming=True,
        )
        check = validator().validate(with_forming)
        assert not check.fresh


class TestFormingBarExclusion:
    def test_broker_never_returns_forming_bar_by_default(self, broker: MT5Broker, fake_mt5):
        result = broker.get_candles("XAUUSD", TimeFrame.M15, 10, include_forming=False)
        assert len(result.candles) == 10
        forming_open = NOW.replace(minute=0, second=0)  # 12:00 bar is forming
        assert result.candles[-1].time < forming_open
        assert result.include_forming is False

    def test_include_forming_explicitly(self, broker: MT5Broker):
        result = broker.get_candles("XAUUSD", TimeFrame.M15, 10, include_forming=True)
        assert len(result.candles) == 10
        forming_open = NOW.replace(minute=0, second=0)
        assert result.candles[-1].time == forming_open
        assert result.last_closed is not None
        assert result.last_closed.time < forming_open

    def test_short_history_returns_what_exists(self, fake_mt5):
        from tests.fakes.mt5_fake import generate_bars

        fake_mt5.add_rates("GOLD", TIMEFRAME_M15, generate_bars(3, TIMEFRAME_M15, end_epoch=0))
        fake_mt5.symbols["GOLD"] = fake_mt5.symbols["GOLD"]._replace(select=True, visible=True)
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        result = broker.get_candles("GOLD", TimeFrame.M15, 50, include_forming=False)

        assert len(result.candles) == 2  # 3 bars exist, one is forming


class TestFetchTimeframes:
    def test_fetches_all_timeframes_independently(self, broker: MT5Broker):
        checks = fetch_timeframes(
            broker,
            "XAUUSD",
            [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4],
            100,
            validator=validator(),
            now=NOW,
        )
        assert set(checks) == {TimeFrame.M15, TimeFrame.H1, TimeFrame.H4}
        for tf, check in checks.items():
            assert check.valid, f"{tf}: {check.report.summary()}"
            assert check.fresh
            assert check.series is not None and len(check.series.candles) == 100

    def test_one_timeframe_failing_does_not_lose_the_others(self, fake_mt5):
        del fake_mt5.rates[("XAUUSD", TIMEFRAME_H4)]
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        checks = fetch_timeframes(
            broker,
            "XAUUSD",
            [TimeFrame.M15, TimeFrame.H4],
            100,
            validator=validator(),
            now=NOW,
        )

        assert checks[TimeFrame.M15].valid
        assert checks[TimeFrame.M15].fresh
        assert not checks[TimeFrame.H4].valid
        assert checks[TimeFrame.H4].series is None
        assert any(i.code == "SERIES_FETCH_FAILED" for i in checks[TimeFrame.H4].report.errors)
