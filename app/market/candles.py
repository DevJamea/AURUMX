"""Candle validation and multi-timeframe fetching (spec §8, §53).

Two correctness rules dominate this module:

1. **Closed bars only** — ``fetch_timeframes`` never returns the forming bar,
   so agents and backtests see exactly the information that existed at bar
   close.  This is the live-side twin of the backtester's no-look-ahead rule.
2. **Never mix timeframes incorrectly** — each timeframe is fetched and
   validated independently; freshness is judged per-timeframe in the broker
   server's frame of reference (clock-offset corrected).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.brokers.interface import BrokerInterface
from app.core.enums import TimeFrame
from app.core.models import CandleSeries, SeriesCheck, ValidationReport


class CandleValidator:
    """Validates OHLC sanity, ordering and per-timeframe freshness."""

    def __init__(
        self,
        *,
        freshness_multiplier: float = 3.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._multiplier = freshness_multiplier
        self._clock = clock

    # ------------------------------------------------------------------
    def validate(
        self,
        series: CandleSeries,
        *,
        now: datetime | None = None,
        clock_offset_seconds: float = 0.0,
    ) -> SeriesCheck:
        report = self.validate_ohlc(series)
        # ``valid`` = OHLC sanity only; staleness is recorded in the report and
        # surfaced separately through ``fresh`` (mirrors TickValidator).
        sanity_ok = report.ok
        age = self.last_closed_age_seconds(
            series, now=now, clock_offset_seconds=clock_offset_seconds
        )
        fresh = self._check_freshness(series, age, report)
        return SeriesCheck(
            timeframe=series.timeframe,
            series=series,
            valid=sanity_ok,
            fresh=fresh and sanity_ok,
            report=report,
            age_seconds=age,
        )

    # ------------------------------------------------------------------
    def validate_ohlc(self, series: CandleSeries) -> ValidationReport:
        """Per-candle sanity + ordering checks."""
        report = ValidationReport()
        if series.is_empty:
            report.error("SERIES_EMPTY", f"no candles for {series.symbol} {series.timeframe.value}")
            return report

        previous_time: datetime | None = None
        for index, candle in enumerate(series.candles):
            if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
                report.error(
                    "CANDLE_NON_POSITIVE_PRICE",
                    f"candle[{index}] has a non-positive price",
                    index=index,
                )
            if candle.high < candle.low:
                report.error(
                    "CANDLE_HIGH_BELOW_LOW",
                    f"candle[{index}] high {candle.high} < low {candle.low}",
                    index=index,
                )
            if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
                report.error(
                    "CANDLE_OHLC_INCONSISTENT",
                    f"candle[{index}] high/low do not bracket open/close",
                    index=index,
                )
            if candle.tick_volume < 0 or candle.real_volume < 0:
                report.error("CANDLE_NEGATIVE_VOLUME", f"candle[{index}] negative volume", index=index)
            if previous_time is not None:
                if candle.time == previous_time:
                    report.error(
                        "CANDLE_DUPLICATE_TIME",
                        f"candle[{index}] duplicates timestamp {candle.time.isoformat()}",
                        index=index,
                    )
                elif candle.time < previous_time:
                    report.error(
                        "CANDLE_UNORDERED",
                        f"candle[{index}] is older than its predecessor",
                        index=index,
                    )
            previous_time = candle.time
        return report

    # ------------------------------------------------------------------
    def last_closed_age_seconds(
        self,
        series: CandleSeries,
        *,
        now: datetime | None = None,
        clock_offset_seconds: float = 0.0,
    ) -> float | None:
        """Age of the last closed candle: seconds since it *should* have closed.

        For a just-closed bar this is 0..duration; during a data gap it grows
        unbounded, which is exactly what the freshness check must catch.
        """
        last = series.last_closed
        if last is None:
            return None
        now = now or self._clock()
        expected_close = last.time.timestamp() + series.timeframe.seconds
        effective_now = now.timestamp() + clock_offset_seconds
        return effective_now - expected_close

    def _check_freshness(
        self,
        series: CandleSeries,
        age: float | None,
        report: ValidationReport,
    ) -> bool:
        if age is None:
            report.error("SERIES_NO_CLOSED_CANDLE", "series contains no closed candle")
            return False
        duration = float(series.timeframe.seconds)
        if age < -duration:
            report.error(
                "CANDLE_FUTURE_TIMESTAMP",
                f"last closed candle is {-age:.0f}s in the future",
            )
            return False
        limit = duration * self._multiplier
        if age > limit:
            report.error(
                "CANDLES_STALE",
                f"last closed candle is {age / 60:.1f} minutes old "
                f"(limit {limit / 60:.1f} minutes for {series.timeframe.value})",
            )
            return False
        return True


def fetch_timeframes(
    broker: BrokerInterface,
    symbol: str,
    timeframes: list[TimeFrame],
    count: int,
    *,
    validator: CandleValidator,
    now: datetime | None = None,
    clock_offset_seconds: float = 0.0,
) -> dict[TimeFrame, SeriesCheck]:
    """Fetch + validate one closed-candle series per timeframe.

    Broker failures on one timeframe do not lose the others: the failing
    timeframe gets an error report instead of an exception, so a snapshot can
    say "H1 unavailable" while M15/H4 remain usable — and the overall data
    gate still blocks trading.
    """
    checks: dict[TimeFrame, SeriesCheck] = {}
    for timeframe in timeframes:
        report = ValidationReport()
        try:
            series = broker.get_candles(symbol, timeframe, count, include_forming=False)
        except Exception as exc:  # broker-level failure -> reported, not raised
            report.error("SERIES_FETCH_FAILED", f"{type(exc).__name__}: {exc}")
            checks[timeframe] = SeriesCheck(timeframe=timeframe, series=None, report=report)
            continue
        checks[timeframe] = validator.validate(
            series, now=now, clock_offset_seconds=clock_offset_seconds
        )
    return checks
