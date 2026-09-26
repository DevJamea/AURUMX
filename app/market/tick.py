"""Tick validation (spec §53 edge cases: zero tick, ask < bid, stale tick).

The validator *reports* rather than raises: a bad tick becomes a ``TickCheck``
with ``valid=False`` and the reasons, which the health monitor / risk gate use
to block trading without crashing the worker.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.core.models import MarketTick, SymbolSpec, TickCheck, ValidationReport

#: How far a tick timestamp may be in the future before it is treated as a
#: data error rather than harmless clock skew (seconds).
_FUTURE_TOLERANCE_SECONDS = 60.0


class TickValidator:
    """Validates raw ticks against sanity and freshness rules."""

    def __init__(
        self,
        *,
        max_age_seconds: float = 60.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._max_age = max_age_seconds
        self._clock = clock

    def validate(
        self,
        tick: MarketTick | None,
        *,
        symbol_spec: SymbolSpec | None = None,
        now: datetime | None = None,
        clock_offset_seconds: float = 0.0,
    ) -> TickCheck:
        """Validate ``tick``.

        ``clock_offset_seconds`` is the estimated broker-server clock offset
        (server epoch minus true UTC); it is applied so freshness is judged in
        the server's frame of reference.
        """
        report = ValidationReport()
        now = now or self._clock()

        if tick is None:
            report.error("TICK_MISSING", "no tick available")
            return TickCheck(report=report)

        if tick.bid <= 0 or tick.ask <= 0:
            report.error("TICK_ZERO_PRICE", f"bid={tick.bid}, ask={tick.ask} (zero/negative)")
        if tick.ask < tick.bid:
            report.error("TICK_INVERTED", f"ask ({tick.ask}) < bid ({tick.bid})")

        # ``valid`` = quote sanity (zero/inverted/future).  Staleness is a
        # freshness finding recorded in the report but reported separately via
        # ``fresh`` — a stale tick is still *sane* data, just unusable now.
        sanity_ok = report.ok

        effective_now = now.timestamp() + clock_offset_seconds
        age = effective_now - tick.time.timestamp()

        if age < -_FUTURE_TOLERANCE_SECONDS:
            report.error("TICK_FUTURE_TIMESTAMP", f"tick is {-age:.1f}s in the future")
            fresh = False
        else:
            fresh = age <= self._max_age
            if not fresh:
                report.error(
                    "TICK_STALE", f"tick age {age:.1f}s exceeds max {self._max_age:.1f}s"
                )

        spread = tick.spread if tick.bid > 0 and tick.ask >= tick.bid else None
        spread_points = (
            spread / symbol_spec.point
            if spread is not None and symbol_spec is not None and symbol_spec.point > 0
            else None
        )

        return TickCheck(
            tick=tick,
            valid=sanity_ok,
            fresh=fresh and sanity_ok,
            report=report,
            spread=spread,
            spread_points=spread_points,
            age_seconds=age,
        )
