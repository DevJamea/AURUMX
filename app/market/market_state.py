"""Market state aggregation (spec §52 data-quality gate).

``MarketSnapshot`` is the single read-only object the rest of the system
(analyzers, health monitor, dashboard) consumes.  It answers the question the
risk gate will ask first: *is the market data trustworthy right now?*
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.core.enums import SessionState, TimeFrame
from app.core.models import SeriesCheck, SymbolSpec, TickCheck, ValidationReport


class MarketSnapshot(BaseModel):
    """One coherent view of symbol + tick + candles + data quality."""

    model_config = ConfigDict(allow_inf_nan=False)

    created_at: datetime
    symbol: SymbolSpec | None = None
    tick: TickCheck
    series: dict[TimeFrame, SeriesCheck]
    clock_offset_seconds: float = 0.0
    session_state: SessionState = SessionState.UNKNOWN
    report: ValidationReport

    @property
    def trading_data_ok(self) -> bool:
        """True only when everything the risk gate needs is present, valid and
        fresh.  ``False`` means: block trading (spec §52)."""
        if self.symbol is None:
            return False
        if not (self.tick.valid and self.tick.fresh):
            return False
        if not self.series:
            return False
        if not self.report.ok:
            return False
        return all(check.valid and check.fresh for check in self.series.values())

    @property
    def bid(self) -> float | None:
        return self.tick.tick.bid if self.tick.tick else None

    @property
    def ask(self) -> float | None:
        return self.tick.tick.ask if self.tick.tick else None

    @property
    def spread_points(self) -> float | None:
        return self.tick.spread_points

    def summary(self) -> dict[str, object]:
        return {
            "created_at": self.created_at.isoformat(),
            "symbol": self.symbol.name if self.symbol else None,
            "session_state": self.session_state.value,
            "trading_data_ok": self.trading_data_ok,
            "bid": self.bid,
            "ask": self.ask,
            "spread_points": self.tick.spread_points,
            "tick_valid": self.tick.valid,
            "tick_fresh": self.tick.fresh,
            "series": {
                tf.value: {
                    "valid": check.valid,
                    "fresh": check.fresh,
                    "candles": len(check.series.candles) if check.series else 0,
                    "age_seconds": check.age_seconds,
                }
                for tf, check in self.series.items()
            },
            "issues": [f"{i.level.value} {i.code}: {i.message}" for i in self.report.issues],
        }


def infer_session_state(tick: TickCheck, series: dict[TimeFrame, SeriesCheck]) -> SessionState:
    """Best-effort session inference.

    A fresh tick means the market is quoting (OPEN).  A stale tick with stale
    candles across every timeframe means the market is probably closed
    (weekend/holiday).  Anything mixed is UNKNOWN.
    """
    if tick.valid and tick.fresh:
        return SessionState.OPEN
    if series and all(not check.fresh for check in series.values()):
        return SessionState.CLOSED
    return SessionState.UNKNOWN
