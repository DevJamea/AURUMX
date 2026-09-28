"""MacroAgent — macro/news interface (spec §16, Phase-2 §10).

Interface-only in Phase 2: the agent consumes a ``MacroDataProvider`` (future
implementations: economic calendar, CPI/NFP/FOMC feeds) and **never fabricates
macro information**.  Without a provider it returns NEUTRAL with
``DataQuality.NO_DATA`` and explicit provenance.

Constraints (spec §47): no paid API, no LLM, no internet, no external news
provider required — the whole system runs with the default ``None`` provider.

The agent is deliberately non-directional in v1: high-impact USD events inside
the blackout window produce a *warning* (the future risk gate may block new
entries around them), never a BUY/SELL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.core.enums import AgentDirection, DataQuality


@dataclass(frozen=True)
class MacroEvent:
    """A future economic event (as provided by a data provider)."""

    name: str
    time: datetime
    currency: str = "USD"
    impact: str = "LOW"  # LOW | MEDIUM | HIGH
    notes: str = ""


@runtime_checkable
class MacroDataProvider(Protocol):
    """Contract for future macro data sources (calendar APIs, local files…)."""

    def events_between(self, start: datetime, end: datetime) -> list[MacroEvent]:
        """Events with ``start <= time <= end``.  Must be deterministic."""
        ...


#: How long before/after a high-impact event entries should be treated carefully.
BLACKOUT_WINDOW = timedelta(minutes=30)


class MacroAgent(BaseAgent):
    name = "macro"
    description = "Macro/news context from a provider; NEUTRAL/NO_DATA without one"
    primary_timeframe = None  # not timeframe-bound
    min_candles = 0

    def __init__(self, provider: MacroDataProvider | None = None) -> None:
        self._provider = provider

    def analyze(self, context: MarketContext) -> AgentResult:
        source_time = context.created_at
        warnings: list[str] = []
        events: list[MacroEvent] = []

        if self._provider is None:
            reasons = ["no macro data provider configured"]
            data_quality = DataQuality.NO_DATA
            provenance = "provider=none (no macro information fabricated)"
        else:
            data_quality = DataQuality.OK
            provenance = f"provider={type(self._provider).__name__}"
            try:
                events = list(
                    self._provider.events_between(
                        source_time - BLACKOUT_WINDOW,
                        source_time + BLACKOUT_WINDOW,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - agents must never raise
                reasons = [f"macro provider failed: {type(exc).__name__}: {exc}"]
                warnings.append("macro provider failed — treated as no data")
                events = []
                data_quality = DataQuality.DEGRADED
            else:
                reasons = [
                    f"{len(events)} macro event(s) within ±{int(BLACKOUT_WINDOW.total_seconds() / 60)} min"
                ]
                if not events:
                    reasons = ["no macro events in the blackout window"]

        high_impact = [e for e in events if e.impact.upper() == "HIGH" and e.currency.upper() == "USD"]
        blackout = bool(high_impact)
        if blackout:
            names = ", ".join(f"{e.name}@{e.time.strftime('%H:%M')}" for e in high_impact)
            warnings.append(
                f"high-impact USD event(s) in window: {names} — consider blocking new entries"
            )

        payload = {
            "provenance": provenance,
            "events": [
                {
                    "name": e.name,
                    "time": e.time.isoformat(),
                    "currency": e.currency,
                    "impact": e.impact.upper(),
                }
                for e in events
            ],
            "blackout": blackout,
            "blackout_window_minutes": int(BLACKOUT_WINDOW.total_seconds() / 60),
        }
        return AgentResult(
            agent=self.name,
            direction=AgentDirection.NEUTRAL,
            signal_strength=0.0,
            primary_timeframe=None,
            timeframes_used=[],
            reasons=reasons,
            features=payload,
            warnings=warnings,
            data_quality=data_quality,
            source_time=source_time,
            snapshot_time=context.created_at,
        )
