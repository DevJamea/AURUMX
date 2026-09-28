"""Agent contract (spec §9, Phase-2 §2).

Every deterministic agent implements ``BaseAgent.analyze(context)`` and returns
an ``AgentResult``.  Contract rules enforced by design:

* agents receive a ``MarketContext`` and nothing else — no broker, no
  credentials, no orders, no network, no wall clock, no global state;
* ``signal_strength`` is a **deterministic evidence score in [0, 1]** built
  from documented evidence tables — it is NOT a probability and must never be
  described as one (spec §57);
* ``source_time`` is the last closed candle of the agent's primary timeframe:
  the same snapshot always produces the same result (backtest-safe);
* missing/invalid data ⇒ ``NEUTRAL`` with an explicit data-quality status and
  reasons — never an exception, never a guess.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.agents.context import MarketContext
from app.core.enums import AgentDirection, DataQuality, TimeFrame


class AgentResult(BaseModel):
    """Provenance-carrying output of one agent run (spec §14).

    ``features`` is JSON-safe (floats/strings/bools/None/lists of primitives)
    so results can be stored in the decision journal unchanged.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    agent: str
    direction: AgentDirection = AgentDirection.NEUTRAL
    signal_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_timeframe: TimeFrame | None = None
    timeframes_used: list[TimeFrame] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    data_quality: DataQuality = DataQuality.OK
    #: time of the decision candle (last closed candle of the primary tf)
    source_time: datetime
    #: snapshot timestamp the analysis is derived from (deterministic)
    snapshot_time: datetime

    @property
    def is_actionable(self) -> bool:
        return self.direction is not AgentDirection.NEUTRAL and self.signal_strength > 0


class BaseAgent(ABC):
    """Base class for all deterministic analysis agents."""

    #: stable identifier used in config, logs and the journal
    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    #: the timeframe whose last closed candle is the decision candle
    primary_timeframe: ClassVar[TimeFrame | None] = None
    #: minimum closed candles required on the primary timeframe
    min_candles: ClassVar[int] = 0

    @abstractmethod
    def analyze(self, context: MarketContext) -> AgentResult:
        """Analyze the context.  Must be pure: same context in, same result out."""

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def make_result(
        self,
        context: MarketContext,
        *,
        direction: AgentDirection,
        signal_strength: float,
        reasons: list[str],
        features: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        data_quality: DataQuality = DataQuality.OK,
        timeframes_used: list[TimeFrame] | None = None,
    ) -> AgentResult:
        primary = self.primary_timeframe
        primary_features = context.features(primary) if primary else None
        source_time = (
            primary_features.last_time
            if primary_features is not None and primary_features.last_time is not None
            else context.created_at
        )
        return AgentResult(
            agent=self.name,
            direction=direction,
            signal_strength=max(0.0, min(1.0, signal_strength)),
            primary_timeframe=primary,
            timeframes_used=timeframes_used or ([primary] if primary else []),
            reasons=reasons,
            features=features or {},
            warnings=warnings or [],
            data_quality=data_quality,
            source_time=source_time,
            snapshot_time=context.created_at,
        )

    def unavailable_result(
        self,
        context: MarketContext,
        *,
        quality: DataQuality,
        reason: str,
        warnings: list[str] | None = None,
    ) -> AgentResult:
        """Neutral result for missing/insufficient/invalid primary data."""
        return self.make_result(
            context,
            direction=AgentDirection.NEUTRAL,
            signal_strength=0.0,
            reasons=[reason],
            data_quality=quality,
            warnings=warnings,
        )

    def abnormal_candle_result(
        self,
        context: MarketContext,
        *,
        quality: DataQuality,
        warnings: list[str] | None = None,
    ) -> AgentResult:
        """Neutral result when the decision candle is physically implausible
        (bad-tick guard, spec Phase-2 §18): hostile data must never
        manufacture a signal."""
        return self.make_result(
            context,
            direction=AgentDirection.NEUTRAL,
            signal_strength=0.0,
            reasons=["abnormal decision candle (possible bad tick) — signal suppressed"],
            features={"abnormal_candle": True},
            data_quality=DataQuality.DEGRADED if quality is DataQuality.OK else quality,
            warnings=(warnings or []) + ["abnormal candle — possible bad tick"],
        )

    # ------------------------------------------------------------------
    def primary_features(self, context: MarketContext):
        """Feature bundle of the primary timeframe, with quality context.

        Returns ``(features, quality, reasons)`` where ``features`` is ``None``
        when the agent must return a neutral result instead.
        """
        primary = self.primary_timeframe
        if primary is None:  # pragma: no cover - contract misuse
            return None, DataQuality.INVALID, ["agent has no primary timeframe"]
        if not context.has_timeframe(primary):
            return None, DataQuality.INVALID, [f"timeframe {primary.value} not in snapshot"]
        if not context.is_usable(primary):
            return None, DataQuality.INVALID, [
                f"timeframe {primary.value} data failed validation"
            ]
        features = context.features(primary)
        if features is None:  # pragma: no cover - defensive
            return None, DataQuality.INVALID, [f"timeframe {primary.value} unusable"]
        if features.length < self.min_candles:
            return None, DataQuality.INSUFFICIENT, [
                f"{primary.value}: {features.length} candles < required {self.min_candles}"
            ]
        warnings: list[str] = []
        quality = DataQuality.OK
        if not context.is_fresh(primary):
            warnings.append(f"{primary.value} data is stale")
            quality = DataQuality.DEGRADED
        return features, quality, warnings
