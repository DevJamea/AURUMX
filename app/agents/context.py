"""Multi-timeframe market context for agents (spec §8, Phase-2 §3).

The context is the ONLY thing agents are allowed to consume.  It wraps a
validated Phase-1 ``MarketSnapshot`` plus:

* an explicit **role per timeframe** (H4 = macro trend, H1 = structure /
  directional context, M15 = entry timing) so agents never invent their own
  timeframe logic;
* a ``TimeframeFeatures`` bundle per usable timeframe, computed once here by
  the shared deterministic feature layer (no per-agent re-computation, no
  global cache — the bundles live on this instance only);
* an optional ``RegimeAssessment`` attached by the pipeline *before* agents
  run (regime-gated agents such as Mean Reversion read it from here).

Immutability: the context is treated as read-only by agents.  ``with_regime``
returns a copy sharing the same feature bundles.  Nothing in this module reads
a wall clock, the network, or a broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agents.features import TimeframeFeatures, compute_features
from app.core.enums import TimeFrame, TimeframeRole
from app.core.models import CandleSeries

if TYPE_CHECKING:  # pragma: no cover
    from app.core.models import MarketSnapshot
    from app.decision.regime import RegimeAssessment

#: Default role mapping (spec §8): H4 macro, H1 structure, M15 entry.
DEFAULT_ROLES: dict[TimeFrame, TimeframeRole] = {
    TimeFrame.D1: TimeframeRole.MACRO,
    TimeFrame.H4: TimeframeRole.MACRO,
    TimeFrame.H1: TimeframeRole.STRUCTURE,
    TimeFrame.M30: TimeframeRole.STRUCTURE,
    TimeFrame.M15: TimeframeRole.ENTRY,
    TimeFrame.M5: TimeframeRole.ENTRY,
    TimeFrame.M1: TimeframeRole.ENTRY,
}


@dataclass(frozen=True)
class MarketContext:
    """Read-only analysis context built from one market snapshot."""

    snapshot: Any  # MarketSnapshot (typed loosely to avoid import cycle)
    roles: dict[TimeFrame, TimeframeRole]
    _features: dict[TimeFrame, TimeframeFeatures]
    _raw_series: dict[TimeFrame, CandleSeries | None]
    _validity: dict[TimeFrame, bool]
    _freshness: dict[TimeFrame, bool]
    regime: Any = None  # RegimeAssessment | None

    # ------------------------------------------------------------------
    @property
    def symbol(self) -> str:
        return self.snapshot.symbol.name if self.snapshot.symbol else ""

    @property
    def created_at(self):
        return self.snapshot.created_at

    # ------------------------------------------------------------------
    def series(self, timeframe: TimeFrame) -> CandleSeries | None:
        """Raw candle series (``None`` when the fetch failed)."""
        return self._raw_series.get(timeframe)

    def features(self, timeframe: TimeFrame) -> TimeframeFeatures | None:
        """Feature bundle for a *valid* timeframe, else ``None``."""
        return self._features.get(timeframe)

    def is_usable(self, timeframe: TimeFrame) -> bool:
        """A timeframe is usable when its data passed validation."""
        return self._validity.get(timeframe, False)

    def is_fresh(self, timeframe: TimeFrame) -> bool:
        return self._freshness.get(timeframe, False)

    def has_timeframe(self, timeframe: TimeFrame) -> bool:
        return timeframe in self._raw_series

    def timeframes_for_role(self, role: TimeframeRole) -> list[TimeFrame]:
        """Usable timeframes assigned to ``role``, best-resolution first."""
        matches = [
            tf
            for tf in sorted(self._features, key=lambda t: t.minutes)
            if self.roles.get(tf) is role
        ]
        return matches

    def role_of(self, timeframe: TimeFrame) -> TimeframeRole | None:
        return self.roles.get(timeframe)

    def primary_for_role(self, role: TimeframeRole) -> TimeFrame | None:
        matches = self.timeframes_for_role(role)
        return matches[0] if matches else None

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "created_at": self.snapshot.created_at.isoformat(),
            "timeframes": {
                tf.value: {
                    "role": self.roles.get(tf).value if self.roles.get(tf) else None,
                    "usable": self.is_usable(tf),
                    "fresh": self.is_fresh(tf),
                    "candles": len(s.candles) if (s := self._raw_series.get(tf)) else 0,
                }
                for tf in self._raw_series
            },
            "trading_data_ok": self.snapshot.trading_data_ok,
            "regime": self.regime.regime.value if self.regime else None,
        }

    # ------------------------------------------------------------------
    def with_regime(self, assessment: RegimeAssessment) -> MarketContext:
        """Copy of this context with the regime attached (shares features)."""
        return MarketContext(
            snapshot=self.snapshot,
            roles=dict(self.roles),
            _features=self._features,
            _raw_series=self._raw_series,
            _validity=self._validity,
            _freshness=self._freshness,
            regime=assessment,
        )


def build_market_context(
    snapshot: MarketSnapshot,
    *,
    roles: dict[TimeFrame, TimeframeRole] | None = None,
    regime: RegimeAssessment | None = None,
) -> MarketContext:
    """Build the analysis context from a snapshot.

    Features are computed for every timeframe whose data passed validation;
    invalid/failed series stay visible (agents report them as degraded data
    rather than silently ignoring them).
    """
    role_map = dict(DEFAULT_ROLES)
    if roles:
        role_map.update(roles)

    raw: dict[TimeFrame, CandleSeries | None] = {}
    validity: dict[TimeFrame, bool] = {}
    freshness: dict[TimeFrame, bool] = {}
    features: dict[TimeFrame, TimeframeFeatures] = {}
    for timeframe, check in snapshot.series.items():
        raw[timeframe] = check.series
        validity[timeframe] = bool(check.series is not None and check.valid)
        freshness[timeframe] = bool(check.fresh)
        if validity[timeframe]:
            features[timeframe] = compute_features(check.series)

    return MarketContext(
        snapshot=snapshot,
        roles=role_map,
        _features=features,
        _raw_series=raw,
        _validity=validity,
        _freshness=freshness,
        regime=regime,
    )
