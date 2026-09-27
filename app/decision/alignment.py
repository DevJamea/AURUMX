"""Multi-timeframe alignment (Phase-3 §3).

Explicit role model — H4 = primary context, H1 = structure/directional,
M15 = entry timing.  The engine refuses to translate an H4/H1 signal into a
proposal when the entry timeframe strongly opposes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.context import MarketContext
from app.core.enums import AgentDirection, TimeFrame, TimeframeRole
from app.decision.config import DecisionEngineConfig


@dataclass(frozen=True)
class TimeframeRead:
    """One timeframe's directional read."""

    timeframe: TimeFrame
    role: TimeframeRole | None
    direction: AgentDirection  # NEUTRAL when flat/mixed/unknown
    strong: bool  # |EMA20-EMA50| beyond the configured ATR multiple
    gap_atr: float | None
    candles: int


@dataclass(frozen=True)
class TimeframeAlignment:
    """Weighted multi-timeframe agreement with a proposed direction."""

    reads: tuple[TimeframeRead, ...] = field(default_factory=tuple)
    missing: tuple[TimeFrame, ...] = field(default_factory=tuple)
    score: float = 0.0  # weighted agreement in [0, 1]
    opposing: tuple[TimeFrame, ...] = field(default_factory=tuple)
    agreeing: tuple[TimeFrame, ...] = field(default_factory=tuple)
    #: True when an expected timeframe was absent and the score was computed
    #: over the remaining ones (policy-gated: the engine only allows this
    #: when decision.timeframe.allow_missing_h4_renormalization is set)
    renormalized: bool = False
    #: weights actually in force after renormalization — (timeframe, weight)
    #: pairs summing to 1.0 over the present timeframes
    effective_weights: tuple[tuple[TimeFrame, float], ...] = field(default_factory=tuple)

    @property
    def entry_timeframe_opposes(self) -> bool:
        """True when any ENTRY-role timeframe (M15) actively opposes."""
        return bool(self.opposing)

    def summary(self) -> dict:
        """JSON-safe summary for the Decision / journal record."""
        return {
            "score": self.score,
            "agreeing": [tf.value for tf in self.agreeing],
            "opposing": [tf.value for tf in self.opposing],
            "missing": [tf.value for tf in self.missing],
            "renormalized": self.renormalized,
            "effective_weights": {tf.value: round(w, 6) for tf, w in self.effective_weights},
        }


def _read(context: MarketContext, timeframe: TimeFrame, config: DecisionEngineConfig) -> TimeframeRead:
    features = context.features(timeframe)
    role = context.role_of(timeframe)
    if features is None:
        return TimeframeRead(timeframe, role, AgentDirection.NEUTRAL, False, None, 0)

    direction = AgentDirection.NEUTRAL
    gap_atr: float | None = None
    if features.ema_alignment in ("up", "down") and features.atr:
        gap_atr = abs(features.ema20 - features.ema50) / features.atr
        direction = (
            AgentDirection.BUY if features.ema_alignment == "up" else AgentDirection.SELL
        )
    strong = bool(gap_atr is not None and gap_atr >= config.strong_alignment_atr)
    return TimeframeRead(timeframe, role, direction, strong, gap_atr, features.length)


def compute_alignment(
    context: MarketContext, direction: AgentDirection, config: DecisionEngineConfig
) -> TimeframeAlignment:
    """Weighted agreement of H4/H1/M15 with ``direction``.

    Score = sum(weights of agreeing directional timeframes) /
            sum(weights of all directional timeframes).  Flat/mixed
    timeframes are neutral: they neither support nor oppose, but the entry
    timeframe actively opposing is recorded in ``opposing`` and fails the
    engine's alignment gate by dropping the score below the threshold.
    Missing expected timeframes are listed in ``missing`` (never silently
    ignored).
    """
    expected = (TimeFrame.H4, TimeFrame.H1, TimeFrame.M15)
    reads: list[TimeframeRead] = []
    missing: list[TimeFrame] = []
    for timeframe in expected:
        if not context.has_timeframe(timeframe) or context.features(timeframe) is None:
            missing.append(timeframe)
            continue
        reads.append(_read(context, timeframe, config))

    agreeing: list[TimeFrame] = []
    opposing: list[TimeFrame] = []
    weight_sum = 0.0
    agree_sum = 0.0
    for read in reads:
        weight = config.timeframe_weights.weight_of(read.timeframe)
        if read.direction is AgentDirection.NEUTRAL or weight <= 0:
            continue
        weight_sum += weight
        if read.direction is direction:
            agree_sum += weight
            agreeing.append(read.timeframe)
        else:
            opposing.append(read.timeframe)

    score = agree_sum / weight_sum if weight_sum > 0 else 0.0

    # effective weights in force: the configured weights renormalized over
    # the expected timeframes that are actually present
    present = [tf for tf in expected if tf not in missing]
    present_weight = {tf: config.timeframe_weights.weight_of(tf) for tf in present}
    total_weight = sum(present_weight.values())
    effective = tuple(
        (tf, w / total_weight) for tf, w in present_weight.items()
    ) if total_weight > 0 else ()

    return TimeframeAlignment(
        reads=tuple(reads),
        missing=tuple(missing),
        score=round(score, 6),
        opposing=tuple(opposing),
        agreeing=tuple(agreeing),
        renormalized=bool(missing),
        effective_weights=effective,
    )
