"""Synthesis input/output contract (spec §17, Phase-2 §12).

Phase 2 defines the contract and a minimal, fully deterministic reference
aggregation.  Phase 3 replaces the aggregation internals with the full
decision engine (configurable per-agent weights, entry/SL/TP proposal, HOLD
bias) — the *contract* stays stable.

Key property (spec §17, Phase-2 §17): **disagreement is preserved.**  The
output carries every agent stance (direction, strength, regime relevance,
weighted strength) plus explicit supporting/opposing/neutral groupings and
conflict strings.  A BUY output never erases the fact that, say, Momentum
voted SELL — that information is exactly what Phase 3's decision engine and
the decision journal need.

The reference aggregation (documented, deterministic):

* every agent's strength is multiplied by its regime relevance weight
  (HIGH 1.25 / NORMAL 1.0 / REDUCED 0.5 / DISABLED 0);
* ``buy_score`` / ``sell_score`` are the sums of weighted strengths per side;
* ``action`` = BUY when (buy_score − sell_score) ≥ 0.80 and buy_score ≥ 1.00,
  SELL mirrored, otherwise HOLD — placeholder thresholds that Phase 3 will
  make configurable and regime-aware;
* ``aggregate_score`` is the normalized score of the chosen action.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import AgentResult
from app.core.enums import AgentDirection, DataQuality, DecisionAction
from app.decision.regime import (
    AGENT_RELEVANCE_WEIGHTS,
    RegimeAssessment,
    relevance_of,
)

#: Placeholder action thresholds (Phase 3 makes them configurable).
ACTION_NET_THRESHOLD = 0.80
ACTION_SIDE_THRESHOLD = 1.00


class AgentStance(BaseModel):
    """One agent's evidence, regime-weighted — the unit of preserved disagreement."""

    model_config = ConfigDict(allow_inf_nan=False)

    agent: str
    direction: AgentDirection
    signal_strength: float = Field(ge=0.0, le=1.0)
    relevance: str
    weighted_strength: float
    data_quality: DataQuality
    reasons: list[str] = []

    @property
    def side(self) -> str:
        return self.direction.value


class SynthesisInput(BaseModel):
    """Everything the synthesis is allowed to know."""

    model_config = ConfigDict(allow_inf_nan=False)

    symbol: str
    timestamp: datetime
    regime: RegimeAssessment
    agent_results: list[AgentResult]
    data_quality_ok: bool = True
    context_summary: dict[str, Any] = Field(default_factory=dict)


class SynthesisOutput(BaseModel):
    """Aggregated evidence — NOT a trade order and never executed directly."""

    model_config = ConfigDict(allow_inf_nan=False)

    action: DecisionAction
    aggregate_score: float = Field(ge=0.0, le=1.0)
    buy_score: float
    sell_score: float
    supporting: list[AgentStance] = Field(default_factory=list)
    opposing: list[AgentStance] = Field(default_factory=list)
    neutral: list[AgentStance] = Field(default_factory=list)
    disabled: list[AgentStance] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    @property
    def has_disagreement(self) -> bool:
        return bool(self.opposing) and self.action is not DecisionAction.HOLD


def synthesize(inputs: SynthesisInput) -> SynthesisOutput:
    """Deterministic reference aggregation of agent evidence."""
    regime = inputs.regime.regime
    stances: list[AgentStance] = []
    for result in inputs.agent_results:
        relevance = relevance_of(regime, result.agent)
        weight = AGENT_RELEVANCE_WEIGHTS.get(relevance, 1.0)
        if result.direction is AgentDirection.NEUTRAL:
            weighted = 0.0
        else:
            weighted = result.signal_strength * weight
        stances.append(
            AgentStance(
                agent=result.agent,
                direction=result.direction,
                signal_strength=result.signal_strength,
                relevance=relevance,
                weighted_strength=round(weighted, 6),
                data_quality=result.data_quality,
                reasons=list(result.reasons),
            )
        )

    buy_score = sum(
        s.weighted_strength for s in stances if s.direction is AgentDirection.BUY
    )
    sell_score = sum(
        s.weighted_strength for s in stances if s.direction is AgentDirection.SELL
    )
    net = buy_score - sell_score

    action = DecisionAction.HOLD
    if net >= ACTION_NET_THRESHOLD and buy_score >= ACTION_SIDE_THRESHOLD:
        action = DecisionAction.BUY
    elif -net >= ACTION_NET_THRESHOLD and sell_score >= ACTION_SIDE_THRESHOLD:
        action = DecisionAction.SELL

    supporting_side = (
        AgentDirection.BUY if action is DecisionAction.BUY else AgentDirection.SELL
    )
    supporting = [
        s for s in stances
        if action is not DecisionAction.HOLD and s.direction is supporting_side and s.weighted_strength > 0
    ]
    opposing = [
        s for s in stances
        if action is not DecisionAction.HOLD
        and s.direction is not AgentDirection.NEUTRAL
        and s.direction is not supporting_side
        and s.weighted_strength > 0
    ]
    # disagreement must survive even on HOLD: strongest minority side vs majority
    if action is DecisionAction.HOLD:
        majority = AgentDirection.BUY if buy_score >= sell_score else AgentDirection.SELL
        minority = AgentDirection.SELL if majority is AgentDirection.BUY else AgentDirection.BUY
        supporting = [s for s in stances if s.direction is majority and s.weighted_strength > 0]
        opposing = [s for s in stances if s.direction is minority and s.weighted_strength > 0]
    neutral = [s for s in stances if s.direction is AgentDirection.NEUTRAL]
    disabled = [s for s in stances if s.relevance == "DISABLED"]

    conflicts = []
    if supporting and opposing:
        supp = ", ".join(f"{s.agent}({s.direction.value} {s.weighted_strength:.2f})" for s in supporting)
        opp = ", ".join(f"{s.agent}({s.direction.value} {s.weighted_strength:.2f})" for s in opposing)
        conflicts.append(f"disagreement: {supp} vs {opp}")
    if stances_with_bad_quality := [
        s for s in stances if s.data_quality in (DataQuality.INVALID, DataQuality.INSUFFICIENT)
    ]:
        conflicts.append(
            "degraded agent data: " + ", ".join(s.agent for s in stances_with_bad_quality)
        )

    reasons = [
        f"regime {regime.value} (volatility {inputs.regime.volatility.value})",
        f"buy_score={buy_score:.2f} sell_score={sell_score:.2f} net={net:+.2f}",
    ]
    if action is DecisionAction.HOLD:
        reasons.append("evidence below action thresholds — HOLD is a valid decision")
    warnings = list(inputs.regime.conflicts)
    if not inputs.data_quality_ok:
        warnings.append("market snapshot not trading-grade")

    total = buy_score + sell_score
    aggregate = (max(buy_score, sell_score) / total) if total > 0 else 0.0

    return SynthesisOutput(
        action=action,
        aggregate_score=round(min(1.0, aggregate), 6),
        buy_score=round(buy_score, 6),
        sell_score=round(sell_score, 6),
        supporting=supporting,
        opposing=opposing,
        neutral=neutral,
        disabled=disabled,
        reasons=reasons,
        warnings=warnings,
        conflicts=conflicts,
    )
