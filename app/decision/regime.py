"""Deterministic market regime detection (spec §18, Phase-2 §11).

The regime classifies the market from H1 features (macro-confirmed by H4) and
decides **which agents are allowed to influence decisions** — never a flat
average across all agents.

Detection rules (ordered, all deterministic):

1. Insufficient H1 data (< 60 candles) → ``UNCERTAIN``.
2. **Trend**: H1 EMA alignment (ATR-buffered) + ADX ≥ 25 + confirmation from
   H4 alignment or H1 structure ⇒ ``TREND_UP`` / ``TREND_DOWN``.  H4 opposing
   ⇒ ``UNCERTAIN`` with the conflict recorded.
3. **Range**: ADX ≤ 20 with flat/mixed EMA alignment and non-trending
   structure ⇒ ``RANGE``; the volatility axis decides the label when extreme:
   ``LOW_VOLATILITY`` (compression) or ``HIGH_VOLATILITY`` (expansion).
4. Anything else (mixed evidence) ⇒ ``UNCERTAIN`` with conflicts listed.

The volatility level (LOW/NORMAL/HIGH/EXTREME) is carried as a **separate
axis** on the assessment — a market can be TREND_UP *and* EXTREME volatility;
agents and the risk gate consume both.

Relevance matrix: each regime maps every agent to HIGH / NORMAL / REDUCED /
DISABLED.  Weight multipliers are documented and consumed by the synthesis
contract (Phase 3 applies them in the full decision engine).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from app.agents.context import MarketContext
from app.agents.features import classify_volatility
from app.core.enums import (
    AgentDirection,
    DataQuality,
    MarketRegime,
    StructureBias,
    TimeFrame,
    VolatilityLevel,
)

#: Relevance of an agent's evidence under a given regime.
AGENT_RELEVANCE_WEIGHTS: dict[str, float] = {
    "HIGH": 1.25,
    "NORMAL": 1.0,
    "REDUCED": 0.5,
    "DISABLED": 0.0,
}


class RegimeAssessment(BaseModel):
    """Regime + volatility axis + evidence, as-of a decision candle."""

    model_config = ConfigDict(allow_inf_nan=False)

    regime: MarketRegime
    volatility: VolatilityLevel = VolatilityLevel.NORMAL
    as_of: datetime
    data_quality: DataQuality = DataQuality.OK
    evidence: list[str] = []
    conflicts: list[str] = []

    def summary(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "volatility": self.volatility.value,
            "as_of": self.as_of.isoformat(),
            "evidence": self.evidence,
            "conflicts": self.conflicts,
        }


#: Regime → agent → relevance.  Volatility agent is always NORMAL (contextual).
REGIME_RELEVANCE: dict[MarketRegime, dict[str, str]] = {
    MarketRegime.TREND_UP: {
        "trend": "HIGH",
        "structure": "HIGH",
        "momentum": "HIGH",
        "liquidity": "NORMAL",
        "volatility": "NORMAL",
        "mean_reversion": "DISABLED",
        "macro": "NORMAL",
    },
    MarketRegime.TREND_DOWN: {
        "trend": "HIGH",
        "structure": "HIGH",
        "momentum": "HIGH",
        "liquidity": "NORMAL",
        "volatility": "NORMAL",
        "mean_reversion": "DISABLED",
        "macro": "NORMAL",
    },
    MarketRegime.RANGE: {
        "trend": "REDUCED",
        "structure": "NORMAL",
        "momentum": "NORMAL",
        "liquidity": "NORMAL",
        "volatility": "NORMAL",
        "mean_reversion": "HIGH",
        "macro": "NORMAL",
    },
    MarketRegime.HIGH_VOLATILITY: {
        "trend": "REDUCED",
        "structure": "NORMAL",
        "momentum": "REDUCED",
        "liquidity": "NORMAL",
        "volatility": "HIGH",
        "mean_reversion": "DISABLED",
        "macro": "NORMAL",
    },
    MarketRegime.LOW_VOLATILITY: {
        "trend": "REDUCED",
        "structure": "NORMAL",
        "momentum": "REDUCED",
        "liquidity": "NORMAL",
        "volatility": "HIGH",
        "mean_reversion": "NORMAL",
        "macro": "NORMAL",
    },
    MarketRegime.UNCERTAIN: {
        "trend": "REDUCED",
        "structure": "REDUCED",
        "momentum": "REDUCED",
        "liquidity": "REDUCED",
        "volatility": "NORMAL",
        "mean_reversion": "DISABLED",
        "macro": "NORMAL",
    },
}


def relevance_for(regime: MarketRegime) -> dict[str, str]:
    """Per-agent relevance under ``regime`` (default NORMAL for unknown agents)."""
    matrix = REGIME_RELEVANCE.get(regime, {})
    return dict(matrix)


def relevance_of(regime: MarketRegime, agent: str) -> str:
    return REGIME_RELEVANCE.get(regime, {}).get(agent, "NORMAL")


class RegimeDetector:
    """Deterministic regime classifier over a ``MarketContext``."""

    #: minimum H1 candles for a confident assessment
    MIN_H1_CANDLES: ClassVar[int] = 60

    def detect(self, context: MarketContext) -> RegimeAssessment:
        h1 = context.features(TimeFrame.H1)
        if h1 is None or h1.length < self.MIN_H1_CANDLES:
            return RegimeAssessment(
                regime=MarketRegime.UNCERTAIN,
                as_of=context.created_at,
                data_quality=DataQuality.INSUFFICIENT,
                evidence=["H1 features unavailable or insufficient"],
            )

        h4 = context.features(TimeFrame.H4)
        volatility = classify_volatility(h1)
        adx = h1.adx or 0.0
        align_h1 = h1.ema_alignment or "flat"
        bias = h1.bias
        align_h4 = h4.ema_alignment if h4 is not None else None

        evidence: list[str] = []
        conflicts: list[str] = []
        evidence.append(f"H1 EMA alignment: {align_h1}")
        evidence.append(f"H1 ADX: {adx:.1f}")
        evidence.append(f"H1 structure bias: {bias.value}")
        if h4 is not None:
            evidence.append(f"H4 EMA alignment: {align_h4}")
        else:
            conflicts.append("H4 macro timeframe unavailable")

        # ---- trend ----------------------------------------------------------
        directional_up = align_h1 == "up" and adx >= 25
        directional_down = align_h1 == "down" and adx >= 25
        if directional_up or directional_down:
            direction = "up" if directional_up else "down"
            h4_opposes = align_h4 is not None and align_h4 == ("down" if direction == "up" else "up")
            structure_confirms = bias is (
                StructureBias.UPTREND if direction == "up" else StructureBias.DOWNTREND
            )
            if h4_opposes:
                conflicts.append(f"H4 alignment opposes H1 ({align_h4} vs {align_h1})")
                return RegimeAssessment(
                    regime=MarketRegime.UNCERTAIN,
                    volatility=volatility,
                    as_of=h1.last_time or context.created_at,
                    data_quality=DataQuality.OK if h4 is not None else DataQuality.DEGRADED,
                    evidence=evidence,
                    conflicts=conflicts,
                )
            if align_h4 == direction or structure_confirms or h4 is None:
                regime = MarketRegime.TREND_UP if direction == "up" else MarketRegime.TREND_DOWN
                if align_h4 == direction:
                    evidence.append("H4 macro alignment confirms")
                if structure_confirms:
                    evidence.append("H1 structure confirms")
                return RegimeAssessment(
                    regime=regime,
                    volatility=volatility,
                    as_of=h1.last_time or context.created_at,
                    data_quality=DataQuality.OK,
                    evidence=evidence,
                    conflicts=conflicts,
                )
            # directional on H1 but neither H4 nor structure confirms
            conflicts.append("H1 directional but H4/structure do not confirm")
            return RegimeAssessment(
                regime=MarketRegime.UNCERTAIN,
                volatility=volatility,
                as_of=h1.last_time or context.created_at,
                evidence=evidence,
                conflicts=conflicts,
            )

        # ---- range / volatility-dominated ------------------------------------
        range_like = (
            adx <= 20
            and align_h1 in ("flat", "mixed")
            and bias in (StructureBias.RANGE, StructureBias.UNKNOWN)
        )
        if range_like:
            evidence.append("ADX ≤ 20 with flat EMAs and non-trending structure")
            regime = MarketRegime.RANGE
            if volatility is VolatilityLevel.LOW:
                regime = MarketRegime.LOW_VOLATILITY
                evidence.append("volatility compressed (LOW)")
            elif volatility in (VolatilityLevel.HIGH, VolatilityLevel.EXTREME):
                regime = MarketRegime.HIGH_VOLATILITY
                evidence.append("volatility elevated in range — labelled HIGH_VOLATILITY")
            return RegimeAssessment(
                regime=regime,
                volatility=volatility,
                as_of=h1.last_time or context.created_at,
                data_quality=DataQuality.OK,
                evidence=evidence,
                conflicts=conflicts,
            )

        # ---- mixed ------------------------------------------------------------
        conflicts.append(
            f"mixed evidence (alignment={align_h1}, ADX={adx:.1f}, bias={bias.value})"
        )
        return RegimeAssessment(
            regime=MarketRegime.UNCERTAIN,
            volatility=volatility,
            as_of=h1.last_time or context.created_at,
            evidence=evidence,
            conflicts=conflicts,
        )


def weighted_strength(direction: AgentDirection, signal_strength: float, relevance: str) -> float:
    """Signed, regime-weighted evidence contribution (helper for Phase 3)."""
    if direction is AgentDirection.NEUTRAL or relevance == "DISABLED":
        return 0.0
    weight = AGENT_RELEVANCE_WEIGHTS.get(relevance, 1.0)
    sign = 1.0 if direction is AgentDirection.BUY else -1.0
    return sign * signal_strength * weight
