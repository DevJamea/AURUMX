"""VolatilityAgent — contextual volatility classification (spec §14, Phase-2 §8).

Primary timeframe: **H1**.

This agent is **contextual by design**: it always returns ``NEUTRAL`` with
``signal_strength = 0`` — high volatility is never a BUY/SELL reason on its
own.  Its output (level LOW/NORMAL/HIGH/EXTREME, expansion/contraction state,
ATR/BB metrics) feeds the regime detector and, later, the risk gate's stop
distance and sizing logic.

Classification thresholds (shared with the regime detector via
``features.classify_volatility``):

* EXTREME: ATR ≥ 2.5× its 100-bar median, or BB width in the top 3%
* HIGH:    ATR ≥ 1.5× median, or BB width in the top 15%
* LOW:     ATR ≤ 0.6× median, or BB width in the bottom 10%
* NORMAL:  otherwise
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.features import (
    classify_volatility,
    safe_float,
    volatility_state,
)
from app.core.enums import AgentDirection, DataQuality, TimeFrame


class VolatilityAgent(BaseAgent):
    name = "volatility"
    description = "ATR/BB-width volatility classification (contextual, never directional)"
    primary_timeframe = TimeFrame.H1
    min_candles = 30  # ATR(14) + BB(20) warm-up; ranking history may still be short

    def analyze(self, context: MarketContext) -> AgentResult:
        features, quality, warnings = self.primary_features(context)
        if features is None:
            return self.unavailable_result(
                context, quality=quality, reason=warnings and warnings[0] or "no data"
            )

        level = classify_volatility(features)
        state = volatility_state(features)

        reasons = [
            f"ATR {features.atr:.2f} ({features.atr_pct * 100:.2f}% of price)"
            if features.atr
            else "ATR unavailable",
            f"ATR ratio vs 100-bar median: {features.atr_ratio:.2f}x"
            if features.atr_ratio
            else "ATR ratio unavailable",
            f"BB width percentile: {features.bb_width_pct:.2f}"
            if features.bb_width_pct is not None
            else "BB width percentile unavailable (short history)",
            f"volatility {state.value.lower()}",
        ]
        if level.value in ("HIGH", "EXTREME"):
            warnings = list(warnings) + [
                f"{level.value} volatility — risk controls should widen stops / reduce size"
            ]
        # full ranking confidence needs a real percentile history: 60 ranked
        # values after warm-up (BB(20) -> ~80 bars, ATR(14) -> ~74 bars)
        ranking_short = (
            features.bb_width_pct is None
            or features.atr_ratio is None
            or features.length < 80
        )
        if ranking_short:
            if quality is DataQuality.OK:
                quality = DataQuality.DEGRADED
            warnings = list(warnings) + ["insufficient history for full volatility ranking"]

        payload = {
            "level": level.value,
            "state": state.value,
            "atr": safe_float(features.atr, 2),
            "atr_pct": safe_float(features.atr_pct, 5),
            "atr_ratio": safe_float(features.atr_ratio, 3),
            "bb_width": safe_float(features.bb_width, 5),
            "bb_width_pct": safe_float(features.bb_width_pct, 3),
            "bb_expansion": safe_float(features.bb_expansion, 3),
        }
        return self.make_result(
            context,
            direction=AgentDirection.NEUTRAL,
            signal_strength=0.0,
            reasons=reasons,
            features=payload,
            warnings=warnings,
            data_quality=quality,
            timeframes_used=[TimeFrame.H1],
        )
