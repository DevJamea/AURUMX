"""StructureAgent — market structure from confirmed swings (spec §12, Phase-2 §6).

Primary timeframe: **H1**; macro context: **H4**.

The agent consumes the shared feature layer's deterministic swing/structure
computation:

* swings are **confirmed fractals** (left=right=2): a swing at bar ``i`` is
  only knowable from bar ``i+2`` onward — the **confirmation delay is 2 closed
  candles** and it is part of the agent's output (`confirmation_delay` feature)
  so downstream phases can account for it;
* BOS = close breaks a confirmed swing level in the direction of the
  prevailing structure bias (continuation); CHOCH = break against it (change
  of character).  Only closed candles and already-confirmed levels are used —
  nothing retroactive.

Direction rules:

* UPTREND bias (HH/HL) with a recent (≤ 20 candles) BOS_UP  → BUY
* DOWNTREND bias (LH/LL) with a recent BOS_DOWN            → SELL
* recent CHOCH (either direction)                          → direction of the
  CHOCH at reduced strength (early reversal, flagged)
* bias without a fresh break                               → NEUTRAL

Strength (documented): ``base + 0.20·recency + 0.20·H4-agreement +
0.15·sequence-quality`` where base = 0.45 (BOS with bias) or 0.35 (CHOCH).
The result also carries the **invalidation level** (last HL for longs, last
LH for shorts) for later stop-loss logic.
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.features import safe_float
from app.core.enums import AgentDirection, StructureEventType, TimeFrame

#: how recent a structure event must be to drive a signal (closed candles)
RECENT_EVENT_WINDOW = 20


class StructureAgent(BaseAgent):
    name = "structure"
    description = "HH/HL/LH/LL swing structure with BOS/CHOCH detection"
    primary_timeframe = TimeFrame.H1
    min_candles = 30

    def analyze(self, context: MarketContext) -> AgentResult:
        features, quality, warnings = self.primary_features(context)
        if features is None:
            return self.unavailable_result(
                context, quality=quality, reason=warnings and warnings[0] or "no data"
            )
        if features.last_candle_anomaly:
            return self.abnormal_candle_result(
                context, quality=quality, warnings=warnings
            )

        used = [TimeFrame.H1]
        h4 = context.features(TimeFrame.H4)
        if h4 is not None:
            used.append(TimeFrame.H4)

        bias = features.bias
        last_event = features.last_event
        event_age = (
            features.length - 1 - last_event.index if last_event is not None else None
        )
        recent = event_age is not None and event_age <= RECENT_EVENT_WINDOW

        base_payload = self._features(features, h4)

        if last_event is None:
            return self._neutral(
                context, features, quality, warnings, used,
                reasons=["no confirmed structure breaks yet"],
                payload=base_payload,
            )

        kind = last_event.kind
        direction = AgentDirection.NEUTRAL
        strength = 0.0
        reasons: list[str] = []

        if kind is StructureEventType.BOS_UP and bias.value == "UPTREND":
            direction = AgentDirection.BUY
            strength = 0.45
            reasons = [
                f"bullish BOS: close broke confirmed swing high {last_event.level:.2f}",
                "H1 structure bias UPTREND (HH/HL)",
            ]
        elif kind is StructureEventType.BOS_DOWN and bias.value == "DOWNTREND":
            direction = AgentDirection.SELL
            strength = 0.45
            reasons = [
                f"bearish BOS: close broke confirmed swing low {last_event.level:.2f}",
                "H1 structure bias DOWNTREND (LH/LL)",
            ]
        elif kind in (StructureEventType.CHOCH_UP, StructureEventType.CHOCH_DOWN):
            direction = (
                AgentDirection.BUY if kind is StructureEventType.CHOCH_UP else AgentDirection.SELL
            )
            strength = 0.35
            reasons = [
                f"CHOCH to {'bullish' if direction is AgentDirection.BUY else 'bearish'}: "
                f"close broke {last_event.level:.2f} against prior bias",
                "early reversal — reduced strength by design",
            ]
            warnings = list(warnings) + ["CHOCH is an early reversal signal — reduced strength"]
        else:
            # BOS direction disagrees with the current bias (e.g. BOS_UP while
            # bias mixed) — treat as neutral, structure is transitioning.
            return self._neutral(
                context, features, quality, warnings, used,
                reasons=[
                    f"recent {kind.value} does not align with {bias.value} bias — transitioning"
                ],
                payload=base_payload,
            )

        if not recent:
            return self._neutral(
                context, features, quality, warnings, used,
                reasons=[
                    f"last structure break is {event_age} candles old (> {RECENT_EVENT_WINDOW}) — no fresh break"
                ],
                payload=base_payload,
            )

        recency = 1.0 if event_age <= 5 else 0.5
        strength += 0.20 * recency
        if h4 is not None and h4.structure_agrees_with(
            "up" if direction is AgentDirection.BUY else "down"
        ):
            strength += 0.20
            reasons.append("H4 structure agrees")
        if self._sequence_quality(features, direction):
            strength += 0.15
            reasons.append("clean swing sequence (≥2 confirming swings)")

        invalidation = self._invalidation_level(features, direction)
        if invalidation is not None:
            reasons.append(
                f"invalidation at last {'HL' if direction is AgentDirection.BUY else 'LH'}: {invalidation:.2f}"
            )

        payload = base_payload
        payload["invalidation_level"] = safe_float(invalidation, 2)
        return self.make_result(
            context,
            direction=direction,
            signal_strength=strength,
            reasons=reasons,
            features=payload,
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    # ------------------------------------------------------------------
    def _neutral(self, context, features, quality, warnings, used, *, reasons, payload):
        payload = dict(payload)
        payload["invalidation_level"] = None
        return self.make_result(
            context,
            direction=AgentDirection.NEUTRAL,
            signal_strength=0.0,
            reasons=reasons,
            features=payload,
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    def _sequence_quality(self, f, direction) -> bool:
        labels = [s.label.value for s in f.swings[-4:]]
        if direction is AgentDirection.BUY:
            return labels.count("HH") >= 1 and labels.count("HL") >= 1
        return labels.count("LH") >= 1 and labels.count("LL") >= 1

    def _invalidation_level(self, f, direction) -> float | None:
        if direction is AgentDirection.BUY:
            lows = f.swing_lows
            return lows[-1].price if lows else None
        highs = f.swing_highs
        return highs[-1].price if highs else None

    def _features(self, f, h4) -> dict:
        last_high = f.swing_highs[-1] if f.swing_highs else None
        last_low = f.swing_lows[-1] if f.swing_lows else None
        payload = {
            "bias": f.bias.value,
            "swing_count": len(f.swings),
            "last_event": f.last_event.to_dict() if f.last_event else None,
            "recent_events": [e.to_dict() for e in f.recent_events(5)],
            "last_swing_high": last_high.to_dict() if last_high else None,
            "last_swing_low": last_low.to_dict() if last_low else None,
            "confirmation_delay_candles": 2,
            "last_event_age": (f.length - 1 - f.last_event.index) if f.last_event else None,
        }
        if h4 is not None:
            payload["bias_h4"] = h4.bias.value
        return payload
