"""TrendAgent — multi-timeframe trend analysis (spec §10, Phase-2 §4).

Primary timeframe: **H1** (directional context).  Macro context: **H4**.

Evidence table (bullish; bearish mirrored).  ``signal_strength`` is the
weighted sum of satisfied evidence — a documented deterministic score, not a
probability:

===  ===============================  ========  =============================
E    Evidence (H1 unless noted)        Weight    Notes
===  ===============================  ========  =============================
E1   EMA20 > EMA50 by > 0.05 ATR      0.25      ATR-buffered: epsilon-thin
                                                 "alignment" in noise doesn't
                                                 count
E2   EMA50 > EMA200 by > 0.05 ATR     0.15      requires EMA200 warm-up
                                                 (200 candles); missing ⇒
                                                 evidence excluded, quality
                                                 DEGRADED
E3   ADX ≥ 25 (1.0) / ≥ 20 (0.6)      0.20      directional strength
E4   regression slope > 0.10 ATR/bar  0.15      last 20 closes
E5   H4 macro agrees (EMA20>EMA50,    0.10      macro timeframe
     ATR-buffered)
E6   H1 structure bias UPTREND        0.15      HH/HL sequence
===  ===============================  ========  =============================

Actionable BUY requires **E1 plus at least two** of {E2, E3>0, E4, E5, E6} —
``EMA20 > EMA50`` alone never produces a signal.

States: ``STRONG_BULLISH`` (E1∧E2∧ADX≥25∧E4), ``WEAK_BULLISH`` (actionable
otherwise), mirrored bearish, else ``NEUTRAL``.
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.features import safe_float
from app.core.enums import AgentDirection, DataQuality, TimeFrame

#: ATR-buffer for EMA comparisons (same constant as the feature layer).
_BUFFER = 0.05


class TrendAgent(BaseAgent):
    name = "trend"
    description = "Multi-timeframe EMA/ADX/slope trend classification"
    primary_timeframe = TimeFrame.H1
    min_candles = 60  # EMA50 + ADX warm-up on H1

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

        if features.ema200 is None:
            # EMA200 warm-up: E2 (long-trend evidence) cannot be evaluated.
            quality = DataQuality.DEGRADED
            warnings = list(warnings) + [
                "EMA200 not warmed up (<200 candles) — long-trend evidence excluded"
            ]

        bull = self._bull_evidence(features, h4)
        bear = self._bear_evidence(features, h4)

        if bull["strength"] >= bear["strength"] and bull["actionable"]:
            direction, evidence = AgentDirection.BUY, bull
        elif bear["actionable"]:
            direction, evidence = AgentDirection.SELL, bear
        else:
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=self._neutral_reasons(features, h4, bull, bear),
                features=self._features(features, h4, None),
                warnings=warnings,
                data_quality=quality,
                timeframes_used=used,
            )

        state = evidence["state"]
        return self.make_result(
            context,
            direction=direction,
            signal_strength=evidence["strength"],
            reasons=evidence["reasons"],
            features=self._features(features, h4, state),
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    # ------------------------------------------------------------------
    def _ema_gap_atr(self, f, fast: float | None, slow: float | None) -> float | None:
        if fast is None or slow is None or not f.atr:
            return None
        return (fast - slow) / f.atr

    def _bull_evidence(self, f, h4) -> dict:
        e1 = (gap := self._ema_gap_atr(f, f.ema20, f.ema50)) is not None and gap > _BUFFER
        e2 = (
            (gap2 := self._ema_gap_atr(f, f.ema50, f.ema200)) is not None and gap2 > _BUFFER
        )
        adx_score = 0.0
        if f.adx is not None:
            if f.adx >= 25:
                adx_score = 1.0
            elif f.adx >= 20:
                adx_score = 0.6
        e3 = adx_score > 0
        e4 = f.slope_atr is not None and f.slope_atr > 0.10
        e5 = False
        if h4 is not None:
            gap_h4 = self._ema_gap_atr(h4, h4.ema20, h4.ema50)
            e5 = gap_h4 is not None and gap_h4 > _BUFFER
        e6 = f.structure_agrees_with("up")

        confirmations = sum([e2, e3, e4, e5, e6])
        strength = (
            0.25 * e1
            + 0.15 * e2
            + 0.20 * adx_score
            + 0.15 * e4
            + 0.10 * e5
            + 0.15 * e6
        )
        actionable = e1 and confirmations >= 2
        strong = e1 and e2 and (f.adx or 0) >= 25 and e4
        reasons = []
        if e1:
            reasons.append("EMA20 above EMA50 (ATR-buffered)")
        if e2:
            reasons.append("EMA50 above EMA200")
        if e3:
            reasons.append(f"ADX {f.adx:.1f} indicates a directional trend")
        if e4:
            reasons.append("positive price slope over last 20 closes")
        if e5:
            reasons.append("H4 macro trend agrees (bullish)")
        if e6:
            reasons.append("H1 structure shows HH/HL sequence")
        return {
            "actionable": actionable,
            "strength": strength if actionable else 0.0,
            "state": "STRONG_BULLISH" if strong else "WEAK_BULLISH",
            "reasons": reasons,
            "count": confirmations,
        }

    def _bear_evidence(self, f, h4) -> dict:
        e1 = (gap := self._ema_gap_atr(f, f.ema20, f.ema50)) is not None and gap < -_BUFFER
        e2 = (
            (gap2 := self._ema_gap_atr(f, f.ema50, f.ema200)) is not None and gap2 < -_BUFFER
        )
        adx_score = 0.0
        if f.adx is not None:
            if f.adx >= 25:
                adx_score = 1.0
            elif f.adx >= 20:
                adx_score = 0.6
        e3 = adx_score > 0
        e4 = f.slope_atr is not None and f.slope_atr < -0.10
        e5 = False
        if h4 is not None:
            gap_h4 = self._ema_gap_atr(h4, h4.ema20, h4.ema50)
            e5 = gap_h4 is not None and gap_h4 < -_BUFFER
        e6 = f.structure_agrees_with("down")

        confirmations = sum([e2, e3, e4, e5, e6])
        strength = (
            0.25 * e1
            + 0.15 * e2
            + 0.20 * adx_score
            + 0.15 * e4
            + 0.10 * e5
            + 0.15 * e6
        )
        actionable = e1 and confirmations >= 2
        strong = e1 and e2 and (f.adx or 0) >= 25 and e4
        reasons = []
        if e1:
            reasons.append("EMA20 below EMA50 (ATR-buffered)")
        if e2:
            reasons.append("EMA50 below EMA200")
        if e3:
            reasons.append(f"ADX {f.adx:.1f} indicates a directional trend")
        if e4:
            reasons.append("negative price slope over last 20 closes")
        if e5:
            reasons.append("H4 macro trend agrees (bearish)")
        if e6:
            reasons.append("H1 structure shows LH/LL sequence")
        return {
            "actionable": actionable,
            "strength": strength if actionable else 0.0,
            "state": "STRONG_BEARISH" if strong else "WEAK_BEARISH",
            "reasons": reasons,
            "count": confirmations,
        }

    # ------------------------------------------------------------------
    def _neutral_reasons(self, f, h4, bull, bear) -> list[str]:
        reasons = ["no actionable trend evidence"]
        if f.ema_alignment == "flat":
            reasons.append("EMA20/EMA50 flat (within ATR buffer) — sideways")
        elif f.ema_alignment == "mixed":
            reasons.append("EMA alignment mixed (fast vs slow disagree)")
        if f.adx is not None and f.adx < 20:
            reasons.append(f"ADX {f.adx:.1f} below 20 — no directional strength")
        if bull["count"] < 2 and bear["count"] < 2 and (bull["actionable"] or bear["actionable"]):
            reasons.append("alignment present but confirmations < 2 — insufficient evidence")
        return reasons

    def _features(self, f, h4, state) -> dict:
        payload: dict = {
            "trend_state": state or "NEUTRAL",
            "ema_alignment_h1": f.ema_alignment,
            "adx": safe_float(f.adx, 2),
            "slope_atr": safe_float(f.slope_atr, 3),
            "structure_bias_h1": f.bias.value,
        }
        if h4 is not None:
            payload["ema_alignment_h4"] = h4.ema_alignment
            payload["adx_h4"] = safe_float(h4.adx, 2)
        return payload
