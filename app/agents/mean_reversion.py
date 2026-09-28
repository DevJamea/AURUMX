"""MeanReversionAgent — heavily constrained range mean-reversion (spec §15, Phase-2 §9).

Primary timeframe: **H1** (deviation statistics); confirmation: **M15**.

This agent is a deliberate redesign of SnipBot's DCA/grid ideas — it contains
**no martingale, no averaging down, no pyramiding, no DCA and no
loss-dependent sizing**.  It has no notion of positions at all: it evaluates a
single setup at a single point in time.

Gates (in order — failing any gate returns NEUTRAL):

1. Data: H1 ≥ 60 closed candles and M15 available, else INSUFFICIENT.
2. **Regime gate**: only RANGE or LOW_VOLATILITY regimes may produce a signal.
   TREND_UP / TREND_DOWN / HIGH_VOLATILITY / UNCERTAIN ⇒ NEUTRAL with the
   reason recorded.  (When no regime is attached the agent applies an
   equivalent local check: ADX < 20 and ATR-normalized EMA gap < 0.5.)

Long setup (short mirrored) — all three required:

* **L1 statistical edge**: %B ≤ 0.05 (at/below lower Bollinger band) or
  deviation from SMA20 ≤ −1.5 ATR;
* **L2 range boundary**: within 0.75 ATR of the 50-bar low (range low);
* **L3 reversal confirmation** on M15: last closed candle bullish with a
  meaningful lower wick (≥ 0.33 of range), or H1 close back inside the band.

L1 without L3 ⇒ NEUTRAL with an explicit warning ("edge reached, no
confirmation") — the agent never catches the falling knife.

Strength = 0.45·edge-depth + 0.30·boundary-proximity + 0.25·confirmation-quality
(deterministic components, documented; not a probability).
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.features import safe_float
from app.core.enums import AgentDirection, DataQuality, MarketRegime, TimeFrame

_EDGE_PERCENT_B = 0.05
_EDGE_DEVIATION_ATR = -1.5
_BOUNDARY_ATR = 0.75
_WICK_RATIO = 0.33


class MeanReversionAgent(BaseAgent):
    name = "mean_reversion"
    description = "Range-only statistical mean reversion (no martingale, no averaging, no DCA)"
    primary_timeframe = TimeFrame.H1
    min_candles = 60

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
        m15 = context.features(TimeFrame.M15)
        if m15 is None:
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=["M15 unavailable — reversal confirmation impossible"],
                data_quality=DataQuality.DEGRADED,
                warnings=warnings,
                timeframes_used=used,
            )
        if m15.last_candle_anomaly:
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=["M15 decision candle abnormal (possible bad tick) — no confirmation"],
                data_quality=DataQuality.DEGRADED,
                warnings=warnings + ["abnormal M15 candle — possible bad tick"],
                timeframes_used=used,
            )
        used.append(TimeFrame.M15)

        # ---- regime gate --------------------------------------------------
        regime = context.regime.regime if context.regime is not None else None
        if regime is not None:
            if regime not in (MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY):
                return self.make_result(
                    context,
                    direction=AgentDirection.NEUTRAL,
                    signal_strength=0.0,
                    reasons=[f"regime gate: {regime.value} excludes mean reversion"],
                    features={"regime": regime.value, "gated": True},
                    warnings=warnings,
                    data_quality=quality,
                    timeframes_used=used,
                )
            regime_label = regime.value
        else:
            # equivalent local range confirmation (ADX + flat EMAs)
            flat_emas = (
                features.atr
                and features.ema20 is not None
                and features.ema50 is not None
                and abs(features.ema20 - features.ema50) / features.atr < 0.5
            )
            if not ((features.adx or 0) < 20 and flat_emas):
                return self.make_result(
                    context,
                    direction=AgentDirection.NEUTRAL,
                    signal_strength=0.0,
                    reasons=["range not confirmed locally (ADX ≥ 20 or EMAs not flat)"],
                    features={"regime": "LOCAL_CHECK", "gated": True},
                    warnings=warnings,
                    data_quality=quality,
                    timeframes_used=used,
                )
            regime_label = "RANGE (local)"

        if features.percent_b is None or features.atr in (None, 0):
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=["Bollinger/ATR statistics unavailable"],
                data_quality=DataQuality.DEGRADED,
                warnings=warnings,
                timeframes_used=used,
            )

        deviation = features.deviation_atr()
        range_low, range_high = features.rolling_extremes(window=50)
        close = features.last_close

        long_edge = features.percent_b <= _EDGE_PERCENT_B or (
            deviation is not None and deviation <= _EDGE_DEVIATION_ATR
        )
        short_edge = features.percent_b >= (1 - _EDGE_PERCENT_B) or (
            deviation is not None and deviation >= -_EDGE_DEVIATION_ATR
        )

        if not (long_edge or short_edge):
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=[f"no statistical edge (percent_b={features.percent_b:.2f})"],
                features=self._features(features, regime_label, None),
                warnings=warnings,
                data_quality=quality,
                timeframes_used=used,
            )

        direction = AgentDirection.BUY if long_edge else AgentDirection.SELL

        # ---- boundary proximity (L2) ---------------------------------------
        boundary_ok = False
        proximity = 0.0
        if direction is AgentDirection.BUY and range_low is not None and close is not None:
            distance = (close - range_low) / features.atr
            boundary_ok = distance <= _BOUNDARY_ATR
            proximity = max(0.0, min(1.0, 1 - distance / _BOUNDARY_ATR))
        elif direction is AgentDirection.SELL and range_high is not None and close is not None:
            distance = (range_high - close) / features.atr
            boundary_ok = distance <= _BOUNDARY_ATR
            proximity = max(0.0, min(1.0, 1 - distance / _BOUNDARY_ATR))

        if not boundary_ok:
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=["statistical edge reached but price not at range boundary"],
                features=self._features(features, regime_label, direction.value),
                warnings=warnings,
                data_quality=quality,
                timeframes_used=used,
            )

        # ---- reversal confirmation (L3) -------------------------------------
        m15_row = m15.df.iloc[-1]
        m15_open, m15_close = float(m15_row["open"]), float(m15_row["close"])
        m15_high, m15_low = float(m15_row["high"]), float(m15_row["low"])
        m15_range = m15_high - m15_low
        m15_bullish = m15_close > m15_open
        m15_bearish = m15_close < m15_open
        lower_wick = (min(m15_open, m15_close) - m15_low) / m15_range if m15_range > 0 else 0.0
        upper_wick = (m15_high - max(m15_open, m15_close)) / m15_range if m15_range > 0 else 0.0

        back_inside = (
            features.bb_lower is not None
            and close is not None
            and (
                (direction is AgentDirection.BUY and close > features.bb_lower)
                or (direction is AgentDirection.SELL and close < features.bb_upper)
            )
        )
        if direction is AgentDirection.BUY:
            confirmed = m15_bullish and (lower_wick >= _WICK_RATIO or back_inside)
            full_quality = m15_bullish and lower_wick >= _WICK_RATIO and back_inside
            confirmation_quality = 1.0 if full_quality else (0.6 if confirmed else 0.0)
        else:
            confirmed = m15_bearish and (upper_wick >= _WICK_RATIO or back_inside)
            full_quality = m15_bearish and upper_wick >= _WICK_RATIO and back_inside
            confirmation_quality = 1.0 if full_quality else (0.6 if confirmed else 0.0)

        if not confirmed:
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=["edge at range boundary but no M15 reversal confirmation"],
                features=self._features(features, regime_label, direction.value),
                warnings=warnings + ["edge reached without confirmation — no falling-knife catches"],
                data_quality=quality,
                timeframes_used=used,
            )

        # ---- strength --------------------------------------------------------
        if direction is AgentDirection.BUY:
            depth = (
                max(0.4, min(1.0, -deviation / 2.5))
                if deviation is not None and deviation < 0
                else 0.4
            )
            reasons = [
                f"price at lower statistical edge (percent_b={features.percent_b:.2f})",
                f"within {_BOUNDARY_ATR} ATR of the 50-bar range low",
                "M15 bullish reversal candle confirmed",
            ]
        else:
            depth = (
                max(0.4, min(1.0, deviation / 2.5))
                if deviation is not None and deviation > 0
                else 0.4
            )
            reasons = [
                f"price at upper statistical edge (percent_b={features.percent_b:.2f})",
                f"within {_BOUNDARY_ATR} ATR of the 50-bar range high",
                "M15 bearish reversal candle confirmed",
            ]
        strength = 0.45 * depth + 0.30 * proximity + 0.25 * confirmation_quality

        return self.make_result(
            context,
            direction=direction,
            signal_strength=strength,
            reasons=reasons,
            features=self._features(
                features,
                regime_label,
                direction.value,
                components={
                    "edge_depth": round(depth, 3),
                    "boundary_proximity": round(proximity, 3),
                    "confirmation": round(confirmation_quality, 3),
                },
            ),
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    # ------------------------------------------------------------------
    def _features(self, f, regime_label, direction, components=None) -> dict:
        payload = {
            "regime": regime_label,
            "direction": direction,
            "percent_b": safe_float(f.percent_b, 3),
            "deviation_atr": safe_float(f.deviation_atr(), 3),
            "atr": safe_float(f.atr, 2),
            "adx": safe_float(f.adx, 2),
            "no_martingale": True,
            "no_averaging": True,
        }
        if components:
            payload.update(components)
        return payload
