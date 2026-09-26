"""MomentumAgent — regime-aware momentum analysis (spec §11, Phase-2 §5).

Primary timeframe: **H1**; entry confirmation: **M15**.

Design rules:

* **No naive RSI reversals.**  ``RSI < 30 → BUY`` is only allowed when the
  regime is a confirmed RANGE (mean-reverting reading); in a trend regime the
  same RSI 30 is a *pullback* against momentum, not a buy signal.  RSI > 70 in
  an uptrend is noted as "extended", never automatically SELL.
* Direction comes from MACD confirmation plus corroboration; a single
  indicator never fires a signal.

Evidence tables (documented scores, not probabilities):

TREND mode (regime TREND_* or, without a regime, local ADX ≥ 20):

===  ============================================  =======
M    Evidence (H1 unless noted)                     Weight
===  ============================================  =======
M1   MACD line > 0 (BUY) / < 0 (SELL)               0.25
     — momentum *direction*; the histogram measures
     acceleration and is ~0 in a steady trend, so it
     cannot be the core evidence
M1b  MACD histogram > 0 (strengthening)             0.15
M2   MACD histogram rising vs previous bar           0.15
M3   RSI > 55 (BUY) / < 45 (SELL)                   0.20
M4   ROC(10) beyond ±0.1% (same sign)               0.15
M5   M15 MACD line agrees                           0.10
===  ============================================  =======

BUY ⇔ M1 ∧ at least two of {M1b, M2, M3, M4, M5}; SELL mirrored.

RANGE mode (regime RANGE/LOW_VOLATILITY or local ADX < 20):
* BUY when RSI < 30 **and** the last M15 candle is bullish (reversal
  confirmation); strength = 0.5·RSI depth + 0.5·confirmation.
* SELL mirrored at RSI > 70.

Guards: MACD/RSI conflict reduces strength by 40% with a warning (a histogram
decline is only counted when it exceeds 10% of |MACD line| — in a steady trend
the histogram idles near zero and is not evidence); trend exhaustion (RSI ≥ 75
with the MACD line still directional but a meaningful bearish histogram plus
fading ROC; mirrored down) returns NEUTRAL with an explicit warning — do not
chase an extended, decelerating move.
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.features import safe_float
from app.core.enums import (
    AgentDirection,
    DataQuality,
    MarketRegime,
    TimeFrame,
)

_RSI_BULL_TREND = 55.0
_RSI_BEAR_TREND = 45.0
_RSI_OVERSOLD = 30.0
_RSI_OVERBOUGHT = 70.0
_RSI_EXHAUSTION = 75.0
_ROC_THRESHOLD = 0.001


class MomentumAgent(BaseAgent):
    name = "momentum"
    description = "Regime-aware RSI/MACD/ROC momentum with exhaustion and conflict guards"
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
        if m15 is not None:
            if m15.last_candle_anomaly:
                warnings = list(warnings) + ["M15 decision candle abnormal — M15 evidence excluded"]
                m15 = None
            else:
                used.append(TimeFrame.M15)

        regime = context.regime.regime if context.regime is not None else None
        if regime in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN):
            mode = "trend"
        elif regime in (MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY):
            mode = "range"
        else:
            # no regime (or uncertain): fall back to a local ADX read
            mode = "trend" if (features.adx or 0) >= 20 else "range"

        if m15 is None:
            warnings = list(warnings) + ["M15 unavailable — entry confirmation skipped"]
            quality = DataQuality.DEGRADED if quality is DataQuality.OK else quality

        divergence = self._divergence(features)
        if divergence:
            warnings = list(warnings) + [divergence]

        if mode == "trend":
            return self._analyze_trend(context, features, m15, regime, warnings, quality, used)
        return self._analyze_range(context, features, m15, warnings, quality, used)

    # ------------------------------------------------------------------
    def _analyze_trend(self, context, f, m15, regime, warnings, quality, used) -> AgentResult:
        m1 = (f.macd_line or 0) > 0
        m1b = (f.macd_hist or 0) > 0
        m2 = (
            f.macd_hist is not None
            and f.macd_hist_prev is not None
            and f.macd_hist > f.macd_hist_prev
        )
        m3 = f.rsi is not None and f.rsi > _RSI_BULL_TREND
        m4 = f.roc is not None and f.roc > _ROC_THRESHOLD
        m5 = m15 is not None and (m15.macd_line or 0) > 0

        confirmations = sum([m1b, m2, m3, m4, m5])
        strength = 0.25 * m1 + 0.15 * m1b + 0.15 * m2 + 0.20 * m3 + 0.15 * m4 + 0.10 * m5
        buy = m1 and confirmations >= 2

        s1 = (f.macd_line or 0) < 0
        s1b = (f.macd_hist or 0) < 0
        s2 = (
            f.macd_hist is not None
            and f.macd_hist_prev is not None
            and f.macd_hist < f.macd_hist_prev
        )
        s3 = f.rsi is not None and f.rsi < _RSI_BEAR_TREND
        s4 = f.roc is not None and f.roc < -_ROC_THRESHOLD
        s5 = m15 is not None and (m15.macd_line or 0) < 0
        sell = s1 and sum([s1b, s2, s3, s4, s5]) >= 2
        sell_strength = 0.25 * s1 + 0.15 * s1b + 0.15 * s2 + 0.20 * s3 + 0.15 * s4 + 0.10 * s5

        # exhaustion guard: extreme RSI + fading histogram + fading ROC
        if regime is MarketRegime.TREND_UP or (regime is None and buy):
            if self._exhausted_up(f):
                return self.make_result(
                    context,
                    direction=AgentDirection.NEUTRAL,
                    signal_strength=0.0,
                    reasons=["bullish momentum exhausted (RSI extreme, histogram fading, ROC fading)"],
                    features=self._features(f, m15, mode="trend", exhausted=True),
                    warnings=warnings + ["momentum exhaustion after extended move"],
                    data_quality=quality,
                    timeframes_used=used,
                )
        if regime is MarketRegime.TREND_DOWN or (regime is None and sell):
            if self._exhausted_down(f):
                return self.make_result(
                    context,
                    direction=AgentDirection.NEUTRAL,
                    signal_strength=0.0,
                    reasons=["bearish momentum exhausted (RSI extreme, histogram fading, ROC fading)"],
                    features=self._features(f, m15, mode="trend", exhausted=True),
                    warnings=warnings + ["momentum exhaustion after extended decline"],
                    data_quality=quality,
                    timeframes_used=used,
                )

        conflict = False
        if buy and f.rsi is not None and f.rsi > _RSI_OVERBOUGHT and self._hist_bearish(f):
            conflict = True
        if sell and f.rsi is not None and f.rsi < _RSI_OVERSOLD and self._hist_bullish(f):
            conflict = True

        direction = AgentDirection.NEUTRAL
        strength_out = 0.0
        reasons: list[str] = ["no momentum confirmation (MACD line sign + ≥2 corroborators required)"]
        if buy and strength >= sell_strength:
            direction = AgentDirection.BUY
            strength_out = strength
            reasons = ["MACD line positive"]
            if m1b:
                reasons.append("MACD histogram positive (momentum strengthening)")
            if m2:
                reasons.append("MACD histogram rising")
            if m3:
                reasons.append(f"RSI {f.rsi:.1f} supports trend momentum")
            if m4:
                reasons.append(f"10-bar ROC {f.roc * 100:.2f}% positive")
            if m5:
                reasons.append("M15 momentum agrees")
        elif sell:
            direction = AgentDirection.SELL
            strength_out = sell_strength
            reasons = ["MACD line negative"]
            if s1b:
                reasons.append("MACD histogram negative (momentum strengthening)")
            if s2:
                reasons.append("MACD histogram falling")
            if s3:
                reasons.append(f"RSI {f.rsi:.1f} supports downside momentum")
            if s4:
                reasons.append(f"10-bar ROC {f.roc * 100:.2f}% negative")
            if s5:
                reasons.append("M15 momentum agrees")

        if direction is not AgentDirection.NEUTRAL and f.rsi is not None:
            if direction is AgentDirection.BUY and f.rsi > _RSI_OVERBOUGHT:
                warnings = list(warnings) + [
                    "RSI overbought in trend context — extended, not a sell signal by itself"
                ]
            if direction is AgentDirection.SELL and f.rsi < _RSI_OVERSOLD:
                warnings = list(warnings) + [
                    "RSI oversold in trend context — extended, not a buy signal by itself"
                ]
        if conflict:
            strength_out *= 0.6
            warnings = list(warnings) + ["RSI/MACD conflict — strength reduced"]

        return self.make_result(
            context,
            direction=direction,
            signal_strength=strength_out,
            reasons=reasons,
            features=self._features(f, m15, mode="trend", conflict=conflict),
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    # ------------------------------------------------------------------
    def _analyze_range(self, context, f, m15, warnings, quality, used) -> AgentResult:
        if f.rsi is None:
            return self.make_result(
                context,
                direction=AgentDirection.NEUTRAL,
                signal_strength=0.0,
                reasons=["RSI unavailable"],
                data_quality=DataQuality.DEGRADED,
                warnings=warnings,
                timeframes_used=used,
            )
        m15_bullish = m15 is not None and m15.last_close is not None and _last_candle_bullish(m15)
        m15_bearish = m15 is not None and not m15_bullish and _last_candle_bearish(m15)

        if f.rsi < _RSI_OVERSOLD and m15_bullish:
            depth = min(1.0, (_RSI_OVERSOLD - f.rsi) / _RSI_OVERSOLD)
            strength = 0.5 * depth + 0.5
            return self.make_result(
                context,
                direction=AgentDirection.BUY,
                signal_strength=strength,
                reasons=[
                    f"RSI {f.rsi:.1f} oversold in RANGE regime (mean-reverting reading)",
                    "M15 reversal candle confirms",
                ],
                features=self._features(f, m15, mode="range"),
                warnings=warnings,
                data_quality=quality,
                timeframes_used=used,
            )
        if f.rsi > _RSI_OVERBOUGHT and m15_bearish:
            depth = min(1.0, (f.rsi - _RSI_OVERBOUGHT) / (100 - _RSI_OVERBOUGHT))
            strength = 0.5 * depth + 0.5
            return self.make_result(
                context,
                direction=AgentDirection.SELL,
                signal_strength=strength,
                reasons=[
                    f"RSI {f.rsi:.1f} overbought in RANGE regime (mean-reverting reading)",
                    "M15 reversal candle confirms",
                ],
                features=self._features(f, m15, mode="range"),
                warnings=warnings,
                data_quality=quality,
                timeframes_used=used,
            )
        reasons = [f"no range-extreme momentum (RSI {f.rsi:.1f})"]
        if f.rsi < _RSI_OVERSOLD:
            reasons.append("oversold but M15 reversal confirmation missing")
        if f.rsi > _RSI_OVERBOUGHT:
            reasons.append("overbought but M15 reversal confirmation missing")
        return self.make_result(
            context,
            direction=AgentDirection.NEUTRAL,
            signal_strength=0.0,
            reasons=reasons,
            features=self._features(f, m15, mode="range"),
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    # ------------------------------------------------------------------
    # -- meaningful MACD-histogram declines ---------------------------------
    # In a STEADY trend the histogram hovers around zero (it measures
    # acceleration, not direction); only a decline large relative to the
    # MACD line counts as opposing evidence.
    _HIST_REL_THRESHOLD = 0.1

    def _hist_bearish(self, f) -> bool:
        if f.macd_hist is None or f.macd_line is None:
            return False
        return f.macd_hist < -self._HIST_REL_THRESHOLD * abs(f.macd_line)

    def _hist_bullish(self, f) -> bool:
        if f.macd_hist is None or f.macd_line is None:
            return False
        return f.macd_hist > self._HIST_REL_THRESHOLD * abs(f.macd_line)

    def _exhausted_up(self, f) -> bool:
        if f.rsi is None or f.macd_hist is None or f.roc is None:
            return False
        if f.rsi < _RSI_EXHAUSTION:
            return False
        # Losing momentum = price below the signal line (meaningfully bearish
        # histogram) while the line is still bullish, or a meaningful drop.
        fading_hist = self._hist_bearish(f) or (
            f.macd_hist_prev is not None
            and f.macd_hist
            < f.macd_hist_prev - self._HIST_REL_THRESHOLD * abs(f.macd_line or 0.0)
        )
        roc_window = f.indicators.get("close")
        fading_roc = abs(f.roc) < 0.5 * _max_abs_roc(f)
        return fading_hist and fading_roc and roc_window is not None

    def _exhausted_down(self, f) -> bool:
        if f.rsi is None or f.macd_hist is None or f.roc is None:
            return False
        if f.rsi > (100 - _RSI_EXHAUSTION):
            return False
        fading_hist = self._hist_bullish(f) or (
            f.macd_hist_prev is not None
            and f.macd_hist
            > f.macd_hist_prev + self._HIST_REL_THRESHOLD * abs(f.macd_line or 0.0)
        )
        fading_roc = abs(f.roc) < 0.5 * _max_abs_roc(f)
        return fading_hist and fading_roc

    def _divergence(self, f) -> str | None:
        """Bearish/bullish RSI divergence at the last two confirmed swing highs/lows."""
        rsi_series = f.indicators.get("rsi")
        close_series = f.indicators.get("close")
        if rsi_series is None or close_series is None or len(f.swing_highs) >= 2:
            highs = f.swing_highs[-2:]
            r1, r2 = rsi_series.iloc[highs[0].index], rsi_series.iloc[highs[1].index]
            p1, p2 = close_series.iloc[highs[0].index], close_series.iloc[highs[1].index]
            if p2 > p1 and r2 < r1:
                return "bearish RSI divergence at swing highs"
        if rsi_series is not None and close_series is not None and len(f.swing_lows) >= 2:
            lows = f.swing_lows[-2:]
            r1, r2 = rsi_series.iloc[lows[0].index], rsi_series.iloc[lows[1].index]
            p1, p2 = close_series.iloc[lows[0].index], close_series.iloc[lows[1].index]
            if p2 < p1 and r2 > r1:
                return "bullish RSI divergence at swing lows"
        return None

    def _features(self, f, m15, *, mode: str, conflict: bool = False, exhausted: bool = False) -> dict:
        payload: dict = {
            "mode": mode,
            "rsi_h1": safe_float(f.rsi, 2),
            "macd_line_h1": safe_float(f.macd_line, 4),
            "macd_hist_h1": safe_float(f.macd_hist, 4),
            "macd_hist_rising": bool(
                f.macd_hist is not None
                and f.macd_hist_prev is not None
                and f.macd_hist > f.macd_hist_prev
            ),
            "roc10": safe_float(f.roc, 5),
            "impulse_atr": safe_float(f.last_candle_body_atr, 3),
            "structure_bias_h1": f.bias.value,
            "conflict": conflict,
            "exhausted": exhausted,
        }
        if m15 is not None:
            payload["macd_line_m15"] = safe_float(m15.macd_line, 4)
            payload["macd_hist_m15"] = safe_float(m15.macd_hist, 4)
            payload["rsi_m15"] = safe_float(m15.rsi, 2)
        return payload


def _last_candle_bullish(m15) -> bool:
    row = m15.df.iloc[-1]
    return float(row["close"]) > float(row["open"])


def _last_candle_bearish(m15) -> bool:
    row = m15.df.iloc[-1]
    return float(row["close"]) < float(row["open"])


def _max_abs_roc(f) -> float:
    """Max |ROC(10)| over the last ~20 readable values (fading measure)."""
    close = f.indicators.get("close")
    if close is None or len(close) < 30:
        return 0.0
    rocs = close.iloc[-30:].pct_change(10).dropna().abs()
    return float(rocs.max()) if not rocs.empty else 0.0
