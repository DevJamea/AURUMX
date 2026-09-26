"""LiquidityAgent — price-action / liquidity analysis (spec §13, Phase-2 §7).

Primary timeframe: **M15** (entry timing); reference levels from **H1**
confirmed swings plus previous-day highs/lows.

Honest naming: this agent uses **candle wicks, tick volume and swing/level
interaction only**.  It does NOT claim order-book / market-depth insight —
real MT5 depth data is not implemented (spec §13).

Pattern table (bullish points; bearish mirrored as negatives).  ``score`` is
the net sum; BUY at ≥ +0.30, SELL at ≤ −0.30, strength = min(1, |score|).
"Previous" levels exclude the newest candle(s) so a pattern candle never
becomes its own reference:

===========================  ==============================  ==============
Pattern                      Definition                      Points
===========================  ==============================  ==============
Liquidity sweep (bullish)    last candle's low pierces a      +0.35
                             reference low, close back above
Liquidity sweep (bearish)    last candle's high pierces a     −0.35
                             reference high, close back below
Wick rejection (bullish)     lower wick ≥ 60% of range and    +0.30
                             close in top 40% of range
Wick rejection (bearish)     upper wick ≥ 60%, close in       −0.30
                             bottom 40%
Breakout (up)                close > level + 0.10 ATR         +0.30 (volume
                             confirmed, ratio ≥ 1.5) else
                             +0.20
Breakout (down)              close < level − 0.10 ATR         mirrored
Failed breakout (up)         *previous* candle closed above   −0.25
                             the level, last candle closes
                             back below it
Failed breakout (down)       previous candle closed below      +0.25
                             the level, last candle closes
                             back above it
===========================  ==============================  ==============

Volume handling (degrade safely): tick-volume ratio (last / 20-bar mean)
upgrades breakout evidence when ≥ 1.5.  If volume data is missing or all-zero
the agent skips volume confirmation and emits a warning — price evidence
still works.  Abnormal volume alone adds no direction.

Reference levels: last confirmed H1 swing high/low, previous UTC-day high/low
(server-session boundaries are not knowable from candle stamps — documented
limitation), and the M15 20-bar high/low.
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.features import safe_float
from app.core.enums import AgentDirection, DataQuality, TimeFrame

_BREAKOUT_BUFFER_ATR = 0.10
_ACTION_THRESHOLD = 0.30


class LiquidityAgent(BaseAgent):
    name = "liquidity"
    description = "Wick/sweep/breakout price-action around reference levels (NOT order-book)"
    primary_timeframe = TimeFrame.M15
    min_candles = 45

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

        used = [TimeFrame.M15]
        h1 = context.features(TimeFrame.H1)
        if h1 is not None:
            used.append(TimeFrame.H1)

        row = features.df.iloc[-1]
        high, low = float(row["high"]), float(row["low"])
        open_, close = float(row["open"]), float(row["close"])
        atr = features.atr or 0.0
        candle_range = high - low

        if features.volume_ratio is None:
            warnings = list(warnings) + [
                "tick volume unavailable/zero — volume confirmation skipped"
            ]
            if quality is DataQuality.OK:
                quality = DataQuality.DEGRADED

        levels = self._reference_levels(features, h1, exclude_last=1)
        pre_levels = self._reference_levels(features, h1, exclude_last=2)
        prev_close = (
            float(features.df.iloc[-2]["close"]) if features.length >= 2 else None
        )
        score = 0.0
        reasons: list[str] = []
        vol_confirmed = features.volume_ratio is not None and features.volume_ratio >= 1.5

        # --- sweeps & breakouts (single-candle vs reference levels) --------
        for label, price in levels:
            if price is None:
                continue
            if label in ("swing_low_h1", "prev_day_low", "recent_low_m15"):
                if low < price and close > price:
                    score += 0.35
                    reasons.append(f"bullish sweep of {label} {price:.2f} — close back above")
                elif atr > 0 and close < price - _BREAKOUT_BUFFER_ATR * atr:
                    score += -0.30 if vol_confirmed else -0.20
                    reasons.append(f"downside breakout of {label} {price:.2f}")
            if label in ("swing_high_h1", "prev_day_high", "recent_high_m15"):
                if high > price and close < price:
                    score -= 0.35
                    reasons.append(f"bearish sweep of {label} {price:.2f} — close back below")
                elif atr > 0 and close > price + _BREAKOUT_BUFFER_ATR * atr:
                    score += 0.30 if vol_confirmed else 0.20
                    reasons.append(
                        f"upside breakout of {label} {price:.2f}"
                        + (" (volume confirmed)" if vol_confirmed else "")
                    )

        # --- failed breakouts (two-candle pattern vs pre-breakout levels) ---
        if prev_close is not None and atr > 0:
            for label, price in pre_levels:
                if price is None:
                    continue
                if label in ("swing_high_h1", "prev_day_high", "recent_high_m15"):
                    if prev_close > price + _BREAKOUT_BUFFER_ATR * atr and close < price:
                        score -= 0.25
                        reasons.append(
                            f"failed breakout above {label} {price:.2f} — close back below"
                        )
                if label in ("swing_low_h1", "prev_day_low", "recent_low_m15"):
                    if prev_close < price - _BREAKOUT_BUFFER_ATR * atr and close > price:
                        score += 0.25
                        reasons.append(
                            f"failed breakdown below {label} {price:.2f} — close back above"
                        )

        # --- wick rejection (no level required) ---------------------------
        if candle_range > 0 and atr > 0:
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            if lower_wick / candle_range >= 0.60 and (close - low) / candle_range >= 0.60:
                score += 0.30
                reasons.append("bullish wick rejection (long lower wick, close near high)")
            elif upper_wick / candle_range >= 0.60 and (high - close) / candle_range >= 0.60:
                score -= 0.30
                reasons.append("bearish wick rejection (long upper wick, close near low)")

        direction = AgentDirection.NEUTRAL
        if score >= _ACTION_THRESHOLD:
            direction = AgentDirection.BUY
        elif score <= -_ACTION_THRESHOLD:
            direction = AgentDirection.SELL

        if not reasons:
            reasons = ["no liquidity-relevant price action at reference levels"]

        payload = {
            "score": safe_float(score, 3),
            "reference_levels": [
                {"level": label, "price": safe_float(price, 2)} for label, price in levels
            ],
            "volume_ratio": safe_float(features.volume_ratio, 2),
            "volume_confirmed": vol_confirmed,
            "last_candle_lower_wick_ratio": safe_float(features.last_candle_lower_wick_ratio, 3),
            "last_candle_upper_wick_ratio": safe_float(features.last_candle_upper_wick_ratio, 3),
            "uses_order_book": False,  # explicit honesty marker (spec §13)
        }
        return self.make_result(
            context,
            direction=direction,
            signal_strength=min(1.0, abs(score)),
            reasons=reasons,
            features=payload,
            warnings=warnings,
            data_quality=quality,
            timeframes_used=used,
        )

    # ------------------------------------------------------------------
    def _reference_levels(
        self, m15_features, h1_features, *, exclude_last: int
    ) -> list[tuple[str, float | None]]:
        """Reference levels; ``exclude_last`` drops the newest N M15 candles so
        pattern candles are never their own level."""
        levels: list[tuple[str, float | None]] = []
        if h1_features is not None:
            if h1_features.swing_highs:
                levels.append(("swing_high_h1", h1_features.swing_highs[-1].price))
            if h1_features.swing_lows:
                levels.append(("swing_low_h1", h1_features.swing_lows[-1].price))
        day = m15_features.day_stats
        levels.append(("prev_day_high", day.prev_day_high))
        levels.append(("prev_day_low", day.prev_day_low))
        recent_low, recent_high = m15_features.rolling_extremes(
            window=20, exclude_last=exclude_last
        )
        levels.append(("recent_high_m15", recent_high))
        levels.append(("recent_low_m15", recent_low))
        return levels
