"""Shared deterministic feature/indicator layer (spec §20).

Every indicator, swing and structure computation used by the agents lives here
as a **pure function** of a ``CandleSeries`` — no globals, no caches, no wall
clock, no network.  Agents never re-derive indicators; they read a
``TimeframeFeatures`` bundle computed once per timeframe when the
``MarketContext`` is built (instance-scoped, recreated for every snapshot, so
backtesting replays are exactly reproducible).

Structure rules (documented because they matter for look-ahead safety):

* Swings are confirmed fractals: a swing high at bar ``i`` needs ``left`` bars
  before and ``right`` after with lower highs (mirrored for lows).  A swing
  therefore only *exists* once bar ``i + right`` has closed — the
  **confirmation delay is 2 closed candles** with the default ``left=right=2``.
* Structure events (BOS/CHOCH) are emitted when a *closed* candle crosses a
  level that was already confirmed at that time.  Nothing in this module ever
  looks ahead of the last candle it was given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.core.enums import (
    StructureBias,
    StructureEventType,
    SwingKind,
    SwingLabel,
    TimeFrame,
    VolatilityLevel,
    VolatilityState,
)
from app.core.models import CandleSeries

#: Default fractal swing parameters (confirmation delay = right = 2 candles).
SWING_LEFT = 2
SWING_RIGHT = 2

#: Minimum swing prominence in ATR units.  A fractal whose excursion above
#: the opposite extreme of its window is smaller than this is treated as noise
#: (wick jitter) and not recorded as a swing.
SWING_MIN_PROMINENCE_ATR = 0.75

#: Normalization buffer for EMA comparisons, in ATR units.  Two EMAs closer
#: than this are considered *flat*, which prevents epsilon-thin "trends" in
#: sideways noise from being read as alignment.
EMA_FLAT_ATR = 0.05


# ==========================================================================
# Swings & structure
# ==========================================================================
@dataclass(frozen=True)
class SwingPoint:
    """A confirmed swing.  ``confirmation_index`` is the bar at which the
    swing became knowable (no look-ahead before it)."""

    index: int
    time: datetime
    price: float
    kind: SwingKind
    confirmation_index: int
    label: SwingLabel = SwingLabel.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time": self.time.isoformat(),
            "price": round(self.price, 6),
            "kind": self.kind.value,
            "label": self.label.value,
            "confirmation_index": self.confirmation_index,
        }


@dataclass(frozen=True)
class StructureEvent:
    kind: StructureEventType
    index: int
    time: datetime
    level: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "index": self.index,
            "time": self.time.isoformat(),
            "level": round(self.level, 6),
        }


def detect_swings(
    df: pd.DataFrame, *, left: int = SWING_LEFT, right: int = SWING_RIGHT
) -> list[SwingPoint]:
    """Confirmed fractal swings, ordered by bar index.

    A swing high at ``i`` requires ``high[i]`` to be the window maximum and
    strictly higher than both immediate neighbours (tie policy: plateaus do
    not produce swings).  The swing is only reported with
    ``confirmation_index = i + right`` — callers analyzing a prefix of length
    ``n`` only ever see swings with ``confirmation_index <= n - 1``.
    """
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = list(df["time"])
    swings: list[SwingPoint] = []
    if len(df) < left + right + 1:
        return swings
    for i in range(left, len(df) - right):
        window_h = highs[i - left : i + right + 1]
        if highs[i] >= window_h.max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swings.append(
                SwingPoint(
                    index=i,
                    time=times[i],
                    price=float(highs[i]),
                    kind=SwingKind.HIGH,
                    confirmation_index=i + right,
                )
            )
        window_l = lows[i - left : i + right + 1]
        if lows[i] <= window_l.min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swings.append(
                SwingPoint(
                    index=i,
                    time=times[i],
                    price=float(lows[i]),
                    kind=SwingKind.LOW,
                    confirmation_index=i + right,
                )
            )
    swings.sort(key=lambda s: (s.index, 0 if s.kind is SwingKind.HIGH else 1))
    return swings


def label_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Label each swing HH/HL/LH/LL against its predecessor of the same kind."""
    labeled: list[SwingPoint] = []
    last_high: float | None = None
    last_low: float | None = None
    for swing in swings:
        label = SwingLabel.NONE
        if swing.kind is SwingKind.HIGH:
            if last_high is not None:
                label = SwingLabel.HH if swing.price > last_high else SwingLabel.LH
            last_high = swing.price
        else:
            if last_low is not None:
                label = SwingLabel.HL if swing.price > last_low else SwingLabel.LL
            last_low = swing.price
        labeled.append(replace(swing, label=label))
    return labeled


def structure_bias(swings: list[SwingPoint]) -> StructureBias:
    """Bias from the last labelled swing of each kind (HH+HL up, LH+LL down)."""
    highs = [s for s in swings if s.kind is SwingKind.HIGH and s.label is not SwingLabel.NONE]
    lows = [s for s in swings if s.kind is SwingKind.LOW and s.label is not SwingLabel.NONE]
    if not highs or not lows:
        return StructureBias.UNKNOWN
    hh = highs[-1].label is SwingLabel.HH
    hl = lows[-1].label is SwingLabel.HL
    lh = highs[-1].label is SwingLabel.LH
    ll = lows[-1].label is SwingLabel.LL
    if hh and hl:
        return StructureBias.UPTREND
    if lh and ll:
        return StructureBias.DOWNTREND
    return StructureBias.RANGE


def structure_events(
    df: pd.DataFrame, swings: list[SwingPoint]
) -> list[StructureEvent]:
    """BOS / CHOCH events from closed-candle breaks of *confirmed* levels.

    A level becomes active only at its ``confirmation_index``; a break is
    recorded when a closed candle's close crosses it.  A break in the direction
    of the prevailing bias is a BOS (continuation); a break against it is a
    CHOCH (change of character).  The very first break is labelled BOS
    (documented convention: it establishes the initial bias).
    """
    closes = df["close"].to_numpy(dtype=float)
    times = list(df["time"])
    by_confirmation: dict[int, list[SwingPoint]] = {}
    for swing in swings:
        by_confirmation.setdefault(swing.confirmation_index, []).append(swing)

    events: list[StructureEvent] = []
    active_high: SwingPoint | None = None
    active_low: SwingPoint | None = None
    bias: str | None = None

    for i in range(len(df)):
        for swing in by_confirmation.get(i, ()):
            if swing.kind is SwingKind.HIGH:
                active_high = swing
            else:
                active_low = swing
        if active_high is not None and closes[i] > active_high.price:
            kind = (
                StructureEventType.BOS_UP
                if bias in (None, "up")
                else StructureEventType.CHOCH_UP
            )
            events.append(
                StructureEvent(kind=kind, index=i, time=times[i], level=active_high.price)
            )
            bias = "up"
            active_high = None
        if active_low is not None and closes[i] < active_low.price:
            kind = (
                StructureEventType.BOS_DOWN
                if bias in (None, "down")
                else StructureEventType.CHOCH_DOWN
            )
            events.append(
                StructureEvent(kind=kind, index=i, time=times[i], level=active_low.price)
            )
            bias = "down"
            active_low = None
    return events


# ==========================================================================
# Indicators (pure pandas implementations, Wilder smoothing where standard)
# ==========================================================================
def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI.  Flat data (no gains, no losses) is defined as 50."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ag = avg_gain.to_numpy(dtype=float)
    al = avg_loss.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(
            np.isnan(ag) | np.isnan(al),
            np.nan,
            np.where(ag + al == 0, 50.0, 100.0 * ag / (ag + al)),
        )
    return pd.Series(values, index=close.index)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return close.pct_change(period)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX, +DI, -DI (±DM and TR are Wilder-smoothed before division)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=high.index, dtype=float
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index, dtype=float
    )
    tr_s = true_range(high, low, close).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    plus_dm_s = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    tr_values = tr_s.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = pd.Series(
            np.where(tr_values > 0, 100.0 * plus_dm_s.to_numpy(dtype=float) / tr_values, 0.0),
            index=high.index,
        )
        minus_di = pd.Series(
            np.where(tr_values > 0, 100.0 * minus_dm_s.to_numpy(dtype=float) / tr_values, 0.0),
            index=high.index,
        )
    di_sum = (plus_di + minus_di).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = pd.Series(
            np.where(
                np.isnan(di_sum) | (di_sum <= 0),
                0.0,
                100.0
                * np.abs(plus_di.to_numpy(dtype=float) - minus_di.to_numpy(dtype=float))
                / di_sum,
            ),
            index=high.index,
        )
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def bollinger(
    close: pd.Series, period: int = 20, dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """(mid, upper, lower, width, percent_b) — population std (ddof=0)."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + dev * std
    lower = mid - dev * std
    band = (upper - lower).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        width = pd.Series(
            np.where(mid.to_numpy(dtype=float) > 0, band / mid.to_numpy(dtype=float), np.nan),
            index=close.index,
        )
        percent_b = pd.Series(
            np.where(band > 0, (close.to_numpy(dtype=float) - lower.to_numpy(dtype=float)) / band, 0.5),
            index=close.index,
        )
    return mid, upper, lower, width, percent_b


def _last_value(series: pd.Series | None) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _percentile_rank(window: pd.Series, current: float) -> float | None:
    """Share of ``window`` values strictly below ``current`` (in [0, 1])."""
    values = window.dropna().to_numpy(dtype=float)
    if len(values) < 20:
        return None
    return float((values < current).mean())


# ==========================================================================
# Timeframe feature bundle
# ==========================================================================
@dataclass
class DayStats:
    prev_day_high: float | None = None
    prev_day_low: float | None = None
    current_day_high: float | None = None
    current_day_low: float | None = None
    prev_day: str | None = None


@dataclass
class TimeframeFeatures:
    """All deterministic features for one timeframe's closed candles."""

    timeframe: TimeFrame
    symbol: str
    df: pd.DataFrame
    length: int
    last_time: datetime | None
    last_close: float | None

    # trend
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    ema_alignment: str | None = None  # up | down | flat | mixed | None
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    slope_atr: float | None = None

    # momentum
    rsi: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    macd_hist_prev: float | None = None
    roc: float | None = None
    impulse_atr: float | None = None

    # data sanity
    #: last candle body is absurd (>20x trailing ATR or >10% of price) —
    #: almost certainly a bad tick; agents must suppress signals
    last_candle_anomaly: bool = False

    # volatility
    atr: float | None = None
    atr_pct: float | None = None
    atr_ratio: float | None = None
    bb_width: float | None = None
    bb_width_pct: float | None = None
    percent_b: float | None = None
    bb_expansion: float | None = None
    bb_mid: float | None = None
    bb_lower: float | None = None
    bb_upper: float | None = None

    # structure
    swings: list[SwingPoint] = field(default_factory=list)
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    bias: StructureBias = StructureBias.UNKNOWN
    events: list[StructureEvent] = field(default_factory=list)
    last_event: StructureEvent | None = None

    # liquidity / volume
    volume_ratio: float | None = None
    day_stats: DayStats = field(default_factory=DayStats)
    last_candle_body_atr: float | None = None
    last_candle_upper_wick_ratio: float | None = None
    last_candle_lower_wick_ratio: float | None = None

    #: raw indicator series for agents needing history (rsi, macd_hist, close…)
    indicators: dict[str, pd.Series] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def has_ema200(self) -> bool:
        return self.ema200 is not None

    def recent_events(self, limit: int = 5) -> list[StructureEvent]:
        return self.events[-limit:]

    def rolling_extremes(
        self, window: int = 50, *, exclude_last: int = 0
    ) -> tuple[float | None, float | None]:
        """(low, high) of the last ``window`` closed candles.

        ``exclude_last`` drops the newest N candles first — used for
        "previous" reference levels (a sweep candle must not become its own
        level).
        """
        usable = self.length - exclude_last
        if usable <= 0:
            return None, None
        window = min(window, usable)
        lows = self.df["low"].to_numpy(dtype=float)[usable - window : usable]
        highs = self.df["high"].to_numpy(dtype=float)[usable - window : usable]
        return float(lows.min()), float(highs.max())

    def deviation_atr(self) -> float | None:
        """(close - SMA20) / ATR — ATR-normalized distance from the mean."""
        if self.atr in (None, 0) or self.bb_mid is None or self.last_close is None:
            return None
        return (self.last_close - self.bb_mid) / self.atr

    def structure_agrees_with(self, direction: str) -> bool:
        if direction == "up":
            return self.bias is StructureBias.UPTREND
        if direction == "down":
            return self.bias is StructureBias.DOWNTREND
        return False


def _day_stats(df: pd.DataFrame) -> DayStats:
    if df.empty:
        return DayStats()
    times = pd.to_datetime(df["time"])
    days = times.dt.date
    last_day = days.iloc[-1]
    current = df[days == last_day]
    prev_day = days[days < last_day].max()
    stats = DayStats(
        current_day_high=float(current["high"].max()),
        current_day_low=float(current["low"].min()),
        prev_day=str(prev_day) if prev_day is not None else None,
    )
    if prev_day is not None:
        previous = df[days == prev_day]
        stats.prev_day_high = float(previous["high"].max())
        stats.prev_day_low = float(previous["low"].min())
    return stats


def _candle_anatomy(df: pd.DataFrame, atr_value: float | None) -> tuple[float | None, float | None, float | None]:
    if df.empty or atr_value in (None, 0):
        return None, None, None
    row = df.iloc[-1]
    high, low = float(row["high"]), float(row["low"])
    open_, close = float(row["open"]), float(row["close"])
    range_ = high - low
    if range_ <= 0:
        return 0.0, 0.0, 0.0
    body = close - open_
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    return body / atr_value, upper / range_, lower / range_


def compute_features(series: CandleSeries) -> TimeframeFeatures:
    """Build the feature bundle for one timeframe (pure, deterministic)."""
    df = series.to_dataframe().reset_index(drop=True)
    length = len(df)
    base = TimeframeFeatures(
        timeframe=series.timeframe,
        symbol=series.symbol,
        df=df,
        length=length,
        last_time=df["time"].iloc[-1] if length else None,
        last_close=_last_value(df["close"]) if length else None,
    )
    if length == 0:
        return base

    close, high, low = df["close"], df["high"], df["low"]

    ema20, ema50, ema200 = ema(close, 20), ema(close, 50), ema(close, 200)
    adx_s, plus_di_s, minus_di_s = directional_index(high, low, close)
    atr_s = atr(high, low, close)
    rsi_s = rsi(close)
    macd_line, macd_signal, macd_hist = macd(close)
    roc_s = roc(close)
    bb_mid, bb_upper, bb_lower, bb_width, percent_b = bollinger(close)

    base.ema20 = _last_value(ema20)
    base.ema50 = _last_value(ema50)
    base.ema200 = _last_value(ema200)
    base.adx = _last_value(adx_s)
    base.plus_di = _last_value(plus_di_s)
    base.minus_di = _last_value(minus_di_s)
    base.rsi = _last_value(rsi_s)
    base.macd_line = _last_value(macd_line)
    base.macd_signal = _last_value(macd_signal)
    base.macd_hist = _last_value(macd_hist)
    base.macd_hist_prev = _last_value(macd_hist.iloc[:-1]) if len(macd_hist) > 1 else None
    base.roc = _last_value(roc_s)
    base.atr = _last_value(atr_s)

    # bad-tick guard: last candle body vs ATR computed WITHOUT it, and vs price
    if length >= 2 and base.last_close:
        last_open = float(df["open"].iloc[-1])
        body = abs(base.last_close - last_open)
        prev_atr = atr_s.iloc[-2]
        atr_ratio_bad = (
            not pd.isna(prev_atr) and prev_atr > 0 and body > 20.0 * float(prev_atr)
        )
        pct_bad = body / base.last_close > 0.10
        base.last_candle_anomaly = bool(atr_ratio_bad or pct_bad)

    base.bb_mid = _last_value(bb_mid)
    base.bb_upper = _last_value(bb_upper)
    base.bb_lower = _last_value(bb_lower)
    base.bb_width = _last_value(bb_width)
    base.percent_b = _last_value(percent_b)

    # volatility ratios & percentile ranks
    if base.atr is not None:
        base.atr_pct = base.atr / base.last_close if base.last_close else None
        atr_window = atr_s.dropna().iloc[-100:]
        if len(atr_window) >= 20:
            median_atr = float(atr_window.median())
            if median_atr > 0:
                base.atr_ratio = base.atr / median_atr
    if base.bb_width is not None:
        width_window = bb_width.iloc[-201:-1]
        base.bb_width_pct = _percentile_rank(width_window, base.bb_width)
        if len(bb_width.dropna()) >= 11:
            past = bb_width.dropna().iloc[-11]
            if past and past > 0:
                base.bb_expansion = base.bb_width / past

    # slope of last 20 closes, ATR-normalized
    if base.atr not in (None, 0) and length >= 20:
        closes = close.to_numpy(dtype=float)[-20:]
        slope = float(np.polyfit(np.arange(len(closes)), closes, 1)[0])
        base.slope_atr = slope / base.atr

    # impulse of the last candle, ATR-normalized
    body_atr, upper_ratio, lower_ratio = _candle_anatomy(df, base.atr)
    base.last_candle_body_atr = body_atr
    base.last_candle_upper_wick_ratio = upper_ratio
    base.last_candle_lower_wick_ratio = lower_ratio

    # EMA alignment with ATR buffer (flat within noise)
    if base.ema20 is not None and base.ema50 is not None and base.atr not in (None, 0):
        gap = (base.ema20 - base.ema50) / base.atr
        if abs(gap) < EMA_FLAT_ATR:
            base.ema_alignment = "flat"
        elif gap > 0:
            above200 = base.ema200 is None or base.ema50 > base.ema200
            base.ema_alignment = "up" if above200 else "mixed"
        else:
            below200 = base.ema200 is None or base.ema50 < base.ema200
            base.ema_alignment = "down" if below200 else "mixed"
    elif base.ema20 is not None and base.ema50 is not None:
        base.ema_alignment = "up" if base.ema20 > base.ema50 else "down"

    # structure
    base.swings = label_swings(detect_swings(df))
    base.swing_highs = [s for s in base.swings if s.kind is SwingKind.HIGH]
    base.swing_lows = [s for s in base.swings if s.kind is SwingKind.LOW]
    base.bias = structure_bias(base.swings)
    base.events = structure_events(df, base.swings)
    base.last_event = base.events[-1] if base.events else None

    # volume
    volumes = df["tick_volume"].to_numpy(dtype=float)
    if length >= 21:
        mean_volume = float(volumes[-21:-1].mean())
        if mean_volume > 0:
            base.volume_ratio = float(volumes[-1]) / mean_volume

    base.day_stats = _day_stats(df)

    base.indicators = {
        "close": close,
        "rsi": rsi_s,
        "macd_hist": macd_hist,
        "macd_line": macd_line,
        "atr": atr_s,
        "adx": adx_s,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
    }
    return base


# ==========================================================================
# Volatility classification (shared by VolatilityAgent and RegimeDetector)
# ==========================================================================
def classify_volatility(features: TimeframeFeatures) -> VolatilityLevel:
    """Documented thresholds:

    * EXTREME: ATR >= 2.5x its 100-bar median OR BB width in the top 3%
    * HIGH:    ATR >= 1.5x median OR BB width in the top 15%
    * LOW:     ATR <= 0.6x median OR BB width in the bottom 10%
    * NORMAL:  otherwise (including degenerate zero-width data)
    """
    atr_ratio = features.atr_ratio
    bb_pct = features.bb_width_pct
    if features.bb_width is not None and features.bb_width == 0:
        return VolatilityLevel.LOW
    if atr_ratio is not None and atr_ratio >= 2.5:
        return VolatilityLevel.EXTREME
    if bb_pct is not None and bb_pct >= 0.97:
        return VolatilityLevel.EXTREME
    if atr_ratio is not None and atr_ratio >= 1.5:
        return VolatilityLevel.HIGH
    if bb_pct is not None and bb_pct >= 0.85:
        return VolatilityLevel.HIGH
    if atr_ratio is not None and atr_ratio <= 0.6:
        return VolatilityLevel.LOW
    if bb_pct is not None and bb_pct <= 0.10:
        return VolatilityLevel.LOW
    return VolatilityLevel.NORMAL


def volatility_state(features: TimeframeFeatures) -> VolatilityState:
    """BB-width expansion/contraction vs 10 bars ago."""
    expansion = features.bb_expansion
    if expansion is None:
        return VolatilityState.STABLE
    if expansion >= 1.3:
        return VolatilityState.EXPANDING
    if expansion <= 0.75:
        return VolatilityState.CONTRACTING
    return VolatilityState.STABLE


def safe_float(value: float | None, digits: int = 6) -> float | None:
    """Round for JSON-safe feature reporting."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)
