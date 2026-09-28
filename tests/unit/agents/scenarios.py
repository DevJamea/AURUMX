"""Deterministic synthetic market scenarios for agent tests.

Every generator is seeded and clock-anchored: the same call always produces
the same candles, so behavioral assertions are exact and reproducible.

Time anchoring follows the Phase-1 convention: a snapshot "created" at
``REF_TIME`` (Monday 2026-01-05 12:00 UTC) has its last *closed* M15 candle at
11:45, H1 at 11:00 and H4 at 08:00.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.enums import SessionState, TimeFrame
from app.core.models import (
    Candle,
    CandleSeries,
    MarketTick,
    SeriesCheck,
    SymbolSpec,
    TickCheck,
    ValidationReport,
)
from app.market.market_state import MarketSnapshot

REF_TIME = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


# ==========================================================================
# time grid helpers
# ==========================================================================
def last_closed_time(timeframe: TimeFrame, now: datetime = REF_TIME) -> datetime:
    step = timeframe.seconds
    floored = datetime.fromtimestamp((int(now.timestamp()) // step) * step, tz=UTC)
    return floored - timedelta(seconds=step)


def candle_times(timeframe: TimeFrame, count: int, now: datetime = REF_TIME) -> list[datetime]:
    end = last_closed_time(timeframe, now)
    return [
        end - timedelta(seconds=timeframe.seconds * (count - 1 - i))
        for i in range(count)
    ]


# ==========================================================================
# close-series generators (deterministic)
# ==========================================================================
def trending_closes(
    count: int,
    *,
    drift: float = 0.0020,
    volatility: float = 0.0012,
    base: float = 2650.0,
    seed: int = 11,
) -> list[float]:
    """Geometric random walk with constant drift (up with +drift, down with −)."""
    rng = random.Random(seed)
    closes: list[float] = []
    price = base
    for _ in range(count):
        price *= 1 + drift + rng.gauss(0.0, volatility)
        closes.append(round(price, 2))
    return closes


def linear_trend_closes(
    count: int,
    *,
    slope: float = 2.0,
    noise: float = 0.30,
    base: float = 2650.0,
    seed: int = 11,
) -> list[float]:
    """Arithmetic (linear) drift with small noise — stable MACD/ADX readings,
    unlike compounding drift which decays into MACD deceleration."""
    rng = random.Random(seed)
    return [round(base + slope * i + rng.gauss(0, noise), 2) for i in range(count)]


def sideways_closes(
    count: int,
    *,
    amplitude: float = 2.0,
    period: int = 40,
    base: float = 2650.0,
    noise: float = 0.05,
    seed: int = 13,
) -> list[float]:
    """Mean-reverting sine oscillation (no random-walk drift)."""
    rng = random.Random(seed)
    return [
        round(base + amplitude * math.sin(2 * math.pi * i / period) + rng.gauss(0, noise), 2)
        for i in range(count)
    ]


def zigzag_closes(waypoints: list[float], *, step: int = 3) -> list[float]:
    """Piecewise-linear path through ``waypoints`` — clean fractal swings.

    Waypoint values land on candles at indices ``step-1, 2*step-1, ...`` and
    become confirmed swings two candles later.
    """
    closes: list[float] = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        for k in range(step):
            closes.append(round(a + (b - a) * (k + 1) / step, 2))
    return closes


def flat_closes(count: int, *, base: float = 2650.0) -> list[float]:
    return [round(base, 2)] * count


# ==========================================================================
# candle series builders
# ==========================================================================
def make_series(
    closes: list[float],
    timeframe: TimeFrame,
    *,
    now: datetime = REF_TIME,
    seed: int = 7,
    volume: float = 500.0,
    wick: float = 0.30,
    symbol: str = "XAUUSD",
) -> CandleSeries:
    """Candles from a close path; opens chain from the previous close."""
    rng = random.Random(seed)
    times = candle_times(timeframe, len(closes), now)
    candles: list[Candle] = []
    previous: float | None = None
    for time, close in zip(times, closes, strict=True):
        open_ = previous if previous is not None else close
        high = max(open_, close) + rng.uniform(0.05, wick)
        low = min(open_, close) - rng.uniform(0.05, wick)
        candles.append(
            Candle(
                time=time,
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=close,
                tick_volume=volume,
                real_volume=0.0,
            )
        )
        previous = close
    return CandleSeries(symbol=symbol, timeframe=timeframe, candles=candles)


def append_candle(
    series: CandleSeries,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 500.0,
) -> CandleSeries:
    """Append one hand-crafted candle at the next slot (newest last)."""
    last_time = series.candles[-1].time
    candle = Candle(
        time=last_time + timedelta(seconds=series.timeframe.seconds),
        open=round(open_, 2),
        high=round(high, 2),
        low=round(low, 2),
        close=round(close, 2),
        tick_volume=volume,
        real_volume=0.0,
    )
    return CandleSeries(
        symbol=series.symbol,
        timeframe=series.timeframe,
        candles=[*series.candles, candle],
    )


def replace_candle(
    series: CandleSeries,
    index: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float | None = None,
) -> CandleSeries:
    """Replace the candle at ``index`` (negative allowed) with a hand-crafted
    one — exact wick/volume control for liquidity patterns."""
    i = index if index >= 0 else len(series.candles) + index
    old = series.candles[i]
    candles = list(series.candles)
    candles[i] = Candle(
        time=old.time,
        open=round(open_, 2),
        high=round(high, 2),
        low=round(low, 2),
        close=round(close, 2),
        tick_volume=old.tick_volume if volume is None else volume,
        real_volume=0.0,
    )
    return CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)


def replace_last_candle(
    series: CandleSeries,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float | None = None,
) -> CandleSeries:
    """Replace the newest candle with a hand-crafted one (exact wick control)."""
    return replace_candle(
        series,
        -1,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def set_volumes(
    series: CandleSeries, volume: float, *, last: float | None = None
) -> CandleSeries:
    """Set a uniform baseline tick volume; ``last`` overrides the newest candle
    (for volume-confirmation patterns without distorting the mean)."""
    candles = [
        c.model_copy(update={"tick_volume": last if (i == len(series.candles) - 1 and last is not None) else volume})
        for i, c in enumerate(series.candles)
    ]
    return CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)


# ==========================================================================
# snapshot / context helpers
# ==========================================================================
def gold_symbol_spec(name: str = "XAUUSD") -> SymbolSpec:
    from app.core.enums import SymbolTradeMode

    return SymbolSpec(
        name=name,
        visible=True,
        selected=True,
        trade_mode=SymbolTradeMode.FULL,
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=1.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level_points=20,
        freeze_level_points=10,
        currency_profit="USD",
    )


def make_tick(*, time: datetime | None = None, bid: float = 2650.0, ask: float = 2650.2) -> MarketTick:
    return MarketTick(symbol="XAUUSD", time=time or REF_TIME - timedelta(seconds=2), bid=bid, ask=ask)


def make_snapshot(
    series_map: dict[TimeFrame, CandleSeries],
    *,
    created_at: datetime = REF_TIME,
    valid: bool = True,
    fresh: bool = True,
    tick: MarketTick | None = None,
    tick_valid: bool = True,
    tick_fresh: bool = True,
    spread_points: float = 20.0,
    session_state: SessionState | None = SessionState.OPEN,
) -> MarketSnapshot:
    """A hand-built snapshot from synthetic series (defaults: valid, fresh,
    OPEN session — Phase-3 decision scenarios tweak the knobs)."""
    tick_check = TickCheck(
        tick=tick or make_tick(time=created_at - timedelta(seconds=2)),
        valid=tick_valid,
        fresh=tick_fresh,
        report=ValidationReport(),
        spread=0.2,
        spread_points=spread_points,
        age_seconds=2.0,
    )
    series_checks = {
        timeframe: SeriesCheck(
            timeframe=timeframe,
            series=series,
            valid=valid,
            fresh=fresh,
            report=ValidationReport(),
            age_seconds=0.0,
        )
        for timeframe, series in series_map.items()
    }
    return MarketSnapshot(
        created_at=created_at,
        symbol=gold_symbol_spec(),
        tick=tick_check,
        series=series_checks,
        session_state=session_state or SessionState.UNKNOWN,
        report=ValidationReport(),
    )


def make_context(
    series_map: dict[TimeFrame, CandleSeries],
    *,
    created_at: datetime = REF_TIME,
    valid: bool = True,
    fresh: bool = True,
    regime: Any = None,
):
    from app.agents.context import build_market_context

    snapshot = make_snapshot(series_map, created_at=created_at, valid=valid, fresh=fresh)
    context = build_market_context(snapshot)
    if regime is not None:
        context = context.with_regime(regime)
    return context


def standard_triple(
    *,
    h1_closes: list[float],
    m15_closes: list[float] | None = None,
    h4_closes: list[float] | None = None,
    h1_seed: int = 7,
) -> dict[TimeFrame, CandleSeries]:
    """M15/H1/H4 map; M15 and H4 default to following the H1 shape."""
    return {
        TimeFrame.M15: make_series(
            m15_closes or h1_closes, TimeFrame.M15, seed=h1_seed + 1
        ),
        TimeFrame.H1: make_series(h1_closes, TimeFrame.H1, seed=h1_seed),
        TimeFrame.H4: make_series(h4_closes or h1_closes, TimeFrame.H4, seed=h1_seed + 2),
    }
