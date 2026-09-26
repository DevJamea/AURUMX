"""LiquidityAgent behavioral tests (Phase-2 §7).

Categories: bullish sweep, bearish sweep, breakout (volume-confirmed,
low-volume, zero-volume), failed breakout (two-candle pattern, with and
without rejection wick), wick rejection, ordinary candle, missing/abnormal
volume.  All patterns are hand-crafted final candles against known levels.
"""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, DataQuality, TimeFrame
from tests.unit.agents.scenarios import (
    make_context,
    make_series,
    replace_candle,
    set_volumes,
    sideways_closes,
)

# 47 flat M15 candles (all same UTC day — no previous-day levels interfere),
# then one hand-crafted pattern candle.
BASE_CLOSES = sideways_closes(47, period=16, amplitude=0.5, seed=91)
BASE = make_series(BASE_CLOSES, TimeFrame.M15, seed=92)


def _levels(series):
    df = series.to_dataframe()
    return float(df["high"].iloc[-21:-1].max()), float(df["low"].iloc[-21:-1].min())


def _with_last(series, *, open_, high, low, close, volume=None):
    return replace_candle(series, -1, open_=open_, high=high, low=low, close=close, volume=volume)


class TestSweeps:
    def test_bullish_sweep_of_recent_low(self, agents):
        high20, low20 = _levels(BASE)
        series = _with_last(
            BASE, open_=low20 + 0.4, high=low20 + 0.55, low=low20 - 0.5, close=low20 + 0.1
        )
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.BUY
        assert result.signal_strength == pytest.approx(0.35)
        assert any("bullish sweep" in r for r in result.reasons)

    def test_bearish_sweep_of_recent_high(self, agents):
        high20, _ = _levels(BASE)
        series = _with_last(
            BASE, open_=high20 - 0.4, high=high20 + 0.5, low=high20 - 0.55, close=high20 - 0.1
        )
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.SELL
        assert result.signal_strength == pytest.approx(0.35)
        assert any("bearish sweep" in r for r in result.reasons)


class TestBreakouts:
    def test_volume_confirmed_breakout_buys(self, agents):
        high20, _ = _levels(BASE)
        series = _with_last(
            BASE, open_=high20 - 0.5, high=high20 + 1.0, low=high20 - 0.55, close=high20 + 0.7
        )
        series = set_volumes(series, 500.0, last=1500.0)  # ratio 3.0
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.BUY
        assert result.signal_strength == pytest.approx(0.30)
        assert any("volume confirmed" in r for r in result.reasons)
        assert result.features["volume_ratio"] == pytest.approx(3.0)

    def test_low_volume_breakout_stays_neutral_but_records_evidence(self, agents):
        """An unconfirmed breakout alone is below the action threshold — the
        evidence is preserved in reasons, the agent declines to signal."""
        high20, _ = _levels(BASE)
        series = _with_last(
            BASE, open_=high20 - 0.5, high=high20 + 1.0, low=high20 - 0.55, close=high20 + 0.7
        )
        series = set_volumes(series, 500.0, last=100.0)  # ratio 0.2
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == pytest.approx(0.2)
        assert any("breakout" in r for r in result.reasons)

    def test_zero_volume_breakout_warns_and_keeps_price_evidence(self, agents):
        high20, _ = _levels(BASE)
        series = _with_last(
            BASE, open_=high20 - 0.5, high=high20 + 1.0, low=high20 - 0.55, close=high20 + 0.7
        )
        series = set_volumes(series, 0.0)
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert any("volume" in w.lower() for w in result.warnings)
        assert any("breakout" in r for r in result.reasons)


class TestFailedBreakout:
    def test_failed_breakout_with_rejection_wick_sells(self, agents):
        """Two-candle pattern: previous close above the level, last close back
        inside with an upper rejection wick — sweep + failed breakout."""
        high20, _ = _levels(BASE)
        series = replace_candle(
            BASE, -2, open_=high20 - 0.3, high=high20 + 0.6, low=high20 - 0.35, close=high20 + 0.4
        )
        series = replace_candle(
            series, -1, open_=high20 + 0.3, high=high20 + 1.0, low=high20 - 0.2, close=high20 - 0.05
        )
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.SELL
        assert result.signal_strength == pytest.approx(0.60, abs=0.05)
        assert any("failed breakout" in r for r in result.reasons)

    def test_lone_failed_breakout_below_threshold_is_neutral(self, agents):
        """A failed breakout alone (0.25) is intentionally below the 0.30
        action threshold — evidence recorded, no signal."""
        high20, _ = _levels(BASE)
        series = replace_candle(
            BASE, -2, open_=high20 - 0.3, high=high20 + 0.6, low=high20 - 0.35, close=high20 + 0.4
        )
        # last candle: closes back below the level with a small body and no
        # rejection wick (upper wick ~25% of range, below the 60% threshold)
        series = replace_candle(
            series, -1, open_=high20 + 0.3, high=high20 + 0.5, low=high20 - 0.3, close=high20 - 0.05
        )
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == pytest.approx(0.25)
        assert any("failed breakout" in r for r in result.reasons)


class TestWickRejection:
    def test_upper_wick_rejection_at_high_sells(self, agents):
        high20, _ = _levels(BASE)
        series = _with_last(
            BASE, open_=high20 - 0.2, high=high20 + 0.8, low=high20 - 0.25, close=high20 - 0.15
        )
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.SELL
        assert any("wick rejection" in r for r in result.reasons)


class TestOrdinaryAndDegraded:
    def test_ordinary_candle_is_neutral(self, agents):
        series = _with_last(BASE, open_=2650.0, high=2650.6, low=2649.5, close=2650.3)
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0
        assert any("no liquidity-relevant" in r for r in result.reasons)

    def test_insufficient_candles(self, agents):
        short = make_series(
            sideways_closes(30, period=16, amplitude=0.5, seed=91), TimeFrame.M15, seed=92
        )
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: short}))
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality is DataQuality.INSUFFICIENT

    def test_abnormal_volume_ratio_is_reported(self, agents):
        high20, _ = _levels(BASE)
        series = _with_last(
            BASE, open_=high20 - 0.5, high=high20 + 1.0, low=high20 - 0.55, close=high20 + 0.7
        )
        series = set_volumes(series, 500.0, last=5000.0)  # ratio 10
        result = agents["liquidity"].analyze(make_context({TimeFrame.M15: series}))
        assert result.features["volume_ratio"] == pytest.approx(10.0)
        assert result.direction is AgentDirection.BUY  # volume-confirmed
