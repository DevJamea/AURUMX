"""MomentumAgent behavioral tests (Phase-2 §5).

Categories: trend-mode BUY/SELL, regime-conditional range mode (no naive
RSI<30 buys), exhausted trend, conflicting indicators, insufficient data.
"""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, DataQuality, TimeFrame
from tests.unit.agents.scenarios import (
    linear_trend_closes,
    make_context,
    replace_last_candle,
    sideways_closes,
    standard_triple,
)


def _oversold_range_triple(*, m15_bullish: bool):
    """Range data with an oversold tail; the M15 close decides confirmation."""
    decline = [round(2650.0 - 0.22 * (i + 1), 2) for i in range(14)]
    h1 = sideways_closes(280, period=16, amplitude=1.2, seed=71) + decline
    triple = standard_triple(h1_closes=h1, h1_seed=71)
    if m15_bullish:
        triple[TimeFrame.M15] = replace_last_candle(
            triple[TimeFrame.M15], open_=2647.6, high=2648.6, low=2646.8, close=2648.2
        )
    else:
        triple[TimeFrame.M15] = replace_last_candle(
            triple[TimeFrame.M15], open_=2648.2, high=2648.6, low=2646.8, close=2647.0
        )
    return triple


class TestMomentumTrendMode:
    def test_steady_uptrend_is_buy(self, agents):
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.BUY
        assert result.is_actionable
        assert result.features["mode"] == "trend"
        assert result.features["rsi_h1"] == pytest.approx(100.0, abs=5.0)
        joined = " | ".join(result.reasons)
        assert "MACD line positive" in joined

    def test_steady_downtrend_is_sell(self, agents):
        closes = linear_trend_closes(300, slope=-2.0, seed=12)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=12))
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.SELL
        assert result.is_actionable
        # full evidence stack on a clean downtrend
        assert result.signal_strength == pytest.approx(1.0)


class TestMomentumRangeMode:
    def test_oversold_in_range_with_m15_confirmation_buys(self, agents, range_regime):
        triple = _oversold_range_triple(m15_bullish=True)
        ctx = make_context(triple, regime=range_regime)
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.BUY
        assert result.is_actionable
        assert result.features["mode"] == "range"
        joined = " | ".join(result.reasons)
        assert "oversold" in joined
        assert "M15" in joined

    def test_oversold_without_m15_confirmation_is_neutral(self, agents, range_regime):
        triple = _oversold_range_triple(m15_bullish=False)
        ctx = make_context(triple, regime=range_regime)
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert not result.is_actionable
        assert any("confirmation missing" in r for r in result.reasons)

    def test_no_naive_rsi_oversold_buy_in_trend_regime(self, agents, trend_down_regime):
        """The same oversold data under a TREND_DOWN regime must NOT become a
        BUY — RSI 20 in a downtrend is pullback continuation, not a reversal."""
        triple = _oversold_range_triple(m15_bullish=True)
        ctx = make_context(triple, regime=trend_down_regime)
        result = agents["momentum"].analyze(ctx)
        assert result.direction is not AgentDirection.BUY


class TestMomentumExhaustion:
    def test_decelerating_rally_is_exhausted_not_a_chase(self, agents, trend_up_regime):
        base = linear_trend_closes(300, slope=2.0, noise=0.2, seed=21)
        tail = [round(base[-1] + 0.2 * (i + 1), 2) for i in range(15)]
        ctx = make_context(
            standard_triple(h1_closes=base + tail, h1_seed=21), regime=trend_up_regime
        )
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert not result.is_actionable
        assert any("exhausted" in r for r in result.reasons)
        assert any("exhaust" in w.lower() for w in result.warnings)


class TestMomentumConflict:
    def test_rsi_macd_conflict_reduces_strength_and_warns(self, agents, trend_up_regime):
        """Rally with a 7x1.2 pullback: RSI ~71 (extended) while MACD line
        stays positive and histogram negative — conflict, not a clean buy."""
        base = linear_trend_closes(280, slope=2.0, noise=0.2, seed=31)
        pullback = [round(base[-1] - 1.2 * (i + 1), 2) for i in range(7)]
        ctx = make_context(
            standard_triple(h1_closes=base + pullback, h1_seed=31), regime=trend_up_regime
        )
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.BUY
        assert result.signal_strength < 0.45  # reduced by the conflict factor
        assert any("conflict" in w.lower() for w in result.warnings)

    def test_clean_uptrend_has_no_conflict_warning(self, agents):
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        result = agents["momentum"].analyze(ctx)
        assert not [w for w in result.warnings if "conflict" in w.lower()]


class TestMomentumData:
    def test_insufficient_candles(self, agents):
        closes = linear_trend_closes(40, slope=2.0)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=15))
        result = agents["momentum"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality is DataQuality.INSUFFICIENT

    def test_missing_m15_degrades_but_keeps_h1_analysis(self, agents):
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        series = standard_triple(h1_closes=closes, h1_seed=11)
        del series[TimeFrame.M15]
        ctx = make_context(series)
        result = agents["momentum"].analyze(ctx)
        # H1 evidence still analyzed; M15 corroboration lost
        assert result.data_quality is DataQuality.DEGRADED
        assert any("M15" in w for w in result.warnings)
