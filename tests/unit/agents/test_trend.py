"""TrendAgent behavioral tests (Phase-2 §4).

Categories: strong bullish, strong bearish, sideways, insufficient data, and
the explicit rule that EMA20 > EMA50 ALONE never produces a signal.
"""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, DataQuality, TimeFrame
from tests.unit.agents.scenarios import (
    linear_trend_closes,
    make_context,
    make_series,
    sideways_closes,
    standard_triple,
)


class TestTrendBullish:
    def test_strong_uptrend_is_buy(self, agents):
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.BUY
        assert result.signal_strength == pytest.approx(0.85)  # all evidence except structure
        assert result.is_actionable
        assert result.data_quality is DataQuality.OK
        assert any("EMA20 above EMA50" in r for r in result.reasons)
        assert result.features["trend_state"] == "STRONG_BULLISH"
        assert result.features["ema_alignment_h1"] == "up"
        assert result.primary_timeframe is TimeFrame.H1
        assert set(result.timeframes_used) == {TimeFrame.H1, TimeFrame.H4}

    def test_uptrend_reasons_explain_every_piece_of_evidence(self, agents):
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        result = agents["trend"].analyze(ctx)
        joined = " | ".join(result.reasons)
        assert "ADX" in joined  # directional strength cited
        assert "slope" in joined.lower()
        assert "H4" in joined
        assert result.reasons, "provenance required"


class TestTrendBearish:
    def test_strong_downtrend_is_sell(self, agents):
        closes = linear_trend_closes(300, slope=-2.0, seed=12)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=12))
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.SELL
        assert result.signal_strength == pytest.approx(0.85)
        assert result.features["trend_state"] == "STRONG_BEARISH"
        assert result.features["ema_alignment_h1"] == "down"


class TestTrendSideways:
    def test_sideways_is_neutral(self, agents):
        # verified RANGE-regime dataset (sine p=16, amp 1.2, seed 71)
        closes = sideways_closes(300, period=16, amplitude=1.2, seed=71)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=71))
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert not result.is_actionable
        assert result.data_quality is DataQuality.OK
        assert result.features["trend_state"] == "NEUTRAL"

    def test_flat_market_is_neutral(self, agents):
        closes = [2650.0] * 300
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=7))
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0


class TestTrendNotNaiveAlignment:
    def test_ema20_above_ema50_alone_is_not_a_buy(self, agents):
        """Young noisy ramp: EMA20 > EMA50 holds, but the agent demands
        E1 + >=2 confirmations and must stay neutral."""
        import random

        rng = random.Random(99)
        ramp = [round(2650.0 + 0.10 * i + rng.gauss(0, 1.5), 2) for i in range(65)]
        # H4 deliberately sideways so E5 (macro agreement) is false
        series = standard_triple(
            h1_closes=ramp,
            h1_seed=99,
            h4_closes=sideways_closes(65, period=16, amplitude=1.0, seed=98),
        )
        ctx = make_context(series)
        f = ctx.features(TimeFrame.H1)
        # precondition: the alignment genuinely exists
        assert f.ema20 is not None and f.ema50 is not None and f.ema20 > f.ema50
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert not result.is_actionable
        assert any("no actionable trend evidence" in r for r in result.reasons)


class TestTrendInsufficient:
    def test_fewer_than_60_candles_is_insufficient(self, agents):
        closes = linear_trend_closes(45, slope=2.0)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=13))
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality is DataQuality.INSUFFICIENT
        assert not result.is_actionable

    def test_missing_h1_is_unavailable(self, agents):
        ctx = make_context({TimeFrame.M15: make_series(linear_trend_closes(80), TimeFrame.M15)})
        result = agents["trend"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality in (DataQuality.INSUFFICIENT, DataQuality.INVALID)
        assert result.signal_strength == 0.0

    def test_ema200_warmup_degrades_quality(self, agents):
        """60-199 candles: EMA200 evidence excluded -> DEGRADED + warning."""
        closes = linear_trend_closes(120, slope=2.0, seed=14)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=14))
        result = agents["trend"].analyze(ctx)
        assert result.data_quality is DataQuality.DEGRADED
        assert any("EMA200" in w for w in result.warnings)
