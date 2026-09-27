"""Timeframe alignment tests (Phase-3 §3)."""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, TimeFrame
from app.decision import DecisionEngineConfig
from app.decision.alignment import compute_alignment
from tests.unit.agents.scenarios import (
    linear_trend_closes,
    make_context,
    make_series,
    standard_triple,
)


class TestAligned:
    def test_full_bullish_alignment_scores_one(self):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=up, h1_seed=11))
        alignment = compute_alignment(ctx, AgentDirection.BUY, DecisionEngineConfig())
        assert alignment.score == 1.0
        assert set(alignment.agreeing) == {TimeFrame.H4, TimeFrame.H1, TimeFrame.M15}
        assert alignment.opposing == ()
        assert alignment.missing == ()

    def test_full_bearish_alignment_against_buy_scores_zero(self):
        down = linear_trend_closes(300, slope=-2.0, seed=12)
        ctx = make_context(standard_triple(h1_closes=down, h1_seed=12))
        alignment = compute_alignment(ctx, AgentDirection.BUY, DecisionEngineConfig())
        assert alignment.score == 0.0
        assert set(alignment.opposing) == {TimeFrame.H4, TimeFrame.H1, TimeFrame.M15}

    def test_m15_opposition_drops_score_below_default_threshold(self):
        """H4+H1 bullish, M15 bearish: 0.75 < 0.80 default -> engine HOLDs."""
        up = linear_trend_closes(300, slope=2.0, seed=11)
        down_m15 = linear_trend_closes(300, slope=-0.8, seed=55)
        series = {
            TimeFrame.M15: make_series(down_m15, TimeFrame.M15, seed=12),
            TimeFrame.H1: make_series(up, TimeFrame.H1, seed=11),
            TimeFrame.H4: make_series(up, TimeFrame.H4, seed=11),
        }
        ctx = make_context(series)
        alignment = compute_alignment(ctx, AgentDirection.BUY, DecisionEngineConfig())
        assert alignment.score == 0.75
        assert alignment.opposing == (TimeFrame.M15,)

    def test_strong_read_flag(self):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=up, h1_seed=11))
        alignment = compute_alignment(ctx, AgentDirection.BUY, DecisionEngineConfig())
        assert all(read.strong for read in alignment.reads)


class TestMissingAndNeutral:
    def test_missing_timeframes_are_listed_and_renormalized(self):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        series = standard_triple(h1_closes=up, h1_seed=11)
        del series[TimeFrame.H4]
        ctx = make_context(series)
        alignment = compute_alignment(ctx, AgentDirection.BUY, DecisionEngineConfig())
        assert alignment.missing == (TimeFrame.H4,)
        # H1 (0.45) + M15 (0.25) both agree -> renormalized score 1.0
        assert alignment.score == 1.0

    def test_renormalized_flag_and_effective_weights(self):
        """With H4 absent the renormalized weights in force are recorded:
        H1 0.45/0.70 = 0.642857, M15 0.25/0.70 = 0.357143."""
        up = linear_trend_closes(300, slope=2.0, seed=11)
        series = standard_triple(h1_closes=up, h1_seed=11)
        del series[TimeFrame.H4]
        alignment = compute_alignment(
            make_context(series), AgentDirection.BUY, DecisionEngineConfig()
        )
        assert alignment.renormalized is True
        weights = dict(alignment.effective_weights)
        assert weights[TimeFrame.H1] == pytest.approx(0.642857, abs=1e-6)
        assert weights[TimeFrame.M15] == pytest.approx(0.357143, abs=1e-6)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_no_renormalization_when_all_present(self):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        alignment = compute_alignment(
            make_context(standard_triple(h1_closes=up, h1_seed=11)),
            AgentDirection.BUY, DecisionEngineConfig(),
        )
        assert alignment.renormalized is False
        assert dict(alignment.effective_weights) == {
            TimeFrame.H4: pytest.approx(0.30),
            TimeFrame.H1: pytest.approx(0.45),
            TimeFrame.M15: pytest.approx(0.25),
        }

    def test_summary_is_json_safe_and_complete(self):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        series = standard_triple(h1_closes=up, h1_seed=11)
        del series[TimeFrame.H4]
        alignment = compute_alignment(
            make_context(series), AgentDirection.BUY, DecisionEngineConfig()
        )
        summary = alignment.summary()
        assert summary["missing"] == ["H4"]
        assert summary["renormalized"] is True
        assert summary["effective_weights"]["H1"] == pytest.approx(0.642857, abs=1e-6)
        import json

        json.dumps(summary)

    def test_flat_market_is_neutral_not_opposing(self):
        ctx = make_context(standard_triple(h1_closes=[2650.0] * 300, h1_seed=7))
        alignment = compute_alignment(ctx, AgentDirection.BUY, DecisionEngineConfig())
        assert alignment.score == 0.0
        assert alignment.opposing == ()  # flat is absence of direction
        assert all(read.direction is AgentDirection.NEUTRAL for read in alignment.reads)

    def test_weights_are_configurable(self):
        from app.decision.config import TimeframeWeights

        config = DecisionEngineConfig(
            timeframe_weights=TimeframeWeights(h4=0.1, h1=0.1, m15=0.8)
        )
        up = linear_trend_closes(300, slope=2.0, seed=11)
        down_m15 = linear_trend_closes(300, slope=-0.8, seed=55)
        series = {
            TimeFrame.M15: make_series(down_m15, TimeFrame.M15, seed=12),
            TimeFrame.H1: make_series(up, TimeFrame.H1, seed=11),
            TimeFrame.H4: make_series(up, TimeFrame.H4, seed=11),
        }
        ctx = make_context(series)
        alignment = compute_alignment(ctx, AgentDirection.BUY, config)
        # M15 dominates: 0.2 agreement out of 1.0
        assert alignment.score == 0.2
