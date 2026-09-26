"""RegimeDetector tests (Phase-2 §11).

Categories: TREND_UP, TREND_DOWN, RANGE, HIGH_VOLATILITY, LOW_VOLATILITY,
UNCERTAIN (conflicting timeframes), insufficient data, and the
regime-relevance model (weights, matrix, weighted strength).
"""

from __future__ import annotations

import math
import random

import pytest

from app.core.enums import AgentDirection, MarketRegime, TimeFrame, VolatilityLevel
from app.decision.regime import (
    AGENT_RELEVANCE_WEIGHTS,
    RegimeDetector,
    relevance_for,
    relevance_of,
    weighted_strength,
)
from tests.unit.agents.scenarios import (
    linear_trend_closes,
    make_context,
    make_series,
    sideways_closes,
    standard_triple,
)


def _sine(count, *, period=16, amp=1.0, amp_end=None, seed=1, base=2650.0, noise=0.05):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        a = amp if amp_end is None else amp + (amp_end - amp) * i / max(count - 1, 1)
        out.append(round(base + a * math.sin(2 * math.pi * i / period) + rng.gauss(0, noise), 2))
    return out


def _detect(closes, *, h1_seed=7, h4_closes=None, m15_closes=None):
    series = standard_triple(
        h1_closes=closes, h1_seed=h1_seed, h4_closes=h4_closes, m15_closes=m15_closes
    )
    return RegimeDetector().detect(make_context(series))


class TestTrendRegimes:
    def test_uptrend_detected(self):
        a = _detect(linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11)
        assert a.regime is MarketRegime.TREND_UP
        assert a.data_quality.value == "OK"
        assert a.evidence, "evidence strings required"

    def test_downtrend_detected(self):
        a = _detect(linear_trend_closes(300, slope=-2.0, seed=12), h1_seed=12)
        assert a.regime is MarketRegime.TREND_DOWN

    def test_h4_conflict_makes_uncertain(self):
        """H1 up but H4 down: the macro timeframe opposes — UNCERTAIN, never
        a confident TREND call."""
        a = _detect(
            linear_trend_closes(300, slope=2.0, seed=11),
            h1_seed=11,
            h4_closes=linear_trend_closes(300, slope=-2.0, seed=12),
        )
        assert a.regime is MarketRegime.UNCERTAIN
        assert any("H4" in c for c in a.conflicts)


class TestRangeAndVolatilityRegimes:
    def test_range_detected(self):
        # verified dataset: sine p=16 amp=1.2 seed=71 -> RANGE/NORMAL
        a = _detect(sideways_closes(300, period=16, amplitude=1.2, seed=71), h1_seed=71)
        assert a.regime is MarketRegime.RANGE
        assert a.volatility is VolatilityLevel.NORMAL

    def test_low_volatility_detected(self):
        closes = _sine(200, amp=1.2, seed=51) + _sine(100, amp=0.5, amp_end=0.1, seed=52)
        a = _detect(closes)
        assert a.regime is MarketRegime.LOW_VOLATILITY
        assert a.volatility is VolatilityLevel.LOW

    def test_high_volatility_detected_on_burst(self):
        closes = _sine(290, amp=1.2, seed=71) + [2656.0, 2644.0, 2656.0, 2644.5, 2655.5, 2645.0]
        a = _detect(closes)
        assert a.regime is MarketRegime.HIGH_VOLATILITY
        assert a.volatility is VolatilityLevel.EXTREME

    def test_regime_volatility_axis_is_independent(self):
        """A trend can coexist with any volatility level."""
        a = _detect(linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11)
        assert a.regime is MarketRegime.TREND_UP
        assert a.volatility is not None  # axis always populated


class TestUncertainAndInsufficient:
    def test_insufficient_h1_is_uncertain(self):
        a = _detect(linear_trend_closes(50, slope=2.0))
        assert a.regime is MarketRegime.UNCERTAIN
        assert a.data_quality.value == "INSUFFICIENT"

    def test_min_h1_constant(self):
        assert RegimeDetector.MIN_H1_CANDLES == 60

    def test_no_h1_at_all(self):
        series = {TimeFrame.M15: make_series(linear_trend_closes(80), TimeFrame.M15)}
        a = RegimeDetector().detect(make_context(series))
        assert a.regime is MarketRegime.UNCERTAIN


class TestRelevanceModel:
    def test_trend_regime_prioritizes_trend_agents(self):
        mapping = relevance_for(MarketRegime.TREND_UP)
        assert mapping["trend"] == "HIGH"
        assert mapping["momentum"] == "HIGH"
        assert mapping["structure"] == "HIGH"
        assert mapping["mean_reversion"] == "DISABLED"

    def test_range_regime_prioritizes_mean_reversion(self):
        mapping = relevance_for(MarketRegime.RANGE)
        assert mapping["mean_reversion"] == "HIGH"
        assert mapping["trend"] == "REDUCED"

    def test_every_regime_covers_every_agent(self):
        agents = {"trend", "momentum", "structure", "liquidity", "volatility", "mean_reversion", "macro"}
        for regime in MarketRegime:
            assert set(relevance_for(regime)) >= agents - {"macro"} | {"macro"}

    def test_weights_are_documented_values(self):
        assert AGENT_RELEVANCE_WEIGHTS == {
            "HIGH": 1.25,
            "NORMAL": 1.0,
            "REDUCED": 0.5,
            "DISABLED": 0.0,
        }

    def test_weighted_strength_math(self):
        assert weighted_strength(AgentDirection.BUY, 0.8, "HIGH") == pytest.approx(1.0)
        assert weighted_strength(AgentDirection.BUY, 0.8, "NORMAL") == pytest.approx(0.8)
        assert weighted_strength(AgentDirection.BUY, 0.8, "REDUCED") == pytest.approx(0.4)
        assert weighted_strength(AgentDirection.BUY, 0.8, "DISABLED") == pytest.approx(0.0)
        assert weighted_strength(AgentDirection.NEUTRAL, 0.8, "HIGH") == 0.0

    def test_relevance_of_known_and_unknown_agents(self):
        assert relevance_of(MarketRegime.TREND_UP, "mean_reversion") == "DISABLED"
        assert relevance_of(MarketRegime.RANGE, "mean_reversion") == "HIGH"
        # unknown agent names degrade to NORMAL, never crash
        assert relevance_of(MarketRegime.RANGE, "nonexistent_agent") == "NORMAL"


class TestRegimeDeterminism:
    def test_same_input_same_regime(self):
        closes = sideways_closes(300, period=16, amplitude=1.2, seed=71)
        first = _detect(closes, h1_seed=71)
        second = _detect(closes, h1_seed=71)
        assert first == second

    def test_assessment_carries_provenance(self):
        a = _detect(linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11)
        assert a.as_of is not None
        assert isinstance(a.evidence, list) and a.evidence
        assert all(isinstance(e, str) for e in a.evidence)
