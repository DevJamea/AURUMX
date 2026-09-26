"""VolatilityAgent behavioral tests (Phase-2 §8).

Categories: LOW / NORMAL / HIGH / EXTREME levels, expansion/contraction
state, risk warning on extremes, insufficient data, and the rule that high
volatility alone never produces an aggressive directional signal.
"""

from __future__ import annotations

import math
import random

from app.core.enums import AgentDirection, DataQuality
from tests.unit.agents.scenarios import (
    make_context,
    standard_triple,
)


def _sine(count, *, period=16, amp=1.0, amp_end=None, seed=1, base=2650.0, noise=0.05):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        a = amp if amp_end is None else amp + (amp_end - amp) * i / max(count - 1, 1)
        out.append(round(base + a * math.sin(2 * math.pi * i / period) + rng.gauss(0, noise), 2))
    return out


def _analyze(agents, closes, seed=53):
    ctx = make_context(standard_triple(h1_closes=closes, h1_seed=seed))
    return agents["volatility"].analyze(ctx)


class TestLevels:
    def test_normal_volatility(self, agents):
        closes = _sine(300, amp=1.2, seed=42)
        result = _analyze(agents, closes)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.features["level"] == "NORMAL"
        assert result.signal_strength == 0.0

    def test_low_volatility_on_shrinking_range(self, agents):
        closes = _sine(200, amp=1.2, seed=51) + _sine(100, amp=0.5, amp_end=0.1, seed=52)
        result = _analyze(agents, closes)
        assert result.features["level"] == "LOW"
        assert result.features["state"] in ("STABLE", "CONTRACTING")

    def test_high_volatility_on_expansion(self, agents):
        # amplitude 0.8 -> 1.3: high (not extreme) volatility band
        closes = _sine(260, amp=0.8, seed=61) + _sine(40, amp=1.3, seed=62)
        result = _analyze(agents, closes)
        assert result.features["level"] == "HIGH"
        assert result.features["atr_ratio"] < 2.5  # not extreme
        assert result.features["bb_width_pct"] >= 0.85

    def test_extreme_volatility_on_burst(self, agents):
        closes = _sine(290, amp=1.2, seed=71) + [
            2656.0, 2644.0, 2656.0, 2644.5, 2655.5, 2645.0,
        ]
        result = _analyze(agents, closes)
        assert result.features["level"] == "EXTREME"
        assert result.features["state"] == "EXPANDING"
        assert any("extreme" in w.lower() for w in result.warnings)


class TestBehaviourRules:
    def test_high_volatility_alone_is_never_a_directional_signal(self, agents):
        closes = _sine(290, amp=1.2, seed=71) + [2656.0, 2644.0, 2656.0, 2644.5, 2655.5, 2645.0]
        result = _analyze(agents, closes)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0

    def test_features_carry_the_measurements(self, agents):
        closes = _sine(300, amp=1.2, seed=42)
        result = _analyze(agents, closes)
        f = result.features
        assert f["atr"] > 0
        assert 0 < f["atr_pct"] < 1.0  # gold M15/H1 sane range in percent
        assert f["bb_width_pct"] is not None
        assert f["atr_ratio"] is not None

    def test_constant_prices_are_low_volatility(self, agents):
        result = _analyze(agents, [2650.0] * 300)
        assert result.features["level"] == "LOW"

    def test_insufficient_candles(self, agents):
        result = _analyze(agents, _sine(29, amp=1.2, seed=43), seed=43)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality is DataQuality.INSUFFICIENT

    def test_short_history_degrades(self, agents):
        """Enough candles to run but not enough ranking history for a
        confident percentile (the agent says so instead of guessing)."""
        result = _analyze(agents, _sine(70, amp=1.2, seed=44), seed=44)
        assert result.data_quality is DataQuality.DEGRADED
        assert any("insufficient history" in w for w in result.warnings)

    def test_insufficient_history_warning_below_warmup(self, agents):
        result = _analyze(agents, _sine(35, amp=1.2, seed=45), seed=45)
        assert result.data_quality is DataQuality.DEGRADED
        assert result.features["level"] is not None  # still classified

    def test_full_history_is_ok(self, agents):
        result = _analyze(agents, _sine(300, amp=1.2, seed=42))
        assert result.data_quality is DataQuality.OK
