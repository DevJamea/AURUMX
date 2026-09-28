"""MacroAgent behavioral tests (Phase-2 §10 — interface only).

Categories: no provider (NEUTRAL/NO_DATA with provenance), provider with a
high-impact USD event in the blackout window (warning, never directional),
benign events, provider failure (fail-safe NEUTRAL), determinism.
"""

from __future__ import annotations

from datetime import timedelta

from app.agents.macro import BLACKOUT_WINDOW, MacroAgent, MacroEvent
from app.core.enums import AgentDirection, DataQuality
from tests.unit.agents.scenarios import REF_TIME, linear_trend_closes, make_context, standard_triple


class StaticProvider:
    """Deterministic in-memory provider (the only kind Phase 2 allows)."""

    def __init__(self, events):
        self._events = list(events)

    def events_between(self, start, end):
        return [e for e in self._events if start <= e.time <= end]


class ExplodingProvider:
    def events_between(self, start, end):
        raise RuntimeError("calendar feed down")


def _ctx():
    return make_context(standard_triple(h1_closes=linear_trend_closes(120, slope=1.0), h1_seed=17))


class TestNoProvider:
    def test_no_provider_is_neutral_no_data(self):
        result = MacroAgent().analyze(_ctx())
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0
        assert result.data_quality is DataQuality.NO_DATA
        assert any("no macro data provider" in r for r in result.reasons)

    def test_provenance_is_explicit(self):
        result = MacroAgent().analyze(_ctx())
        assert result.features["provenance"].startswith("provider=none")
        assert "fabricated" in result.features["provenance"]

    def test_blackout_false_without_provider(self):
        result = MacroAgent().analyze(_ctx())
        assert result.features["blackout"] is False


class TestWithProvider:
    def test_high_impact_usd_event_in_window_warns_but_stays_neutral(self):
        provider = StaticProvider(
            [MacroEvent(name="FOMC Rate Decision", time=REF_TIME + timedelta(minutes=10), impact="HIGH")]
        )
        result = MacroAgent(provider).analyze(_ctx())
        assert result.direction is AgentDirection.NEUTRAL  # never directional
        assert result.signal_strength == 0.0
        assert result.features["blackout"] is True
        assert any("FOMC" in w for w in result.warnings)
        assert any("blocking new entries" in w for w in result.warnings)

    def test_low_impact_event_does_not_blackout(self):
        provider = StaticProvider(
            [MacroEvent(name="EU Speech", time=REF_TIME, impact="LOW", currency="EUR")]
        )
        result = MacroAgent(provider).analyze(_ctx())
        assert result.features["blackout"] is False
        assert not result.warnings

    def test_non_usd_high_impact_does_not_blackout(self):
        provider = StaticProvider(
            [MacroEvent(name="EUR CPI", time=REF_TIME, impact="HIGH", currency="EUR")]
        )
        result = MacroAgent(provider).analyze(_ctx())
        assert result.features["blackout"] is False

    def test_event_just_outside_window_is_ignored(self):
        provider = StaticProvider(
            [MacroEvent(name="NFP", time=REF_TIME + BLACKOUT_WINDOW + timedelta(minutes=1), impact="HIGH")]
        )
        result = MacroAgent(provider).analyze(_ctx())
        assert result.features["blackout"] is False

    def test_provider_failure_fails_safe(self):
        result = MacroAgent(ExplodingProvider()).analyze(_ctx())
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality is DataQuality.DEGRADED
        assert any("provider failed" in w for w in result.warnings)

    def test_provider_provenance_names_the_source(self):
        provider = StaticProvider([])
        result = MacroAgent(provider).analyze(_ctx())
        assert "StaticProvider" in result.features["provenance"]
        assert result.data_quality is DataQuality.OK


class TestDeterminism:
    def test_same_provider_same_result(self):
        provider = StaticProvider(
            [MacroEvent(name="CPI", time=REF_TIME - timedelta(minutes=5), impact="HIGH")]
        )
        agent = MacroAgent(provider)
        first = agent.analyze(_ctx())
        second = agent.analyze(_ctx())
        assert first.model_dump() == second.model_dump()
