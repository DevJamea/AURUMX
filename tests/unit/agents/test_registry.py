"""AgentRegistry tests: roster, ordering, enable/disable, failure isolation."""

from __future__ import annotations

import pytest

from app.agents import default_agents
from app.agents.base import AgentResult, BaseAgent
from app.agents.registry import AgentRegistry
from app.core.enums import AgentDirection, DataQuality, TimeFrame
from tests.unit.agents.scenarios import linear_trend_closes, make_context, standard_triple


class ExplodingAgent(BaseAgent):
    name = "exploding"
    description = "always raises — tests isolation"
    primary_timeframe = TimeFrame.H1
    min_candles = 1

    def analyze(self, context):
        raise RuntimeError("boom")


EXPECTED_ORDER = [
    "trend",
    "momentum",
    "structure",
    "liquidity",
    "volatility",
    "mean_reversion",
    "macro",
]


class TestRoster:
    def test_default_roster_order_is_stable(self):
        names = [a.name for a in default_agents()]
        assert names == EXPECTED_ORDER

    def test_run_all_returns_one_result_per_agent(self):
        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        results = AgentRegistry().run_all(ctx)
        assert [r.agent for r in results] == EXPECTED_ORDER

    def test_results_are_well_formed(self):
        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        for result in AgentRegistry().run_all(ctx):
            assert isinstance(result, AgentResult)
            assert result.signal_strength >= 0.0 and result.signal_strength <= 1.0
            assert result.reasons or result.agent == "macro"


class TestEnableDisable:
    def test_disable_skips_agent(self):
        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        registry = AgentRegistry()
        registry.enable("macro", False)
        assert not registry.is_enabled("macro")
        names = [r.agent for r in registry.run_all(ctx)]
        assert "macro" not in names
        assert len(names) == len(EXPECTED_ORDER) - 1

    def test_reenable_restores(self):
        registry = AgentRegistry()
        registry.enable("trend", False)
        registry.enable("trend", True)
        assert registry.is_enabled("trend")

    def test_run_single_agent_by_name(self):
        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        registry = AgentRegistry()
        result = registry.run("trend", ctx)
        assert result.agent == "trend"

    def test_run_unknown_agent_raises(self):
        with pytest.raises(KeyError):
            AgentRegistry().run("nonexistent", None)


class TestFailureIsolation:
    def test_raising_agent_becomes_invalid_neutral(self):
        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        registry = AgentRegistry(agents=[*default_agents(), ExplodingAgent()])
        results = registry.run_all(ctx)
        by_name = {r.agent: r for r in results}
        assert by_name["exploding"].direction is AgentDirection.NEUTRAL
        assert by_name["exploding"].signal_strength == 0.0
        assert by_name["exploding"].data_quality is DataQuality.INVALID
        assert any("failed" in w for w in by_name["exploding"].warnings)
        # everyone else unaffected:
        assert by_name["trend"].data_quality is DataQuality.OK

    def test_agent_error_is_emitted_on_the_bus(self):
        from app.core.events import get_event_bus

        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        bus = get_event_bus()
        bus.clear()
        registry = AgentRegistry(agents=[ExplodingAgent()])
        registry.run_all(ctx)
        events = bus.history("AGENT_ERROR")
        assert events, "AGENT_ERROR must be published"
        assert events[-1].payload["agent"] == "exploding"
        assert events[-1].payload["error"] == "boom"

    def test_agent_signal_events_emitted(self):
        from app.core.events import get_event_bus

        ctx = make_context(standard_triple(h1_closes=linear_trend_closes(300, slope=2.0), h1_seed=11))
        bus = get_event_bus()
        bus.clear()
        AgentRegistry().run_all(ctx)
        events = bus.history("AGENT_SIGNAL")
        assert len(events) == len(EXPECTED_ORDER)
        assert all("direction" in e.payload and "signal_strength" in e.payload for e in events)
