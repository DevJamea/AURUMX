"""Cross-cutting contract tests (Phase-2 §2/§13/§14/§15/§19).

Every agent, across a spread of market scenarios, must satisfy:

* the common result contract (identity, direction, bounded strength, reasons,
  JSON-safe features, data quality, decision-candle timestamps);
* determinism: same context in, byte-identical result out;
* purity: the context/snapshot is never mutated by analysis;
* wall-clock independence: signals derive from candle data only — an
  identical snapshot created at a different time yields the same signal;
* source hygiene: the agents/decision packages contain no MT5, network,
  wall-clock, randomness or file-system calls (backtest-safe by construction).
"""

from __future__ import annotations

import ast
import json
from datetime import timedelta
from pathlib import Path

import pytest

from app.agents import default_agents
from app.core.enums import AgentDirection, DataQuality, TimeFrame
from tests.unit.agents.scenarios import (
    REF_TIME,
    flat_closes,
    linear_trend_closes,
    make_context,
    make_series,
    sideways_closes,
    standard_triple,
)

SCENARIOS = {
    "uptrend": linear_trend_closes(300, slope=2.0, seed=11),
    "downtrend": linear_trend_closes(300, slope=-2.0, seed=12),
    "range": sideways_closes(300, period=16, amplitude=1.2, seed=71),
    "flat": flat_closes(300),
}


def _contexts():
    for label, closes in SCENARIOS.items():
        yield label, make_context(standard_triple(h1_closes=closes, h1_seed=hash(label) % 1000))


AGENT_NAMES = [a.name for a in default_agents()]


# ---------------------------------------------------------------------------
# contract shape
# ---------------------------------------------------------------------------


class TestResultContract:
    @pytest.mark.parametrize("label", list(SCENARIOS))
    def test_every_agent_result_satisfies_the_contract(self, agents, label):
        ctx = dict(_contexts())[label]
        for name, agent in agents.items():
            result = agent.analyze(ctx)
            assert result.agent == name
            assert isinstance(result.direction, AgentDirection)
            assert 0.0 <= result.signal_strength <= 1.0
            assert isinstance(result.reasons, list) and all(
                isinstance(r, str) for r in result.reasons
            )
            assert isinstance(result.warnings, list)
            assert isinstance(result.data_quality, DataQuality)
            assert result.snapshot_time == ctx.created_at
            # decision candle: the last CLOSED candle of the primary timeframe
            if result.primary_timeframe is not None:
                series = ctx.series(result.primary_timeframe)
                assert result.source_time == series.candles[-1].time
            # features must be JSON-safe for the decision journal
            json.dumps(result.features)
            json.dumps(result.model_dump(mode="json"))

    def test_field_named_signal_strength_not_confidence(self):
        """The word confidence/probability must not appear in the contract."""
        from app.agents.base import AgentResult

        fields = AgentResult.model_fields
        assert "signal_strength" in fields
        assert "confidence" not in fields
        assert "probability" not in fields


# ---------------------------------------------------------------------------
# determinism & purity
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("label", list(SCENARIOS))
    def test_same_context_identical_result(self, agents, label):
        ctx = dict(_contexts())[label]
        for agent in agents.values():
            first = agent.analyze(ctx)
            second = agent.analyze(ctx)
            assert first.model_dump() == second.model_dump(), agent.name

    def test_rebuilt_context_identical_result(self, agents):
        """A second snapshot/context built from the same candles (fresh
        feature computation, fresh objects) must produce the same output."""
        closes = SCENARIOS["uptrend"]
        ctx_a = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        ctx_b = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        for agent in agents.values():
            assert agent.analyze(ctx_a).model_dump() == agent.analyze(ctx_b).model_dump()

    def test_agents_share_instances_safely(self):
        """default_agents() instances are stateless — reusing one across
        different contexts cannot leak state between analyses."""
        trend = default_agents()[0]
        up = trend.analyze(make_context(standard_triple(h1_closes=SCENARIOS["uptrend"], h1_seed=11)))
        down = trend.analyze(make_context(standard_triple(h1_closes=SCENARIOS["downtrend"], h1_seed=12)))
        up_again = trend.analyze(make_context(standard_triple(h1_closes=SCENARIOS["uptrend"], h1_seed=11)))
        assert up.direction is AgentDirection.BUY
        assert down.direction is AgentDirection.SELL
        assert up_again.model_dump() == up.model_dump()


class TestPurity:
    @pytest.mark.parametrize("label", list(SCENARIOS))
    def test_analysis_does_not_mutate_the_context(self, agents, label):
        ctx = dict(_contexts())[label]
        before = ctx.snapshot.model_dump()
        series_before = {
            tf: [c.model_dump() for c in check.series.candles]
            for tf, check in ctx.snapshot.series.items()
        }
        for agent in agents.values():
            agent.analyze(ctx)
        assert ctx.snapshot.model_dump() == before
        for tf, check in ctx.snapshot.series.items():
            assert [c.model_dump() for c in check.series.candles] == series_before[tf]


class TestWallClockIndependence:
    def test_created_at_only_affects_provenance_not_signal(self, agents):
        """Same candles, snapshot taken 6 hours later: identical signals, only
        the snapshot_time provenance changes."""
        closes = SCENARIOS["uptrend"]
        early = make_context(standard_triple(h1_closes=closes, h1_seed=11), created_at=REF_TIME)
        late = make_context(
            standard_triple(h1_closes=closes, h1_seed=11), created_at=REF_TIME + timedelta(hours=6)
        )
        for name, agent in agents.items():
            a = agent.analyze(early)
            b = agent.analyze(late)
            assert a.direction is b.direction, name
            assert a.signal_strength == pytest.approx(b.signal_strength), name
            assert a.reasons == b.reasons, name
            # provenance does shift with the snapshot timestamp
            assert a.snapshot_time == early.created_at
            assert b.snapshot_time == late.created_at


# ---------------------------------------------------------------------------
# multi-timeframe roles
# ---------------------------------------------------------------------------


class TestTimeframeRoles:
    def test_roles_are_assigned(self):
        ctx = make_context(standard_triple(h1_closes=SCENARIOS["uptrend"], h1_seed=11))
        assert ctx.role_of(TimeFrame.H4) is not None
        assert ctx.role_of(TimeFrame.H1) is not None
        assert ctx.role_of(TimeFrame.M15) is not None

    def test_primary_for_role(self):
        ctx = make_context(standard_triple(h1_closes=SCENARIOS["uptrend"], h1_seed=11))
        assert ctx.primary_for_role.__self__ is ctx  # bound method sanity
        # H1 is the directional timeframe, H4 macro, M15 entry
        assert ctx.role_of(TimeFrame.H4).value == "MACRO"
        assert ctx.role_of(TimeFrame.H1).value == "STRUCTURE"
        assert ctx.role_of(TimeFrame.M15).value == "ENTRY"

    def test_agents_declare_primary_timeframes(self, agents):
        expectations = {
            "trend": TimeFrame.H1,
            "momentum": TimeFrame.H1,
            "structure": TimeFrame.H1,
            "liquidity": TimeFrame.M15,
            "volatility": TimeFrame.H1,
            "mean_reversion": TimeFrame.H1,
            "macro": None,
        }
        for name, agent in agents.items():
            assert agent.primary_timeframe is expectations[name], name

    def test_h4_context_changes_trend_result(self, agents):
        """H4 is a real input, not decoration: flipping H4 flips the H4-
        agreement evidence and with it the trend strength."""
        up = SCENARIOS["uptrend"]
        h4_up = make_context(standard_triple(h1_closes=up, h1_seed=11))
        series = standard_triple(h1_closes=up, h1_seed=11)
        series[TimeFrame.H4] = make_series(
            linear_trend_closes(300, slope=-2.0, seed=12), TimeFrame.H4, seed=12
        )
        h4_down = make_context(series)
        r_up = agents["trend"].analyze(h4_up)
        r_down = agents["trend"].analyze(h4_down)
        assert r_up.signal_strength > r_down.signal_strength
        assert r_up.direction is AgentDirection.BUY


# ---------------------------------------------------------------------------
# static source hygiene (backtest-safety by construction)
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES = {
    "MetaTrader5",
    "mt5",
    "requests",
    "urllib",
    "http",
    "socket",
    "httpx",
    "websockets",
    "random",
    "secrets",
}
FORBIDDEN_DOTTED_CALLS = {
    "datetime.now",
    "datetime.today",
    "datetime.utcnow",
    "time.time",
    "time.time_ns",
    "time.monotonic",
    "time.sleep",
    "random.random",
    "random.randint",
    "random.uniform",
    "random.choice",
    "random.gauss",
    "np.random.seed",
    "open",
    "eval",
    "exec",
}

AGENT_SOURCES = sorted(Path("app/agents").glob("*.py")) + sorted(Path("app/decision").glob("*.py"))


class TestSourceHygiene:
    @pytest.mark.parametrize("source", AGENT_SOURCES, ids=lambda p: p.name)
    def test_no_mt5_network_wallclock_randomness_or_io(self, source: Path):
        code = ast.parse(source.read_text())
        for node in ast.walk(code):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_MODULES, f"{source.name}: imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in FORBIDDEN_MODULES, f"{source.name}: imports from {node.module}"
                if root in ("datetime", "time"):
                    # only timedelta/timezone/UTC-style members, never now/today
                    for alias in node.names:
                        assert alias.name not in ("now", "today", "utcnow", "time"), (
                            f"{source.name}: from {node.module} import {alias.name}"
                        )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    dotted = ast.unparse(func)
                    assert dotted not in FORBIDDEN_DOTTED_CALLS, (
                        f"{source.name}: calls {dotted}()"
                    )
                elif isinstance(func, ast.Name):
                    assert func.id not in ("open", "eval", "exec", "random"), (
                        f"{source.name}: calls {func.id}()"
                    )

    def test_agents_never_import_fastapi_or_app_broker_layers(self):
        for source in AGENT_SOURCES:
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
                    allowed_prefixes = ("app.core", "app.agents", "app.decision", "app.market")
                    assert node.module.startswith(allowed_prefixes), (
                        f"{source.name}: agents must not depend on {node.module}"
                    )
