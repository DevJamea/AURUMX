"""Determinism and no-look-ahead tests (Phase-3 §20/§21).

No-look-ahead contract: ``decision(prefix) == decision(full series, evaluated
at the prefix's time)``. Future candles exist in the snapshot but can never
influence a decision finalized at an earlier time — the engine slices every
series to candles CLOSED before ``now`` before any analysis runs.
"""

from __future__ import annotations

from datetime import timedelta

from app.core.enums import DecisionAction, TimeFrame
from app.decision import DecisionEngine, DecisionEngineConfig
from app.risk import RiskState
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_series,
    make_snapshot,
    standard_triple,
)

STATE = RiskState(equity=10_000.0)


N_FUTURE = 6  # extra H1 bars that close after REF_TIME


def full_and_prefix_series():
    """(full, prefix) series maps built independently.

    The full series is anchored 6 hours later, so its windows end after
    REF_TIME and contain bars that close in the "future". Both series share
    the same close path and grid, so the bars closed by REF_TIME are
    value-identical — the only difference is the future tail."""
    base = linear_trend_closes(320, slope=2.0, seed=11)
    later = REF_TIME + timedelta(hours=N_FUTURE)
    full = {
        TimeFrame.M15: make_series(base, TimeFrame.M15, now=later, seed=12),
        TimeFrame.H1: make_series(base, TimeFrame.H1, now=later, seed=11),
        TimeFrame.H4: make_series(base, TimeFrame.H4, now=later, seed=11),
    }
    # the prefix sees exactly the bars that had CLOSED by REF_TIME on each TF
    # (the shared close path is simply cut per-timeframe; wick rng is
    # position-based so the surviving bars stay value-identical)
    prefix = {}
    for tf, series in full.items():
        closed = [c for c in series.candles
                  if c.time + timedelta(minutes=tf.minutes) <= REF_TIME]
        prefix[tf] = make_series(
            base[: len(closed)], tf, seed=12 if tf is TimeFrame.M15 else 11,
        )
    return full, prefix


class TestNoLookAhead:
    def test_future_candles_cannot_change_a_finalized_decision(self):
        engine = DecisionEngine(DecisionEngineConfig())
        full, prefix = full_and_prefix_series()

        # sanity: the full series really carries future bars on every TF
        for tf in (TimeFrame.M15, TimeFrame.H1, TimeFrame.H4):
            future = [c for c in full[tf].candles
                      if c.time + timedelta(minutes=tf.minutes) > REF_TIME]
            assert future, f"expected future bars on {tf}"
        # and the prefix's bars are a subset (same values) of the full's
        assert prefix[TimeFrame.H1].candles == full[TimeFrame.H1].candles[: len(prefix[TimeFrame.H1].candles)]

        d_full = engine.evaluate(make_snapshot(full, created_at=REF_TIME), risk_state=STATE, now=REF_TIME)
        d_prefix = engine.evaluate(make_snapshot(prefix, created_at=REF_TIME), risk_state=STATE, now=REF_TIME)

        # identical decision INCLUDING ids (same now, same fingerprint)
        assert d_full.decision is d_prefix.decision is DecisionAction.BUY
        assert d_full.decision_id == d_prefix.decision_id
        assert d_full.proposal.model_dump() == d_prefix.proposal.model_dump()
        assert d_full.supporting_agents == d_prefix.supporting_agents

    def test_forming_candle_is_excluded(self):
        """A candle whose close time is after `now` is never analysed — the
        sliced snapshot equals the snapshot that never had the future bars."""
        engine = DecisionEngine(DecisionEngineConfig())
        full, prefix = full_and_prefix_series()
        sliced = engine._slice_to_now(make_snapshot(full, created_at=REF_TIME), REF_TIME)
        for tf, check in sliced.series.items():
            for candle in check.series.candles:
                assert candle.time + timedelta(minutes=tf.minutes) <= REF_TIME
            # the surviving bars are value-identical to the prefix snapshot
            assert check.series.candles == prefix[tf].candles

    def test_slicing_keeps_enough_depth(self):
        engine = DecisionEngine(DecisionEngineConfig())
        full, _ = full_and_prefix_series()
        sliced = engine._slice_to_now(make_snapshot(full, created_at=REF_TIME), REF_TIME)
        assert len(sliced.series[TimeFrame.H1].series.candles) >= 60


class TestDeterminism:
    def test_repeated_evaluation_is_identical(self):
        engine = DecisionEngine(DecisionEngineConfig())
        up = linear_trend_closes(300, slope=2.0, seed=11)
        snap = make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)
        first = engine.evaluate(snap, risk_state=STATE, now=REF_TIME)
        for _ in range(3):
            again = engine.evaluate(snap, risk_state=STATE, now=REF_TIME)
            assert again.model_dump() == first.model_dump()

    def test_rebuilt_engine_is_identical(self):
        """No hidden state: a fresh engine with the same config reproduces
        the decision exactly."""
        up = linear_trend_closes(300, slope=2.0, seed=11)
        snap = make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)
        first = DecisionEngine(DecisionEngineConfig()).evaluate(snap, risk_state=STATE, now=REF_TIME)
        rebuilt = DecisionEngine(DecisionEngineConfig()).evaluate(snap, risk_state=STATE, now=REF_TIME)
        assert rebuilt.model_dump() == first.model_dump()

    def test_agent_order_does_not_matter(self):
        """Agent roster order must not affect the decision (synthesis is
        order-independent)."""
        from app.agents import default_agents

        up = linear_trend_closes(300, slope=2.0, seed=11)
        snap = make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)
        agents = default_agents()
        forward = DecisionEngine(DecisionEngineConfig(), agents=list(agents)).evaluate(
            snap, risk_state=STATE, now=REF_TIME,
        )
        backward = DecisionEngine(DecisionEngineConfig(), agents=list(agents)[::-1]).evaluate(
            snap, risk_state=STATE, now=REF_TIME,
        )
        assert forward.model_dump() == backward.model_dump()

    def test_same_now_same_decision_id(self):
        engine = DecisionEngine(DecisionEngineConfig())
        up = linear_trend_closes(300, slope=2.0, seed=11)
        snap = make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)
        a = engine.evaluate(snap, risk_state=STATE, now=REF_TIME)
        b = engine.evaluate(snap, risk_state=STATE, now=REF_TIME)
        assert a.decision_id == b.decision_id

    def test_no_wall_clock_in_decision_path(self):
        """`now` is always explicit: the engine source never CALLS wall-clock
        or RNG functions (AST-level check, immune to docstring mentions)."""
        import ast
        import inspect

        from app.decision import engine as engine_module

        tree = ast.parse(inspect.getsource(engine_module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                assert name not in {"now", "time", "monotonic", "random", "randint", "uuid4"}, (
                    f"non-deterministic call: {name}"
                )
