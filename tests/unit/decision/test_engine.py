"""DecisionEngine gate tests — Phase-3 scenarios A–J plus edge gates.

Every scenario pins an exact action and the exact rejection reason, because
operators must be able to grep the journal for these strings.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.enums import AgentDirection, DecisionAction, SessionState, TimeFrame
from app.decision import DecisionEngine, DecisionEngineConfig
from app.risk import RiskState
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_snapshot,
    make_tick,
    standard_triple,
)
from tests.unit.decision.conftest import FixedAgent


def trend_up_snapshot():
    up = linear_trend_closes(300, slope=2.0, seed=11)
    return make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)


def trend_down_snapshot():
    down = linear_trend_closes(300, slope=-2.0, seed=12)
    return make_snapshot(standard_triple(h1_closes=down, h1_seed=12), created_at=REF_TIME)


class TestScenarioABuy:
    """A: strong BUY — H4/H1/M15 bullish, normal spread/vol, RR ok, risk available."""

    def test_buy_with_valid_proposal(self, engine, risk_state):
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        p = d.proposal
        assert p is not None
        assert p.direction is AgentDirection.BUY
        # pinned values from the smoke run (regression pins)
        assert p.entry_price == pytest.approx(2650.20)  # validated ASK, not close
        assert p.stop_loss == pytest.approx(2645.49, abs=0.02)
        assert p.take_profit == pytest.approx(2659.61, abs=0.02)
        assert p.risk_reward == pytest.approx(2.0, abs=0.01)
        assert p.suggested_volume == pytest.approx(0.10)
        assert p.expires_at - p.created_at == timedelta(minutes=15)
        assert d.conflict_score == pytest.approx(0.0)
        assert d.alignment_score == pytest.approx(1.0)
        assert "trend" in d.supporting_agents

    def test_proposal_rejects_when_expired(self, engine, risk_state):
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.proposal is not None
        assert d.proposal.status(REF_TIME + timedelta(minutes=20)) == "EXPIRED"


class TestScenarioBSell:
    def test_sell_with_valid_proposal(self, engine, risk_state):
        d = engine.evaluate(trend_down_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.SELL
        p = d.proposal
        assert p is not None
        assert p.direction is AgentDirection.SELL
        assert p.entry_price == pytest.approx(2650.0)  # validated BID
        assert p.stop_loss > p.entry_price
        assert p.take_profit < p.entry_price
        assert p.suggested_volume == pytest.approx(0.10)


class TestScenarioCTimeframeConflict:
    """C: H4/H1 bullish + M15 strongly bearish -> HOLD unless configured otherwise."""

    def test_hold_on_timeframe_conflict(self, engine, risk_state, conflict_snapshot):
        d = engine.evaluate(conflict_snapshot, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["timeframe_conflict"]
        assert d.proposal is None
        assert d.alignment_score == pytest.approx(0.75)

    def test_lower_threshold_explicitly_permits(self, risk_state, conflict_snapshot):
        engine = DecisionEngine(DecisionEngineConfig(minimum_timeframe_alignment=0.70))
        d = engine.evaluate(conflict_snapshot, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        assert d.proposal is not None


class TestScenarioDSpread:
    def test_excessive_spread_aborts(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, spread_points=200.0,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["spread_too_high"]
        assert d.proposal is None

    def test_acceptable_spread_trades(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, spread_points=20.0,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.BUY

    def test_spread_limit_is_configurable(self, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, spread_points=60.0,
        )
        # default max_spread_points 50 -> ABORT
        d = DecisionEngine(DecisionEngineConfig()).evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.rejection_reasons == ["spread_too_high"]
        # raised limit -> BUY
        d = DecisionEngine(DecisionEngineConfig(max_spread_points=100.0)).evaluate(
            snap, risk_state=risk_state, now=REF_TIME,
        )
        assert d.decision is DecisionAction.BUY


class TestScenarioEInvalidTick:
    def test_invalid_tick_aborts_before_any_analysis(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, tick_valid=False,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["invalid_tick"]

    def test_stale_tick_aborts(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, tick_fresh=False,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["stale_tick"]

    def test_stale_candles_abort(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, fresh=False,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons[0].startswith("stale_candles")

    def test_invalid_series_aborts(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, valid=False,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons[0].startswith("invalid_series")


class TestScenarioFSessions:
    def test_market_closed_holds(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, session_state=SessionState.CLOSED,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["market_closed"]
        assert d.proposal is None

    def test_market_open_trades(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, session_state=SessionState.OPEN,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.BUY


class TestScenarioGUnknownState:
    def test_unknown_session_aborts_never_assumed_open(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, session_state=SessionState.UNKNOWN,
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["session_unknown"]


class TestScenarioHInsufficientRR:
    def test_rr_below_minimum_holds(self, risk_state):
        engine = DecisionEngine(DecisionEngineConfig(minimum_rr=2.5, target_rr=2.0))
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["insufficient_rr"]
        assert d.proposal is None
        assert any("below minimum" in r for r in d.reasons)

    def test_structure_target_rr_can_fail_rr_gate(self, risk_state):
        """With tp_method='structure' the RR is market-determined, so the
        minimum_rr gate is a real constraint (not tautological)."""
        engine = DecisionEngine(DecisionEngineConfig(tp_method="structure", minimum_rr=3.0))
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        # linear trend has no opposing swing -> rr_target fallback (RR 2.0 < 3.0)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["insufficient_rr"]


class TestScenarioIRiskStateGates:
    """Also covered in tests/unit/risk/test_state.py; pinned here as scenarios."""

    def test_daily_loss_limit_aborts(self, engine):
        d = engine.evaluate(
            trend_up_snapshot(),
            risk_state=RiskState(equity=10_000.0, daily_loss=150.0, daily_loss_limit=100.0),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["daily_loss_limit"]


class TestScenarioJPositionLimits:
    def test_open_position_limit_blocks_new_proposal(self, engine):
        d = engine.evaluate(
            trend_up_snapshot(),
            risk_state=RiskState(equity=10_000.0, open_positions=1),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["position_limit"]
        assert d.proposal is None


class TestEdgeGateFailures:
    """Every edge gate ABORTs (data unavailable = no decision), except those
    that are genuinely market-condition HOLDs."""

    def test_unknown_symbol_aborts(self, engine, risk_state):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME,
        )
        snap = snap.model_copy(update={"symbol": None})
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["missing_symbol"]

    def test_no_candles_aborts(self, engine, risk_state):
        snap = make_snapshot({}, created_at=REF_TIME)
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["no_candle_data"]

    def test_missing_h1_aborts(self, engine, risk_state):
        series = standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11)
        del series[TimeFrame.H1]
        d = engine.evaluate(make_snapshot(series, created_at=REF_TIME), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["missing_primary_timeframe_h1"]

    def test_missing_m15_holds(self, engine, risk_state):
        series = standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11)
        del series[TimeFrame.M15]
        d = engine.evaluate(make_snapshot(series, created_at=REF_TIME), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["missing_entry_timeframe"]

    def test_missing_h4_proceeds_with_renormalized_alignment(self, engine, risk_state):
        """Documented policy: H4 context is valuable but not required — H1+M15
        can still agree (score renormalized over present timeframes)."""
        series = standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11)
        del series[TimeFrame.H4]
        d = engine.evaluate(make_snapshot(series, created_at=REF_TIME), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        assert d.alignment_score == pytest.approx(1.0)
        assert d.proposal is not None

    def test_insufficient_h1_candles_holds(self, engine, risk_state):
        short = linear_trend_closes(50, slope=2.0, seed=11)
        series = standard_triple(h1_closes=short, h1_seed=11)
        d = engine.evaluate(make_snapshot(series, created_at=REF_TIME), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["insufficient_candles"]

    def test_range_market_holds_no_edge(self, engine, risk_state, range_snapshot):
        d = engine.evaluate(range_snapshot, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert "no_edge" in d.rejection_reasons or "below_threshold" in d.rejection_reasons


class TestGateOrder:
    """Data-quality gates run BEFORE risk gates, which run BEFORE analysis:
    a snapshot with BOTH an invalid tick and a breached daily-loss limit
    reports the data problem (fail-closed on inputs first)."""

    def test_invalid_tick_wins_over_risk_state(self, engine):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, tick_valid=False,
        )
        d = engine.evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, daily_loss=999.0, daily_loss_limit=1.0),
            now=REF_TIME,
        )
        assert d.rejection_reasons == ["invalid_tick"]

    def test_closed_market_wins_over_risk_state(self, engine):
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, session_state=SessionState.CLOSED,
        )
        d = engine.evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, open_positions=5),
            now=REF_TIME,
        )
        assert d.rejection_reasons == ["market_closed"]

    def test_risk_state_wins_over_analysis(self, engine):
        """Blocked risk state prevents any agent work."""
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME,
        )
        d = engine.evaluate(
            snap, risk_state=RiskState(equity=10_000.0, open_positions=9), now=REF_TIME,
        )
        assert d.rejection_reasons == ["position_limit"]
        assert d.agent_results == []  # agents never ran


class TestStubAgentConflicts:
    """Exact control over agent disagreement to test the conflict gate in
    isolation. Trend BUY + structure BUY + liquidity BUY vs momentum SELL
    (requirement §5's example shape)."""

    def stub_engine(self, **config) -> DecisionEngine:
        agents = [
            FixedAgent("trend", AgentDirection.BUY, 0.9),
            FixedAgent("structure", AgentDirection.BUY, 0.9),
            FixedAgent("liquidity", AgentDirection.BUY, 0.5),
            FixedAgent("momentum", AgentDirection.SELL, 0.9),
        ]
        return DecisionEngine(DecisionEngineConfig(**config), agents=agents)

    def test_disagreement_preserved_on_buy(self, risk_state):
        d = self.stub_engine().evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        assert "momentum" in d.opposing_agents
        assert {"trend", "structure"} <= set(d.supporting_agents)
        assert d.conflict_score > 0.0

    def test_conflict_above_tolerance_holds(self, risk_state):
        d = self.stub_engine(max_conflict=0.20).evaluate(
            trend_up_snapshot(), risk_state=risk_state, now=REF_TIME,
        )
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["conflict_exceeds_tolerance"]
        assert d.proposal is None

    def test_weak_signal_holds(self, risk_state):
        """Enough total evidence to clear the threshold gate, but the leading
        agent's own strength (0.44) is below minimum_signal_strength (0.45)."""
        agents = [
            FixedAgent("trend", AgentDirection.BUY, 0.44),
            FixedAgent("structure", AgentDirection.BUY, 0.44),
            FixedAgent("liquidity", AgentDirection.BUY, 0.44),
        ]
        engine = DecisionEngine(DecisionEngineConfig(), agents=agents)
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["weak_signal"]

    def test_below_threshold_holds(self, risk_state):
        """Synthesis reaches a BUY (>= 1.0) but the weighted score stays
        under a raised buy_threshold."""
        agents = [
            FixedAgent("trend", AgentDirection.BUY, 0.5),
            FixedAgent("structure", AgentDirection.BUY, 0.5),
        ]
        engine = DecisionEngine(
            DecisionEngineConfig(buy_threshold=1.5), agents=agents,
        )
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["below_threshold"]

    def test_no_edge_when_synthesis_itself_holds(self, risk_state):
        agents = [FixedAgent("trend", AgentDirection.BUY, 0.3)]  # 0.375 < 1.0
        engine = DecisionEngine(DecisionEngineConfig(), agents=agents)
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["no_edge"]


class TestAntiOvertrading:
    """§23: identical setup fingerprints must not produce unlimited proposals."""

    def test_duplicate_fingerprint_holds(self, engine, risk_state):
        snap = trend_up_snapshot()
        first = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert first.decision is DecisionAction.BUY
        fp = first.proposal.fingerprint

        blocked = RiskState(equity=10_000.0, active_setup_fingerprints=[fp])
        second = engine.evaluate(snap, risk_state=blocked, now=REF_TIME + timedelta(minutes=1))
        assert second.decision is DecisionAction.HOLD
        assert second.rejection_reasons == ["duplicate_setup"]
        assert second.proposal is None

    def test_new_setup_is_allowed(self, engine, risk_state):
        first = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        second = engine.evaluate(trend_down_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert first.proposal.fingerprint != second.proposal.fingerprint
        assert second.decision is DecisionAction.SELL

    def test_fingerprint_stable_for_same_candle(self, engine, risk_state):
        snap = trend_up_snapshot()
        d1 = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        d2 = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME + timedelta(minutes=5))
        assert d1.proposal.fingerprint == d2.proposal.fingerprint
        # decision_id, however, is per-evaluation (time-unique for the journal)
        assert d1.decision_id != d2.decision_id

    def test_fingerprint_changes_with_direction(self, engine, risk_state):
        d1 = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        d2 = engine.evaluate(trend_down_snapshot(), risk_state=risk_state, now=REF_TIME)
        assert d1.proposal.fingerprint != d2.proposal.fingerprint


class TestEntryPriceValidation:
    def test_buy_entry_is_ask_not_close(self, engine, risk_state):
        d = engine.evaluate(trend_up_snapshot(), risk_state=risk_state, now=REF_TIME)
        tick = trend_up_snapshot().tick.tick
        assert d.proposal.entry_price == pytest.approx(tick.ask)
        # the entry must NOT be a historical close (2652.8 = last close here)
        assert d.proposal.entry_price != pytest.approx(2652.8)

    def test_sell_entry_is_bid(self, engine, risk_state):
        d = engine.evaluate(trend_down_snapshot(), risk_state=risk_state, now=REF_TIME)
        tick = trend_down_snapshot().tick.tick
        assert d.proposal.entry_price == pytest.approx(tick.bid)

    def test_zero_price_tick_aborts_at_entry_gate(self, engine, risk_state):
        """A hand-marked-valid tick with a zero price still cannot produce a
        proposal — the engine re-checks entry-price sanity itself."""
        snap = make_snapshot(
            standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
            created_at=REF_TIME, tick=make_tick(bid=0.0, ask=0.0),
        )
        d = engine.evaluate(snap, risk_state=risk_state, now=REF_TIME)
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["invalid_entry_price"]
        assert d.proposal is None
