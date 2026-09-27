"""RiskState tests (Phase-3 §15): caller-supplied, read-only by the engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.enums import DecisionAction
from app.decision import DecisionEngine, DecisionEngineConfig
from app.risk import RiskState
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_snapshot,
    standard_triple,
)


def buy_snapshot():
    up = linear_trend_closes(300, slope=2.0, seed=11)
    return make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)


@pytest.fixture()
def snap():
    return buy_snapshot()


class TestRiskStateModel:
    def test_defaults_are_unlimited(self):
        state = RiskState(equity=10_000.0)
        assert state.daily_loss == 0.0
        assert state.daily_loss_limit is None
        assert state.open_positions == 0
        assert state.pending_orders == 0
        assert state.consecutive_losses == 0
        assert state.active_setup_fingerprints == frozenset()
        assert state.has_equity

    def test_no_equity_flag(self):
        assert RiskState().has_equity is False

    def test_negative_equity_rejected(self):
        with pytest.raises(ValidationError):
            RiskState(equity=-1.0)

    def test_fingerprints_immutable(self):
        state = RiskState(equity=1.0, active_setup_fingerprints=["abc"])
        assert isinstance(state.active_setup_fingerprints, frozenset)


class TestEngineAbortsOnRiskState:
    """Risk-state violations ABORT (fail-closed) — they are not HOLDs."""

    def test_missing_equity_aborts(self, snap):
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap, risk_state=RiskState(), now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["missing_equity"]

    def test_daily_loss_limit_aborts(self, snap):
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, daily_loss=150.0, daily_loss_limit=100.0),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["daily_loss_limit"]

    def test_default_daily_limit_from_equity(self, snap):
        """Without an explicit limit, 2% of equity is the budget ($200 here)."""
        state = RiskState(equity=10_000.0, daily_loss=199.0)  # just inside budget
        assert state.daily_loss_limit is None
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap, risk_state=state, now=REF_TIME,
        )
        assert d.rejection_reasons != ["daily_loss_limit"]
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, daily_loss=201.0),
            now=REF_TIME,
        )
        assert d.rejection_reasons == ["daily_loss_limit"]

    def test_consecutive_losses_aborts(self, snap):
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, consecutive_losses=3),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["consecutive_losses"]

    def test_open_position_limit_aborts(self, snap):
        """max_open_positions default 1 -> a single open position blocks new proposals."""
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, open_positions=1),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["position_limit"]

    def test_position_limit_configurable(self, snap):
        d = DecisionEngine(DecisionEngineConfig(max_open_positions=2)).evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, open_positions=1),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.BUY

    def test_pending_order_limit_aborts(self, snap):
        d = DecisionEngine(DecisionEngineConfig()).evaluate(
            snap,
            risk_state=RiskState(equity=10_000.0, pending_orders=2),
            now=REF_TIME,
        )
        assert d.decision is DecisionAction.ABORT
        assert d.rejection_reasons == ["pending_order_limit"]


class TestEngineNeverMutatesRiskState:
    def test_state_unchanged_after_evaluation(self, snap):
        state = RiskState(equity=10_000.0, open_positions=1)
        before = state.model_dump()
        DecisionEngine(DecisionEngineConfig()).evaluate(snap, risk_state=state, now=REF_TIME)
        assert state.model_dump() == before
