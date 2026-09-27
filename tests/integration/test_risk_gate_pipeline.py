"""End-to-end risk-gate pipeline (Phase 4):

FakeMT5 -> MT5Broker -> MarketDataService -> DecisionEngine -> TradeProposal
-> HardRiskGate -> RiskDecision.

Phase 4 still executes nothing: the gate's APPROVED is a data object, and
the pipeline ends there.  This test proves the two layers compose exactly
as the boundary diagram promises — and that the gate works on a REAL
engine-produced proposal (not just hand-built fixtures).
"""

from __future__ import annotations

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.enums import RiskAction
from app.decision import DecisionEngine, DecisionEngineConfig, InMemoryDecisionJournal
from app.market.data_service import MarketDataService
from app.risk import (
    AccountState,
    HardRiskGate,
    RiskEvent,
    RiskEventType,
    RiskGateConfig,
    RiskState,
)
from tests.conftest import REF_TIME
from tests.integration.test_agent_pipeline import inject_scenario
from tests.unit.agents.scenarios import gold_symbol_spec, linear_trend_closes


def make_service(fake) -> MarketDataService:
    broker = MT5Broker(mt5_module=fake)
    broker.connect()
    service = MarketDataService(
        broker, symbol="AUTO", candle_count=300, max_tick_age_seconds=60,
        candle_freshness_multiplier=3.0, clock=lambda: REF_TIME,
    )
    service.connect()
    return service


def build_gate(fake, events: list[RiskEvent] | None = None, **config) -> HardRiskGate:
    spec = gold_symbol_spec()
    return HardRiskGate(
        RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0, **config),
        symbol_specs={spec.name: spec},
        event_sink=(events.append if events is not None else None),
    )


class TestFullPipeline:
    def test_engine_proposal_is_approved_by_the_gate(self, fake_mt5):
        from app.core.enums import TimeFrame

        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up})
        service = make_service(fake_mt5)
        snapshot = service.get_snapshot()

        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        decision = engine.evaluate(
            snapshot, risk_state=RiskState(equity=10_000.0), now=REF_TIME
        )
        assert decision.proposal is not None  # the engine proposed a BUY

        account = AccountState(
            equity=10_000.0, balance=10_000.0, trade_allowed=True,
            open_positions_notional=0.0,
            spread_points=snapshot.tick.spread_points,
        )
        events: list[RiskEvent] = []
        gate = build_gate(fake_mt5, events)
        risk_decision = gate.evaluate(decision.proposal, RiskState(equity=10_000.0), account)

        assert risk_decision.action is RiskAction.APPROVED, risk_decision.reasons
        assert risk_decision.proposal_id == decision.proposal.decision_id
        # the gate's risk is INDEPENDENT math on the proposal geometry
        p = decision.proposal
        expected_risk = p.suggested_volume * (
            abs(p.entry_price - p.stop_loss) / 0.01 * 1.0
        )
        assert risk_decision.risk_amount == pytest.approx(expected_risk)
        assert risk_decision.risk_amount <= 50.0 + 1e-6  # within the 0.5% budget
        assert events[0].type is RiskEventType.RISK_APPROVED

    def test_halted_operator_blocks_engine_proposal(self, fake_mt5):
        from app.core.enums import TimeFrame

        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up})
        service = make_service(fake_mt5)
        engine = DecisionEngine(DecisionEngineConfig())
        decision = engine.evaluate(
            service.get_snapshot(), risk_state=RiskState(equity=10_000.0), now=REF_TIME
        )
        assert decision.proposal is not None

        events: list[RiskEvent] = []
        gate = build_gate(fake_mt5, events)
        account = AccountState(
            equity=10_000.0, trade_allowed=True, open_positions_notional=0.0,
            spread_points=20.0, kill_switch_active=True,
        )
        risk_decision = gate.evaluate(decision.proposal, RiskState(equity=10_000.0), account)

        assert risk_decision.action is RiskAction.EMERGENCY_STOP
        assert events[0].type is RiskEventType.KILL_SWITCH_ACTIVE

    def test_gate_independent_of_engine_state(self, fake_mt5):
        """The same proposal is judged on the ACCOUNT/RISK evidence alone:
        a healthy account approves, a breached daily loss rejects."""
        from app.core.enums import TimeFrame

        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up})
        service = make_service(fake_mt5)
        engine = DecisionEngine(DecisionEngineConfig())
        proposal = engine.evaluate(
            service.get_snapshot(), risk_state=RiskState(equity=10_000.0), now=REF_TIME
        ).proposal
        assert proposal is not None

        gate = build_gate(fake_mt5)
        account = AccountState(
            equity=10_000.0, trade_allowed=True,
            open_positions_notional=0.0, spread_points=20.0,
        )
        healthy = gate.evaluate(proposal, RiskState(equity=10_000.0), account)
        breached = gate.evaluate(
            proposal, RiskState(equity=10_000.0, daily_loss=250.0), account
        )
        assert healthy.action is RiskAction.APPROVED
        assert breached.action is RiskAction.REJECTED
        assert "daily_loss_limit" in breached.failed_checks

    def test_phase4_adds_no_execution_to_the_pipeline(self):
        """The integration proof of §28: the gate's output is data — there
        is no order, no broker call, nothing to send."""
        import inspect

        from app.risk import engine as risk_engine_module

        src = inspect.getsource(risk_engine_module)
        for forbidden in ("order_send", "positions_add", "orders_add", "modify", "close_position"):
            assert forbidden not in src
