"""End-to-end decision pipeline (Phase 3):

FakeMT5 -> MT5Broker -> MarketDataService (validation) -> DecisionEngine
(regime -> agents -> synthesis -> gates -> risk sizing) -> TradeProposal
-> DecisionJournal.

Phase 3 is proposal-only: nothing in this pipeline can place an order.
"""

from __future__ import annotations

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.enums import AgentDirection, DecisionAction, TimeFrame
from app.decision import (
    DecisionEngine,
    DecisionEngineConfig,
    InMemoryDecisionJournal,
)
from app.market.data_service import MarketDataService
from app.risk import RiskState
from tests.conftest import REF_TIME
from tests.integration.test_agent_pipeline import inject_scenario
from tests.unit.agents.scenarios import linear_trend_closes

STATE = RiskState(equity=10_000.0)


def make_service(fake, **kwargs) -> MarketDataService:
    broker = MT5Broker(mt5_module=fake)
    broker.connect()
    defaults = dict(
        symbol="AUTO",
        candle_count=300,
        max_tick_age_seconds=60,
        candle_freshness_multiplier=3.0,
        clock=lambda: REF_TIME,
    )
    defaults.update(kwargs)
    service = MarketDataService(broker, **defaults)
    service.connect()
    return service


class TestFullPipeline:
    def test_trending_market_produces_journalled_buy(self, fake_mt5):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {
            TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up,
        })
        service = make_service(fake_mt5)
        snapshot = service.get_snapshot()
        assert snapshot.trading_data_ok  # Phase-1/2 layers are happy

        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        decision = engine.evaluate(snapshot, risk_state=STATE, now=REF_TIME)

        assert decision.decision is DecisionAction.BUY
        proposal = decision.proposal
        assert proposal is not None
        assert proposal.direction is AgentDirection.BUY
        assert proposal.entry_price == pytest.approx(snapshot.ask)
        assert proposal.stop_loss < proposal.entry_price < proposal.take_profit
        assert proposal.suggested_volume > 0

        # the journal captured the full story
        record = journal.by_decision_id(decision.decision_id)
        assert record is not None
        assert record.decision is DecisionAction.BUY
        assert record.agent_results
        assert record.proposal["decision_id"] == decision.decision_id

    def test_downtrend_produces_sell(self, fake_mt5):
        down = linear_trend_closes(300, slope=-2.0, seed=12)
        inject_scenario(fake_mt5, {
            TimeFrame.M15: down, TimeFrame.H1: down, TimeFrame.H4: down,
        })
        service = make_service(fake_mt5)
        engine = DecisionEngine(DecisionEngineConfig())
        decision = engine.evaluate(service.get_snapshot(), risk_state=STATE, now=REF_TIME)
        assert decision.decision is DecisionAction.SELL
        assert decision.proposal.entry_price == pytest.approx(service.get_snapshot().bid)

    def test_pipeline_is_deterministic(self, fake_mt5):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {
            TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up,
        })
        first_service = make_service(fake_mt5)
        d1 = DecisionEngine(DecisionEngineConfig()).evaluate(
            first_service.get_snapshot(), risk_state=STATE, now=REF_TIME,
        )

        # a completely rebuilt pipeline reproduces the decision byte-for-byte
        inject_scenario(fake_mt5, {
            TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up,
        })
        second_service = make_service(fake_mt5)
        d2 = DecisionEngine(DecisionEngineConfig()).evaluate(
            second_service.get_snapshot(), risk_state=STATE, now=REF_TIME,
        )
        assert d1.model_dump() == d2.model_dump()

    def test_blocked_risk_state_yields_no_proposal(self, fake_mt5):
        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {
            TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up,
        })
        service = make_service(fake_mt5)
        engine = DecisionEngine(DecisionEngineConfig())
        decision = engine.evaluate(
            service.get_snapshot(),
            risk_state=RiskState(equity=10_000.0, open_positions=1),
            now=REF_TIME,
        )
        assert decision.decision is DecisionAction.ABORT
        assert decision.proposal is None


class TestPhase3IsProposalOnly:
    """§1: the decision layer is completely separated from execution."""

    def test_no_order_send_anywhere_in_decision_layer(self):
        import inspect

        import app.decision.engine as engine_module
        import app.risk.sizing as sizing_module

        for module in (engine_module, sizing_module):
            src = inspect.getsource(module)
            assert "order_send" not in src
            assert "positions_get" not in src
            assert "orders_total" not in src

    def test_engine_has_no_broker_dependency(self):
        import ast
        import inspect

        from app.decision import engine as engine_module

        tree = ast.parse(inspect.getsource(engine_module))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        for forbidden in ("app.brokers", "app.market.data_service", "MetaTrader5", "requests", "httpx"):
            assert not any(imp.startswith(forbidden) for imp in imports), (
                f"decision layer must not import {forbidden}"
            )

    def test_evaluation_returns_data_only(self, fake_mt5):
        """evaluate() returns a Decision — a pydantic model. No handles, no
        callbacks, no side effects beyond the (optional) journal."""
        up = linear_trend_closes(300, slope=2.0, seed=11)
        inject_scenario(fake_mt5, {
            TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up,
        })
        service = make_service(fake_mt5)
        engine = DecisionEngine(DecisionEngineConfig())
        snapshot = service.get_snapshot()
        before = snapshot.model_dump()
        decision = engine.evaluate(snapshot, risk_state=STATE, now=REF_TIME)
        assert snapshot.model_dump() == before  # snapshot untouched
        assert decision.proposal is not None
        # the proposal is inert: it has no methods that could reach a broker
        assert not hasattr(decision.proposal, "send")
        assert not hasattr(decision.proposal, "execute")
        assert not hasattr(decision.proposal, "place")
