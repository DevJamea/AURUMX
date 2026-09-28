"""EngineRuntime behavior tests: evaluate -> gate -> execute -> reconcile,
with every safety property of the composition (5E)."""

from __future__ import annotations

import pytest

from app.control import EngineRuntime
from app.control.errors import ExecutionRefused
from app.core.enums import TradingMode
from tests.control.conftest import CONTROL_REF, make_runtime_config
from tests.fakes.mt5_fake import default_account


def inject_trend(fake_mt5):
    """A clean uptrend on all timeframes -> the engine proposes BUY."""
    from app.core.enums import TimeFrame
    from tests.integration.test_agent_pipeline import inject_scenario
    from tests.unit.agents.scenarios import linear_trend_closes

    up = linear_trend_closes(300, slope=2.0, seed=11)
    inject_scenario(fake_mt5, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up})
    fake_mt5.set_tick("XAUUSD", epoch=CONTROL_REF.timestamp() - 2, bid=2650.0, ask=2650.20)


class TestEvaluateCycle:
    def test_cycle_requires_started_engine(self, runtime):
        from app.control.errors import EngineNotStartedError

        with pytest.raises(EngineNotStartedError):
            runtime.evaluate_cycle()

    def test_cycle_journals_decision_and_risk(self, runtime, fake_mt5):
        inject_trend(fake_mt5)
        runtime.control.start()
        result = runtime.evaluate_cycle()
        assert result["decision"]["decision"] == "BUY"
        assert result["risk"]["action"] == "APPROVED"
        assert len(runtime.decision_journal) == 1
        assert runtime._last_proposal is not None

    def test_cycle_never_executes(self, runtime, fake_mt5):
        """Evaluating is analysis only — no order leaves the system."""
        inject_trend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()
        assert fake_mt5.order_sends == []
        assert fake_mt5.order_checks == []
        assert len(runtime.execution_journal) == 0


class TestExecuteApproved:
    def test_dry_run_execution_via_runtime(self, runtime, fake_mt5):
        inject_trend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()
        result = runtime.execute_approved()
        assert result.status.value == "DRY_RUN"
        assert fake_mt5.order_sends == []  # DRY_RUN: provably nothing sent
        assert len(runtime.execution_journal) == 1

    def test_demo_execution_via_runtime(self, fake_mt5, control_config):
        from app.brokers.mt5 import MT5Broker

        config = make_runtime_config(
            dry_run=False, real_trading_confirmed="I ACCEPT REAL TRADING RISK"
        )
        fake_mt5.execution_enabled = True
        broker = MT5Broker(mt5_module=fake_mt5)
        runtime = EngineRuntime(config, broker, clock=lambda: CONTROL_REF)
        runtime.connect()
        inject_trend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()
        result = runtime.execute_approved()
        assert result.status.value == "FILLED"
        assert result.stage.value == "VERIFIED"
        assert len(fake_mt5.order_sends) == 1

        # the whole chain reconciles cleanly
        report = runtime.reconcile()
        assert report.clean

    def test_halt_blocks_execution_even_when_approved(self, runtime, fake_mt5):
        """Halt engaged AFTER approval still refuses execution (runtime
        defense-in-depth on top of the gate)."""
        inject_trend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()
        runtime.control.engage_emergency_stop("post-approval halt")
        with pytest.raises(ExecutionRefused, match="halt"):
            runtime.execute_approved()
        assert fake_mt5.order_sends == []

    def test_no_approved_proposal_refuses(self, runtime, fake_mt5):
        runtime.control.start()
        with pytest.raises(ExecutionRefused, match="no proposal"):
            runtime.execute_approved()

    def test_unapproved_outcome_refuses(self, runtime, fake_mt5):
        """Whatever a cycle decides, only a gate-APPROVED pair executes:
        a flat market (HOLD, no proposal) must refuse too."""
        from app.core.enums import TimeFrame
        from tests.integration.test_agent_pipeline import inject_scenario
        from tests.unit.agents.scenarios import flat_closes

        flat = flat_closes(300)
        inject_scenario(fake_mt5, {TimeFrame.M15: flat, TimeFrame.H1: flat, TimeFrame.H4: flat})
        runtime.control.start()
        outcome = runtime.evaluate_cycle()
        risk = outcome["risk"]
        if risk.get("risk_decision") is None:
            assert outcome["decision"]["decision"] in ("HOLD", "ABORT")
        else:
            assert risk["action"] != "APPROVED"
        with pytest.raises(ExecutionRefused):
            runtime.execute_approved()
        assert fake_mt5.order_sends == []


class TestReconciliationLoop:
    def test_mismatch_halts_then_acknowledge_restores(self, fake_mt5, control_config):
        from app.brokers.mt5 import MT5Broker

        config = make_runtime_config(
            dry_run=False, real_trading_confirmed="I ACCEPT REAL TRADING RISK"
        )
        fake_mt5.execution_enabled = True
        broker = MT5Broker(mt5_module=fake_mt5)
        runtime = EngineRuntime(config, broker, clock=lambda: CONTROL_REF)
        runtime.connect()
        inject_trend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()
        runtime.execute_approved()
        assert runtime.guard.execution_allowed is True

        # the position disappears (e.g. SL hit) -> mismatch -> halt
        fake_mt5.positions.clear()
        report = runtime.reconcile()
        assert not report.clean
        assert runtime.guard.execution_allowed is False
        # the execution service itself blocks (independent of the runtime):
        proposal, decision = runtime._last_proposal, runtime._last_risk_decision
        request = __import__("app.execution", fromlist=["ExecutionRequest"]).ExecutionRequest.from_proposal(
            proposal, risk_decision_id=decision.gate_decision_id
        )
        blocked = runtime.execution_service.execute(request, decision)
        assert blocked.status.value == "NOT_ATTEMPTED"
        assert "reconciliation_halt" in blocked.message

        # explicit, audited acknowledgement restores execution permission
        assert runtime.acknowledge_reconciliation("position closed at SL") is True
        assert runtime.guard.execution_allowed is True

    def test_events_flow_through_the_runtime_bus(self, runtime, fake_mt5):
        inject_trend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()
        result = runtime.execute_approved()
        types = [e.type for e in runtime.bus.history()]
        assert "EXECUTION_REQUESTED" in types
        assert "EXECUTION_SIMULATED" in types
        assert result.mode is TradingMode.DRY_RUN


class TestStatusHonesty:
    def test_account_type_from_broker_not_config(self, fake_mt5, control_config):
        """§31: config says DRY_RUN; the status shows the BROKER's account."""
        from app.brokers.mt5 import MT5Broker
        from tests.fakes.mt5_fake import ACCOUNT_TRADE_MODE_REAL

        fake_mt5.account = default_account(trade_mode=ACCOUNT_TRADE_MODE_REAL)
        broker = MT5Broker(mt5_module=fake_mt5)
        runtime = EngineRuntime(control_config, broker, clock=lambda: CONTROL_REF)
        runtime.connect()
        assert runtime.status()["account_type"] == "REAL"
        assert runtime.status()["mode"] == "DRY_RUN"  # mode != account type

    def test_status_survives_disconnection(self, runtime):
        runtime.broker.disconnect()
        status = runtime.status()
        assert status["mt5_connected"] is False
        assert status["account_type"] == "unknown"
        assert status["bid"] is None

    def test_read_only_when_trading_disabled(self, fake_mt5):
        from app.brokers.mt5 import MT5Broker

        config = make_runtime_config(trading_enabled=False)
        broker = MT5Broker(mt5_module=fake_mt5)
        runtime = EngineRuntime(config, broker, clock=lambda: CONTROL_REF)
        runtime.connect()
        assert runtime.status()["mode"] == "READ_ONLY"
        assert runtime.execution_service.mode is TradingMode.READ_ONLY

    def test_gate_fails_closed_without_exposure_limit(self, fake_mt5):
        """No MAX_TOTAL_EXPOSURE_USD -> the gate rejects every proposal."""
        from app.brokers.mt5 import MT5Broker

        config = make_runtime_config(max_total_exposure_usd=None)
        broker = MT5Broker(mt5_module=fake_mt5)
        runtime = EngineRuntime(config, broker, clock=lambda: CONTROL_REF)
        runtime.connect()
        inject_trend(fake_mt5)
        runtime.control.start()
        outcome = runtime.evaluate_cycle()
        risk = outcome["risk"]
        if risk.get("risk_decision") is not None:
            assert risk["action"] == "REJECTED"
            assert any("max_total_exposure" in c for c in risk["failed_checks"])
        with pytest.raises(ExecutionRefused):
            runtime.execute_approved()
        assert fake_mt5.order_sends == []
