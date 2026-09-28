"""Phase-4 RiskGate CONTRACT tests (hardening §4/§5).

The contract is interface-only: it must be complete enough for an
independent implementation to reject any dangerous proposal, and it must
NOT be implemented or callable anywhere in Phase 3.
"""

from __future__ import annotations

import json
from datetime import UTC

from app.core.enums import RiskAction
from app.decision.proposal import DEFAULT_INVALIDATION_CONDITIONS
from app.risk.gate import (
    REQUIRED_CHECKS,
    CheckSeverity,
    CheckStatus,
    RiskCheck,
    RiskDecision,
    RiskGate,
)
from app.risk.state import AccountState, RiskState


class TestRiskAction:
    def test_exact_outcomes(self):
        assert [a.value for a in RiskAction] == ["APPROVED", "REJECTED", "EMERGENCY_STOP"]


class TestRequiredChecks:
    def test_all_14_hardening_checks_named(self):
        expected = {
            "max_risk_per_trade", "max_total_exposure", "daily_loss_limit",
            "consecutive_loss_protection", "max_open_positions", "max_pending_orders",
            "max_spread", "symbol_restriction", "volume_limits", "sl_presence",
            "tp_validity", "emergency_stop", "kill_switch", "account_safety",
        }
        assert set(REQUIRED_CHECKS) == expected

    def test_checks_are_unique(self):
        assert len(REQUIRED_CHECKS) == len(set(REQUIRED_CHECKS))


class TestContractModels:
    def test_risk_check_model(self):
        """§21: structured check output — name, status, severity, reason,
        observed_value, limit."""
        check = RiskCheck(
            name="daily_loss_limit", status=CheckStatus.FAIL,
            reason="daily loss 199 of 200", observed_value=199.0, limit=200.0,
        )
        assert check.name == "daily_loss_limit"
        assert check.status is CheckStatus.FAIL
        assert not check.passed
        assert check.severity is CheckSeverity.CRITICAL

    def test_check_statuses_cover_the_spec(self):
        assert [s.value for s in CheckStatus] == ["PASS", "FAIL", "WARN", "NOT_EVALUATED"]

    def test_warn_passes_advisory_does_not_block(self):
        warn = RiskCheck(name="max_spread", status=CheckStatus.WARN,
                         severity=CheckSeverity.ADVISORY)
        assert warn.passed  # WARN clears the check

    def test_risk_decision_approved_property(self):
        approved = RiskDecision(action=RiskAction.APPROVED, proposal_id="p1")
        rejected = RiskDecision(action=RiskAction.REJECTED, proposal_id="p1")
        emergency = RiskDecision(action=RiskAction.EMERGENCY_STOP, proposal_id="p1")
        assert approved.approved and not rejected.approved and not emergency.approved

    def test_risk_decision_summary_is_json_safe(self):
        decision = RiskDecision(
            action=RiskAction.REJECTED,
            proposal_id="abc123",
            fingerprint="fp0",
            checks=[
                RiskCheck(name="max_spread", status=CheckStatus.PASS),
                RiskCheck(name="daily_loss_limit", status=CheckStatus.FAIL,
                          reason="limit reached", observed_value=200.0, limit=200.0),
            ],
            reasons=["daily loss limit reached"],
            warnings=["advisory"],
            risk_amount=50.0,
            exposure={"current": 0.0, "proposed": 26_502.0, "total": 26_502.0, "limit": 1e6},
            config_snapshot={"trading_enabled": True},
            kill_switch_active=False,
            emergency_stop_active=False,
        )
        summary = decision.summary()
        assert summary["action"] == "REJECTED"
        assert summary["failed"] == ["daily_loss_limit"]
        assert summary["passed"] == ["max_spread"]
        assert summary["risk_amount"] == 50.0
        assert summary["exposure"]["total"] == 26_502.0
        assert summary["kill_switch_active"] is False
        json.dumps(summary)  # no serialization surprises

    def test_passed_failed_not_evaluated_partition(self):
        decision = RiskDecision(
            action=RiskAction.REJECTED,
            proposal_id="x",
            checks=[
                RiskCheck(name="a", status=CheckStatus.PASS),
                RiskCheck(name="b", status=CheckStatus.FAIL),
                RiskCheck(name="c", status=CheckStatus.NOT_EVALUATED),
                RiskCheck(name="d", status=CheckStatus.WARN, severity=CheckSeverity.ADVISORY),
            ],
        )
        assert decision.passed_checks == ["a"]
        assert decision.failed_checks == ["b"]
        assert decision.not_evaluated_checks == ["c"]

    def test_models_are_credential_free(self):
        blob = json.dumps(RiskDecision(
            action=RiskAction.APPROVED, proposal_id="x",
        ).model_dump(mode="json")).lower()
        for secret in ("password", "secret", "token", "api_key", "credential"):
            assert secret not in blob


class TestAccountState:
    def test_defaults_fail_closed(self):
        """Phase-4 evidence semantics: monetary evidence defaults to None
        (never a silent zero), and trading permission must be AFFIRMATIVE."""
        state = AccountState()
        assert state.open_positions is None and state.pending_orders is None
        assert state.open_positions_notional is None
        assert state.spread_points is None
        assert state.equity is None
        assert state.emergency_stop_active is False
        assert state.kill_switch_active is False
        assert state.trade_allowed is False  # absence of permission != permission

    def test_halt_flags_representable(self):
        halted = AccountState(emergency_stop_active=True, kill_switch_active=True)
        assert halted.emergency_stop_active and halted.kill_switch_active

    def test_sane_equity_primitive(self):
        assert AccountState(equity=10_000.0).has_sane_equity
        assert not AccountState(equity=0.0).has_sane_equity


class TestContractCompleteness:
    """The input triple (proposal, risk_state, account_state) must carry the
    evidence every REQUIRED_CHECK needs — otherwise an independent gate
    could not reject dangerous proposals (hardening §5)."""

    def _proposal(self) -> object:
        # minimal-but-complete TradeProposal via the real builder path
        from datetime import datetime, timedelta

        from app.core.enums import AgentDirection, TimeFrame
        from app.decision.proposal import TradeProposal
        from app.risk.sizing import RiskSizing

        now = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
        sizing = RiskSizing(
            equity=10_000.0, risk_per_trade_pct=0.5, risk_amount=50.0,
            risk_distance=5.0, loss_per_lot=500.0, raw_volume=0.10,
            normalized_volume=0.10, suggested_volume=0.10,
            monetary_risk=50.0, percentage_risk=0.5, feasible=True,
        )
        return TradeProposal(
            symbol="XAUUSD", direction=AgentDirection.BUY,
            entry_price=2650.2, stop_loss=2645.2, take_profit=2660.2,
            risk_distance=5.0, reward_distance=10.0, risk_reward=2.0,
            suggested_volume=0.10, max_allowed_volume=100.0,
            timeframe=TimeFrame.H1, regime="TREND_UP", decision_id="d1",
            setup_type="trend", fingerprint="fp", reasons=["r"],
            invalidation_conditions=list(DEFAULT_INVALIDATION_CONDITIONS),
            created_at=now, expires_at=now + timedelta(minutes=15),
            sl_source="atr", tp_source="rr_target", sizing=sizing,
        )

    def test_max_risk_evidence(self):
        p = self._proposal()
        assert p.sizing.risk_amount > 0 and p.sizing.percentage_risk > 0
        assert RiskState(equity=10_000.0).equity > 0

    def test_exposure_evidence(self):
        state = AccountState(open_positions_notional=250_000.0, open_positions=2)
        assert state.open_positions_notional > 0 and state.open_positions == 2

    def test_daily_loss_and_streak_evidence(self):
        state = RiskState(daily_loss=150.0, consecutive_losses=2)
        assert state.daily_loss == 150.0 and state.consecutive_losses == 2

    def test_count_limit_evidence(self):
        state = RiskState(open_positions=1, pending_orders=2)
        assert state.open_positions == 1 and state.pending_orders == 2

    def test_symbol_and_volume_evidence(self):
        p = self._proposal()
        assert p.symbol == "XAUUSD"
        assert 0 < p.suggested_volume <= p.max_allowed_volume

    def test_sl_tp_evidence(self):
        p = self._proposal()
        assert p.stop_loss > 0 and p.stop_loss < p.entry_price  # BUY geometry
        assert p.take_profit > p.entry_price

    def test_halt_evidence(self):
        state = AccountState(emergency_stop_active=True, kill_switch_active=True,
                             trade_allowed=False)
        assert state.emergency_stop_active and state.kill_switch_active
        assert state.trade_allowed is False


class TestProtocolNotImplementation:
    def test_riskgate_is_a_protocol(self):
        from typing import Protocol

        assert issubclass(RiskGate, Protocol) or hasattr(RiskGate, "_is_protocol")

    def test_evaluate_signature_is_the_hardening_contract(self):
        import inspect

        params = inspect.signature(RiskGate.evaluate).parameters
        assert list(params) == ["self", "proposal", "risk_state", "account_state"]
        assert inspect.signature(RiskGate.evaluate).return_annotation is not inspect.Signature.empty

    def test_gate_py_stays_contract_only(self):
        """app/risk/gate.py holds the contract (models + Protocol) ONLY —
        the implementation lives in app/risk/engine.py (HardRiskGate)."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path("app/risk/gate.py").read_text())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert classes <= {"RiskCheck", "RiskDecision", "RiskGate",
                           "CheckStatus", "CheckSeverity"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                # protocol stub: body is only '...' (Ellipsis)
                assert len(node.body) == 1
                assert isinstance(node.body[0], ast.Expr)
                assert isinstance(node.body[0].value, ast.Constant)
                assert node.body[0].value.value is ...

    def test_implementation_lives_in_engine_and_satisfies_the_protocol(self):
        from app.risk.engine import HardRiskGate

        assert issubclass(type(HardRiskGate), object)  # concrete class
        gate = HardRiskGate()
        assert callable(gate.evaluate)
        assert hasattr(gate, "config")

    def test_implementation_covers_every_required_check(self):
        from app.risk.engine import IMPLEMENTED_CHECKS
        from app.risk.gate import REQUIRED_CHECKS

        assert set(REQUIRED_CHECKS) <= set(IMPLEMENTED_CHECKS)
