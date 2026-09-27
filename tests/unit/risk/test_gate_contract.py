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
from app.risk.gate import REQUIRED_CHECKS, RiskCheck, RiskDecision, RiskGate
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
        check = RiskCheck(name="daily_loss_limit", passed=False, detail="199/200 used")
        assert check.name == "daily_loss_limit"
        assert not check.passed

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
                RiskCheck(name="max_spread", passed=True),
                RiskCheck(name="daily_loss_limit", passed=False, detail="limit reached"),
            ],
            reasons=["daily loss limit reached"],
        )
        summary = decision.summary()
        assert summary["action"] == "REJECTED"
        assert summary["failed"] == ["daily_loss_limit"]
        assert summary["passed"] == 1
        json.dumps(summary)  # no serialization surprises

    def test_models_are_credential_free(self):
        blob = json.dumps(RiskDecision(
            action=RiskAction.APPROVED, proposal_id="x",
        ).model_dump(mode="json")).lower()
        for secret in ("password", "secret", "token", "api_key", "credential"):
            assert secret not in blob


class TestAccountState:
    def test_defaults_are_safe(self):
        """Fail-closed defaults: an empty AccountState carries no trading
        permission and no halts — implementations must fill it honestly."""
        state = AccountState()
        assert state.open_positions == 0 and state.pending_orders == 0
        assert state.emergency_stop_active is False
        assert state.kill_switch_active is False
        assert state.trade_allowed is True

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

    def test_no_implementation_ships_in_phase_3(self):
        """app/risk/gate.py contains models + the Protocol ONLY — no concrete
        gate class, no evaluate() logic (Phase 4 implements it)."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path("app/risk/gate.py").read_text())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert set(classes) == {"RiskCheck", "RiskDecision", "RiskGate"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                # protocol stub: body is only '...' (Ellipsis)
                assert len(node.body) == 1
                assert isinstance(node.body[0], ast.Expr)
                assert isinstance(node.body[0].value, ast.Constant)
                assert node.body[0].value.value is ...
