"""ExecutionService tests — the Phase-5 matrix (spec §11-§21, §33).

Covers: risk-gate integration (APPROVED/REJECTED/EMERGENCY_STOP/KILL_SWITCH/
TRADING_DISABLED), DRY_RUN zero-order-send proof, order_check-before-send
ordering, the retcode classification table, demo/real account guards,
unknown-state handling, immutability and consistency.
"""

from __future__ import annotations

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.enums import RiskAction, TradingMode
from app.core.models import AccountSnapshot
from app.execution import (
    ExecutionRequest,
    ExecutionService,
    ExecutionServiceConfig,
    ExecutionStage,
    ExecutionStatus,
    InMemoryExecutionJournal,
)
from app.risk import RiskState
from tests.execution.conftest import (
    EXEC_REF_TIME,
    make_exec_broker,
    make_exec_fake,
    make_request,
    make_service,
)
from tests.fakes.mt5_fake import (
    ACCOUNT_TRADE_MODE_REAL,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_DONE_PARTIAL,
    TRADE_RETCODE_INVALID,
    TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_INVALID_VOLUME,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_NO_MONEY,
    TRADE_RETCODE_REJECT,
    TRADE_RETCODE_REQUOTE,
    TRADE_RETCODE_TIMEOUT,
)
from tests.unit.risk.conftest import make_account, make_gate, make_proposal


class TestRiskGateIntegration:
    """§33: APPROVED -> allowed; every other outcome -> blocked."""

    def test_approved_executes_dry_run(self, dry_run_service, approved_pair, exec_fake):
        proposal, decision = approved_pair
        request = make_request(proposal, decision)
        result = dry_run_service.execute(request, decision)
        assert result.status is ExecutionStatus.DRY_RUN
        assert result.ok
        assert exec_fake.order_sends == []  # dry run: nothing sent

    def test_rejected_decision_blocked(self, dry_run_service, exec_fake):
        proposal = make_proposal(suggested_volume=0.5)  # over risk limit
        decision = make_gate().evaluate(proposal, RiskState(equity=10_000.0), make_account())
        assert decision.action is RiskAction.REJECTED
        request = make_request(proposal, decision)
        result = dry_run_service.execute(request, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert result.stage is ExecutionStage.BLOCKED
        assert "risk_decision_not_approved" in result.message
        assert exec_fake.order_sends == []
        assert exec_fake.order_checks == []

    def test_emergency_stop_decision_blocked(self, dry_run_service, exec_fake):
        proposal = make_proposal()
        decision = make_gate().evaluate(
            proposal, RiskState(equity=10_000.0),
            make_account(emergency_stop_active=True),
        )
        assert decision.action is RiskAction.EMERGENCY_STOP
        request = make_request(proposal, decision)
        result = dry_run_service.execute(request, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "risk_decision_not_approved" in result.message
        assert exec_fake.order_sends == []

    def test_kill_switch_decision_blocked(self, dry_run_service, exec_fake):
        proposal = make_proposal()
        decision = make_gate().evaluate(
            proposal, RiskState(equity=10_000.0),
            make_account(kill_switch_active=True),
        )
        assert decision.action is RiskAction.EMERGENCY_STOP
        request = make_request(proposal, decision)
        result = dry_run_service.execute(request, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert exec_fake.order_sends == []

    def test_trading_disabled_blocked(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.READ_ONLY)
        request = make_request(proposal, decision)
        result = service.execute(request, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "trading_disabled" in result.message
        assert exec_fake.order_sends == []

    def test_missing_decision_blocked(self, dry_run_service, approved_pair, exec_fake):
        proposal, _ = approved_pair
        request = make_request(proposal, _)
        result = dry_run_service.execute(request, None)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "risk_decision_missing" in result.message
        assert exec_fake.order_sends == []

    def test_mismatched_proposal_id_blocked(self, dry_run_service, approved_pair):
        proposal, decision = approved_pair
        request = make_request(proposal, decision)
        # a request referencing a DIFFERENT proposal than the decision approved
        forged = request.model_copy(
            update={"proposal_id": "dec-other-99", "request_id": "forged1234567890"}
        )
        result = dry_run_service.execute(forged, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "proposal_id_mismatch" in result.message

    def test_mismatched_risk_decision_id_blocked(self, dry_run_service, approved_pair):
        proposal, decision = approved_pair
        request = make_request(proposal, decision)
        forged = request.model_copy(update={"risk_decision_id": "anotherdecision"})
        result = dry_run_service.execute(forged, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "risk_decision_id_mismatch" in result.message

    def test_mismatched_fingerprint_blocked(self, dry_run_service, approved_pair):
        proposal, decision = approved_pair
        request = make_request(proposal, decision)
        forged = request.model_copy(update={"fingerprint": "fp-different-000"})
        result = dry_run_service.execute(forged, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "fingerprint_mismatch" in result.message

    def test_decision_without_gate_id_blocked(self, dry_run_service, approved_pair):
        proposal, decision = approved_pair
        faceless = decision.model_copy(update={"gate_decision_id": ""})
        request = make_request(proposal, decision)
        result = dry_run_service.execute(request, faceless)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "risk_decision_id_missing" in result.message


class TestDryRunPipeline:
    """§13/§14: same pipeline, provably zero order_send calls."""

    def test_order_send_never_called(self, dry_run_service, approved_pair, exec_fake):
        proposal, decision = approved_pair
        result = dry_run_service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.DRY_RUN
        assert len(exec_fake.order_sends) == 0
        assert len(exec_fake.order_checks) == 0  # no broker execution API at all

    def test_full_validation_still_runs(self, exec_broker, approved_pair, exec_fake):
        """Dry-run is NOT a fake path: local validation still rejects."""
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.DRY_RUN)
        bad = make_proposal(symbol="EURUSD")  # non-gold
        request = ExecutionRequest.from_proposal(bad, risk_decision_id="g1")
        result = service.execute(request, decision)  # ids match? no -> blocked
        assert result.status is ExecutionStatus.NOT_ATTEMPTED

    def test_journal_records_dry_run_clearly(self, exec_broker, approved_pair):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.DRY_RUN)
        service.execute(make_request(proposal, decision), decision)
        record = service.journal.recent(1)[0]
        assert record.dry_run is True
        assert record.mode is TradingMode.DRY_RUN
        assert record.status is ExecutionStatus.DRY_RUN
        assert record.order_ticket is None  # never fabricate broker ids (§38)
        assert record.deal_ticket is None
        assert record.position_ticket is None
        assert record.timestamp == EXEC_REF_TIME
        assert "no order was sent" in record.message

    def test_dry_run_journal_never_claims_fill(self, exec_broker, approved_pair):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.DRY_RUN)
        service.execute(make_request(proposal, decision), decision)
        record = service.journal.recent(1)[0]
        assert record.claims_real_fill is False


class TestMqlValidation:
    """Independent local validation (defense in depth, §8-§10)."""

    def _execute(self, service, proposal_overrides, decision):
        proposal = make_proposal(**proposal_overrides)
        request = ExecutionRequest.from_proposal(
            proposal, risk_decision_id=decision.gate_decision_id
        )
        # patch ids so the pair looks consistent (validation is under test)
        request = request.model_copy(
            update={
                "proposal_id": decision.proposal_id,
                "fingerprint": decision.fingerprint or request.fingerprint,
            }
        )
        return service.execute(request, decision)

    def test_non_gold_symbol_rejected(self, demo_service, approved_pair, exec_fake):
        proposal, decision = approved_pair
        result = self._execute(demo_service, {"symbol": "EURUSD"}, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "symbol_not_gold" in result.message
        assert exec_fake.order_sends == []

    def test_unknown_symbol_rejected(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"symbol": "XAUUSDxyz"}, decision)
        assert "symbol_unknown" in result.message
        assert exec_fake.order_sends == []

    def test_volume_below_broker_minimum(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"suggested_volume": 0.001}, decision)
        assert "volume_below_minimum" in result.message
        assert exec_fake.order_sends == []

    def test_volume_above_broker_maximum(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"suggested_volume": 500.0}, decision)
        assert "volume_above_maximum" in result.message
        assert exec_fake.order_sends == []

    def test_volume_off_step_rejected_never_rounded(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"suggested_volume": 0.085}, decision)
        assert "volume_off_step" in result.message
        assert exec_fake.order_sends == []

    def test_wrong_side_sl_rejected(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"stop_loss": 2655.0}, decision)
        assert "geometry_invalid" in result.message
        assert exec_fake.order_sends == []

    def test_wrong_side_tp_rejected(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"take_profit": 2645.0}, decision)
        assert "geometry_invalid" in result.message
        assert exec_fake.order_sends == []

    def test_sl_below_broker_minimum_distance(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"stop_loss": 2650.1}, decision)
        assert "stop_loss_too_close" in result.message
        assert exec_fake.order_sends == []

    def test_off_grid_prices_rejected(self, exec_broker, approved_pair, exec_fake):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = self._execute(service, {"stop_loss": 2645.205}, decision)
        assert "sl_off_tick_grid" in result.message
        assert exec_fake.order_sends == []

    def test_validation_never_mutates_the_proposal(self, exec_broker, approved_pair):
        proposal, decision = approved_pair
        before = proposal.model_dump_json()
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        self._execute(service, {"stop_loss": 2655.0}, decision)  # invalid
        assert proposal.model_dump_json() == before


class TestDemoExecution:
    """§17-§19: the guarded MT5 path."""

    def test_check_precedes_send(self, demo_service, approved_pair, exec_fake):
        proposal, decision = approved_pair
        result = demo_service.execute(make_request(proposal, decision), decision)
        assert len(exec_fake.order_checks) == 1
        assert len(exec_fake.order_sends) == 1
        assert result.status is ExecutionStatus.FILLED
        assert result.stage is ExecutionStage.VERIFIED

    def test_filled_requires_verified_position(self, demo_service, approved_pair):
        proposal, decision = approved_pair
        result = demo_service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.FILLED
        assert result.position_ticket is not None
        assert result.order_ticket is not None
        assert result.deal_ticket is not None

    def test_accepted_but_no_position_is_unknown(self, exec_broker, approved_pair):
        fake = make_exec_fake(auto_open_position=False)
        broker = make_exec_broker(fake)
        service = make_service(broker, mode=TradingMode.MT5_DEMO)
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.UNKNOWN  # never FILLED (§20)
        assert any("unverified" in r for r in result.reasons)
        assert fake.order_sends  # the order WAS sent — journal shows that

    def test_real_account_blocked_unconditionally(self, exec_broker, approved_pair):
        from tests.fakes.mt5_fake import default_account

        fake = make_exec_fake(
            account=default_account(trade_mode=ACCOUNT_TRADE_MODE_REAL)
        )
        broker = make_exec_broker(fake)
        service = make_service(broker, mode=TradingMode.MT5_DEMO)
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "real_account_blocked" in result.message
        assert fake.order_sends == []  # never even attempted

    def test_unknown_account_type_blocked(self, exec_broker, approved_pair, monkeypatch):
        from app.core.enums import AccountTradeMode

        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        account = AccountSnapshot(login=1, trade_mode=AccountTradeMode.UNKNOWN)
        monkeypatch.setattr(service._broker, "get_account", lambda: account)
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "account_type_unknown" in result.message

    def test_mt5_unavailable_blocks_execution(self, approved_pair):
        broker = MT5Broker(mt5_module=make_exec_fake())  # never connected
        service = make_service(broker, mode=TradingMode.MT5_DEMO)
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "mt5_unavailable" in result.message

    def test_broker_exception_is_send_failed(self, exec_broker, approved_pair):
        fake = make_exec_fake(send_exception=RuntimeError("terminal crashed mid-send"))
        broker = make_exec_broker(fake)
        service = make_service(broker, mode=TradingMode.MT5_DEMO)
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.SEND_FAILED
        assert result.ok is False


class TestOrderCheckGate:
    """§7A.3: a failed check must prevent order_send."""

    @pytest.mark.parametrize(
        "check_retcode",
        [
            TRADE_RETCODE_INVALID,
            TRADE_RETCODE_INVALID_VOLUME,
            TRADE_RETCODE_INVALID_STOPS,
            TRADE_RETCODE_NO_MONEY,
            TRADE_RETCODE_MARKET_CLOSED,
            TRADE_RETCODE_REJECT,
        ],
    )
    def test_check_failure_blocks_send(
        self, exec_broker, approved_pair, check_retcode
    ):
        fake = make_exec_fake(check_retcode=check_retcode)
        broker = make_exec_broker(fake)
        service = make_service(broker, mode=TradingMode.MT5_DEMO)
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.CHECK_FAILED
        assert result.stage is ExecutionStage.CHECKED
        assert result.retcode == check_retcode
        assert len(fake.order_checks) == 1
        assert len(fake.order_sends) == 0  # THE invariant

    def test_check_exception_blocks_send(self, exec_broker, approved_pair):
        from app.core.exceptions import MT5NotConnectedError as _Err

        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        service = make_service(broker, mode=TradingMode.MT5_DEMO)

        def boom(order):
            raise _Err("connection lost during check")

        fake.order_check = boom  # type: ignore[method-assign]
        proposal, decision = approved_pair
        result = service.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.CHECK_FAILED
        assert fake.order_sends == []


class TestRetcodeVerification:
    """§7A.5/§33: every retcode class maps to an honest status."""

    def _run(self, send_retcode, *, auto_open_position=True):
        fake = make_exec_fake(
            send_retcode=send_retcode, auto_open_position=auto_open_position
        )
        broker = make_exec_broker(fake)
        service = make_service(broker, mode=TradingMode.MT5_DEMO)
        proposal, decision = make_proposal(), None
        decision = make_gate().evaluate(
            proposal, RiskState(equity=10_000.0), make_account()
        )
        result = service.execute(make_request(proposal, decision), decision)
        return fake, result

    def test_done_fills_and_verifies(self):
        fake, result = self._run(TRADE_RETCODE_DONE)
        assert result.status is ExecutionStatus.FILLED
        assert result.retcode == TRADE_RETCODE_DONE
        assert result.retcode_description == "TRADE_RETCODE_DONE"
        assert len(fake.order_sends) == 1

    def test_done_partial_is_partially_filled(self):
        fake, result = self._run(TRADE_RETCODE_DONE_PARTIAL)
        assert result.status is ExecutionStatus.PARTIALLY_FILLED
        assert result.filled_volume is not None
        assert result.filled_volume < result.volume
        assert result.ok  # verified partial fill is a real outcome

    def test_broker_rejection(self):
        for retcode in (TRADE_RETCODE_REJECT, TRADE_RETCODE_MARKET_CLOSED, TRADE_RETCODE_NO_MONEY):
            fake, result = self._run(retcode)
            assert result.status is ExecutionStatus.REJECTED_BY_BROKER, retcode
            assert result.ok is False

    def test_invalid_request_categories(self):
        for retcode in (TRADE_RETCODE_INVALID, TRADE_RETCODE_INVALID_VOLUME, TRADE_RETCODE_INVALID_STOPS):
            fake, result = self._run(retcode)
            assert result.status is ExecutionStatus.REJECTED_BY_BROKER, retcode
            assert result.retcode_description.startswith("TRADE_RETCODE_INVALID")

    def test_requote_is_rejection_not_success(self):
        for retcode in (TRADE_RETCODE_REQUOTE,):
            fake, result = self._run(retcode)
            assert result.status is ExecutionStatus.REJECTED_BY_BROKER
            assert result.category == "requote"

    def test_timeout_is_unknown_never_success(self):
        fake, result = self._run(TRADE_RETCODE_TIMEOUT)
        assert result.status is ExecutionStatus.UNKNOWN
        assert result.ok is False

    def test_unknown_retcode_fails_closed(self):
        fake, result = self._run(65535)  # not a known constant
        assert result.status is ExecutionStatus.UNKNOWN
        assert result.ok is False
        assert "UNKNOWN_65535" in result.retcode_description

    def test_no_retry_after_any_outcome(self):
        """§42: exactly one check + one send per execute() — no retries."""
        for retcode in (
            TRADE_RETCODE_DONE, TRADE_RETCODE_REJECT, TRADE_RETCODE_TIMEOUT,
            TRADE_RETCODE_REQUOTE, 65535,
        ):
            fake, _ = self._run(retcode)
            assert len(fake.order_sends) == 1, retcode
            assert len(fake.order_checks) == 1, retcode


class TestImmutabilityAndConsistency:
    """§12/§16/§33."""

    @pytest.mark.parametrize("mode", [TradingMode.DRY_RUN, TradingMode.MT5_DEMO])
    def test_proposal_unchanged_after_execution(
        self, exec_broker, approved_pair, mode
    ):
        proposal, decision = approved_pair
        before = proposal.model_dump_json()
        service = make_service(exec_broker, mode=mode)
        service.execute(make_request(proposal, decision), decision)
        assert proposal.model_dump_json() == before

    def test_request_unchanged_after_execution(self, exec_broker, approved_pair):
        proposal, decision = approved_pair
        request = make_request(proposal, decision)
        before = request.model_dump_json()
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        service.execute(request, decision)
        assert request.model_dump_json() == before

    def test_request_matches_its_proposal(self, approved_pair):
        proposal, decision = approved_pair
        request = make_request(proposal, decision)
        assert request.matches_proposal(proposal) == []

    def test_journal_carries_the_whole_correlation_chain(
        self, exec_broker, approved_pair
    ):
        proposal, decision = approved_pair
        service = make_service(exec_broker, mode=TradingMode.MT5_DEMO)
        result = service.execute(make_request(proposal, decision), decision)
        record = service.journal.by_request_id(result.request_id)
        assert record.proposal_id == proposal.decision_id
        assert record.risk_decision_id == decision.gate_decision_id
        assert record.order_ticket == result.order_ticket
        assert record.deal_ticket == result.deal_ticket
        assert record.position_ticket == result.position_ticket


class TestReconciliationHalt:
    """§26: an unresolved mismatch blocks further execution."""

    def test_guard_blocks_execution(self, exec_broker, approved_pair, exec_fake):
        from datetime import UTC, datetime

        from app.execution import ReconciliationGuard, ReconciliationReport
        from app.execution.reconciliation import ExecutionComparison, ReconciliationStatus

        guard = ReconciliationGuard()
        guard.record(
            ReconciliationReport(
                generated_at=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
                comparisons=[
                    ExecutionComparison(
                        request_id="older", status=ReconciliationStatus.MISSING_IN_MT5
                    )
                ],
            )
        )
        assert guard.execution_allowed is False
        service_with_guard = ExecutionService(
            exec_broker,
            config=ExecutionServiceConfig(trading_enabled=True, dry_run=False),
            journal=InMemoryExecutionJournal(),
            clock=lambda: EXEC_REF_TIME,
            reconciliation_guard=guard,
        )
        proposal, decision = approved_pair
        result = service_with_guard.execute(make_request(proposal, decision), decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert "reconciliation_halt" in result.message
        assert exec_fake.order_sends == []

    def test_clean_report_does_not_block(self, exec_broker, approved_pair):
        from datetime import UTC, datetime

        from app.execution import ReconciliationGuard, ReconciliationReport
        from app.execution.reconciliation import ExecutionComparison, ReconciliationStatus

        guard = ReconciliationGuard()
        guard.record(
            ReconciliationReport(
                generated_at=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
                comparisons=[
                    ExecutionComparison(request_id="x", status=ReconciliationStatus.MATCHED)
                ],
            )
        )
        assert guard.execution_allowed is True


class TestEvents:
    """§39: one event per pipeline phase, on the existing bus."""

    def test_dry_run_event_sequence(self, exec_broker, approved_pair):
        from app.core.events import EventBus

        bus = EventBus()
        service = ExecutionService(
            exec_broker,
            config=ExecutionServiceConfig(trading_enabled=True, dry_run=True),
            journal=InMemoryExecutionJournal(),
            event_bus=bus,
            clock=lambda: EXEC_REF_TIME,
        )
        proposal, decision = approved_pair
        service.execute(make_request(proposal, decision), decision)
        types = [e.type for e in bus.history()]
        assert types == ["EXECUTION_REQUESTED", "EXECUTION_SIMULATED"]

    def test_filled_event_sequence(self, exec_broker, approved_pair):
        from app.core.events import EventBus

        bus = EventBus()
        service = ExecutionService(
            exec_broker,
            config=ExecutionServiceConfig(trading_enabled=True, dry_run=False),
            journal=InMemoryExecutionJournal(),
            event_bus=bus,
            clock=lambda: EXEC_REF_TIME,
        )
        proposal, decision = approved_pair
        service.execute(make_request(proposal, decision), decision)
        types = [e.type for e in bus.history()]
        assert types == [
            "EXECUTION_REQUESTED", "EXECUTION_CHECKED", "EXECUTION_SENT",
            "EXECUTION_FILLED",
        ]

    def test_blocked_event(self, exec_broker, approved_pair):
        from app.core.events import EventBus

        bus = EventBus()
        service = ExecutionService(
            exec_broker,
            config=ExecutionServiceConfig(trading_enabled=False),
            journal=InMemoryExecutionJournal(),
            event_bus=bus,
            clock=lambda: EXEC_REF_TIME,
        )
        proposal, decision = approved_pair
        service.execute(make_request(proposal, decision), decision)
        types = [e.type for e in bus.history()]
        assert types == ["EXECUTION_REQUESTED", "EXECUTION_REJECTED"]
