"""Execution idempotency (request_id) — in-process and across restarts.

A request counts as "already attempted" ONLY when it reached the broker
(ATTEMPTED_STATUSES).  Blocked / DRY_RUN / duplicate-rejection records never
touched the broker and must not stop a later legitimate attempt.
"""

from __future__ import annotations

import pytest

from app.core.enums import Direction, TradingMode
from app.execution import (
    ExecutionRecord,
    ExecutionResult,
    ExecutionService,
    ExecutionServiceConfig,
    ExecutionStage,
    ExecutionStatus,
    InMemoryExecutionJournal,
)
from app.execution.journal import ATTEMPTED_STATUSES
from app.storage import SQLiteExecutionJournal
from tests.execution.conftest import (
    EXEC_REF_TIME,
    make_approved_decision,
    make_exec_broker,
    make_exec_fake,
    make_request,
)
from tests.unit.risk.conftest import make_proposal


def _service(broker, journal, **config) -> ExecutionService:
    return ExecutionService(
        broker,
        config=ExecutionServiceConfig(**config),
        journal=journal,
        clock=lambda: EXEC_REF_TIME,
    )


def _pair():
    proposal = make_proposal()
    decision = make_approved_decision(proposal)
    assert decision.approved, decision.reasons
    return proposal, make_request(proposal, decision), decision


DEMO = dict(trading_enabled=True, dry_run=False)
DRY = dict(trading_enabled=True, dry_run=True)


def _record(status: ExecutionStatus, request_id: str = "req-1") -> ExecutionRecord:
    return ExecutionRecord(
        request_id=request_id, proposal_id="p", risk_decision_id="r", symbol="XAUUSD",
        direction=Direction.LONG, volume=0.1, entry_price=2650.0, stop_loss=2640.0,
        take_profit=2670.0, mode=TradingMode.DRY_RUN, status=status,
        stage=ExecutionStage.SENT, dry_run=False, timestamp=EXEC_REF_TIME,
    )


@pytest.fixture(params=["memory", "sqlite"])
def journal_factory(request, tmp_path):
    if request.param == "memory":
        shared = InMemoryExecutionJournal()
        return lambda: shared
    return lambda: SQLiteExecutionJournal(tmp_path / "exec.db")


class TestDuplicateRejection:
    def test_second_execute_is_rejected_and_never_reaches_broker(self, journal_factory):
        _, request, decision = _pair()
        fake = make_exec_fake()
        service = _service(make_exec_broker(fake), journal_factory(), **DEMO)

        first = service.execute(request, decision)
        second = service.execute(request, decision)

        assert first.status is ExecutionStatus.FILLED
        assert second.status is ExecutionStatus.NOT_ATTEMPTED
        assert second.reasons == ["duplicate_request"]
        assert "duplicate_request" in second.message
        assert len(fake.order_sends) == 1  # exactly one order, ever

    def test_duplicate_rejected_after_restart_with_same_sqlite_file(self, tmp_path):
        db = tmp_path / "exec.db"
        _, request, decision = _pair()

        fake1 = make_exec_fake()
        journal1 = SQLiteExecutionJournal(db)
        first = _service(make_exec_broker(fake1), journal1, **DEMO).execute(request, decision)
        assert first.status is ExecutionStatus.FILLED
        journal1.close()  # "process exits"

        fake2 = make_exec_fake()  # fresh terminal, fresh service, same file
        second = _service(
            make_exec_broker(fake2), SQLiteExecutionJournal(db), **DEMO
        ).execute(request, decision)
        assert second.status is ExecutionStatus.NOT_ATTEMPTED
        assert second.reasons == ["duplicate_request"]
        assert fake2.order_sends == []

    def test_duplicate_result_is_fresh_and_carries_no_old_tickets(self, journal_factory):
        _, request, decision = _pair()
        service = _service(make_exec_broker(make_exec_fake()), journal_factory(), **DEMO)
        first = service.execute(request, decision)
        assert first.order_ticket is not None and first.fill_price is not None

        duplicate = service.execute(request, decision)
        assert isinstance(duplicate, ExecutionResult)
        assert duplicate.stage is ExecutionStage.BLOCKED
        assert duplicate.order_ticket is None
        assert duplicate.deal_ticket is None
        assert duplicate.position_ticket is None
        assert duplicate.fill_price is None
        assert duplicate.filled_volume is None

    def test_rejected_duplicates_are_journaled_not_hidden(self, journal_factory):
        _, request, decision = _pair()
        journal = journal_factory()
        service = _service(make_exec_broker(make_exec_fake()), journal, **DEMO)
        service.execute(request, decision)
        service.execute(request, decision)
        service.execute(request, decision)

        records = journal.recent(10)
        assert len(records) == 3
        assert [r.status for r in records].count(ExecutionStatus.NOT_ATTEMPTED) == 2
        # the ORIGINAL attempt stays the idempotency authority
        original = journal.find_attempted(request.request_id)
        assert original is not None and original.status is ExecutionStatus.FILLED
        assert original.order_ticket is not None


class TestBlockedAndDryRunDoNotConsumeTheRequest:
    def test_dry_run_then_demo_for_same_request_is_allowed(self, journal_factory):
        _, request, decision = _pair()
        journal = journal_factory()
        fake = make_exec_fake()
        broker = make_exec_broker(fake)

        dry = _service(broker, journal, **DRY).execute(request, decision)
        assert dry.status is ExecutionStatus.DRY_RUN
        assert fake.order_sends == []

        demo = _service(broker, journal, **DEMO).execute(request, decision)
        assert demo.status is ExecutionStatus.FILLED
        assert len(fake.order_sends) == 1

    def test_blocked_by_trading_disabled_then_enabled_is_allowed(self, journal_factory):
        _, request, decision = _pair()
        journal = journal_factory()
        fake = make_exec_fake()
        broker = make_exec_broker(fake)

        blocked = _service(broker, journal, trading_enabled=False, dry_run=True).execute(
            request, decision
        )
        assert blocked.status is ExecutionStatus.NOT_ATTEMPTED
        assert "trading_disabled" in blocked.message

        enabled = _service(broker, journal, **DEMO).execute(request, decision)
        assert enabled.status is ExecutionStatus.FILLED
        assert len(fake.order_sends) == 1


class TestAttemptedSemantics:
    @pytest.mark.parametrize("status", sorted(ATTEMPTED_STATUSES, key=lambda s: s.value))
    def test_broker_reaching_statuses_count_as_attempted(self, journal_factory, status):
        journal = journal_factory()
        journal.record(_record(status))
        assert journal.find_attempted("req-1") is not None

    @pytest.mark.parametrize(
        "status", [ExecutionStatus.NOT_ATTEMPTED, ExecutionStatus.DRY_RUN]
    )
    def test_non_broker_statuses_do_not_count(self, journal_factory, status):
        journal = journal_factory()
        journal.record(_record(status))
        assert journal.find_attempted("req-1") is None
        assert journal.by_request_id("req-1") is not None  # still journaled

    @pytest.mark.parametrize(
        "status",
        [
            ExecutionStatus.UNKNOWN,
            ExecutionStatus.SEND_FAILED,
            ExecutionStatus.REJECTED_BY_BROKER,
            ExecutionStatus.CHECK_FAILED,
        ],
    )
    def test_failed_or_unknown_attempts_block_replay(self, journal_factory, status):
        """A failed/unknown attempt may still have moved money: no replay."""
        _, request, decision = _pair()
        journal = journal_factory()
        journal.record(_record(status, request_id=request.request_id))
        fake = make_exec_fake()
        result = _service(make_exec_broker(fake), journal, **DEMO).execute(request, decision)
        assert result.status is ExecutionStatus.NOT_ATTEMPTED
        assert result.reasons == ["duplicate_request"]
        assert fake.order_sends == []
