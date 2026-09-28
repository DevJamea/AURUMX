"""Reconciliation tests (spec §22-§26, §33): every state + the guard."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.enums import Direction, TradingMode
from app.execution import (
    ExecutionService,
    ExecutionServiceConfig,
    ExecutionStage,
    ExecutionStatus,
    InMemoryExecutionJournal,
    Reconciler,
    ReconciliationGuard,
    ReconciliationStatus,
)
from app.execution.contracts import AURUMX_MAGIC
from app.execution.journal import ExecutionRecord
from app.risk import RiskState
from tests.execution.conftest import EXEC_REF_TIME, make_exec_broker, make_exec_fake, make_request
from tests.fakes.mt5_fake import FakePosition
from tests.unit.risk.conftest import make_account, make_gate, make_proposal

REF = datetime(2026, 1, 5, 12, 45, tzinfo=UTC)


def make_fill_record(**overrides) -> ExecutionRecord:
    base = dict(
        request_id="req-0001", proposal_id="prop-0001", risk_decision_id="gate-0001",
        symbol="XAUUSD", direction=Direction.LONG, volume=0.08,
        entry_price=2650.2, stop_loss=2645.2, take_profit=2660.2,
        mode=TradingMode.MT5_DEMO, status=ExecutionStatus.FILLED,
        stage=ExecutionStage.VERIFIED, order_ticket=900001, deal_ticket=900002,
        position_ticket=900001, fill_price=2650.2, filled_volume=0.08,
        dry_run=False, timestamp=REF,
    )
    base.update(overrides)
    return ExecutionRecord(**base)


def make_position(**overrides) -> FakePosition:
    base = dict(
        ticket=900001, symbol="XAUUSD", type=0, volume=0.08,
        price_open=2650.2, price_current=2650.2, sl=2645.2, tp=2660.2,
        profit=0.0, swap=0.0, time=1_767_225_600, comment="AURUMX",
        magic=AURUMX_MAGIC,
    )
    base.update(overrides)
    return FakePosition(**base)


@pytest.fixture()
def journal() -> InMemoryExecutionJournal:
    return InMemoryExecutionJournal()


@pytest.fixture()
def setup(journal):
    """(fake, broker, reconciler) — fake with one verified fill + position."""
    fake = make_exec_fake()
    fake.positions.append(make_position())
    broker = make_exec_broker(fake)
    journal.record(make_fill_record())
    return fake, broker, Reconciler(broker, journal)


class TestReconciliationStates:
    def test_matched(self, setup):
        fake, broker, reconciler = setup
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MATCHED
        assert report.clean
        assert report.counts == {"MATCHED": 1}

    def test_missing_in_mt5(self, journal):
        """Journal claims FILLED but the position is gone -> MISSING_IN_MT5
        (never silently repaired, never adopted as fine)."""
        fake = make_exec_fake()  # no positions
        broker = make_exec_broker(fake)
        journal.record(make_fill_record())
        report = Reconciler(broker, journal).reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MISSING_IN_MT5
        assert not report.clean

    def test_missing_in_journal(self, setup):
        """An AURUMX-magic position the journal cannot explain."""
        fake, broker, reconciler = setup
        fake.positions.append(make_position(ticket=888001, volume=0.10))
        report = reconciler.reconcile(now=REF)
        statuses = [c.status for c in report.comparisons]
        assert ReconciliationStatus.MISSING_IN_JOURNAL in statuses
        assert not report.clean

    def test_mismatch_volume(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(volume=0.20)  # beyond one step
        report = reconciler.reconcile(now=REF)
        comparison = report.comparisons[0]
        assert comparison.status is ReconciliationStatus.MISMATCH
        assert any("volume" in d for d in comparison.differences)

    def test_mismatch_sl(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(sl=2640.0)  # > one tick away
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MISMATCH
        assert any("stop_loss" in d for d in report.comparisons[0].differences)

    def test_mismatch_tp(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(tp=2670.0)
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MISMATCH
        assert any("take_profit" in d for d in report.comparisons[0].differences)

    def test_mismatch_direction(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(type=1)  # SELL vs journal LONG
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MISMATCH
        assert any("direction" in d for d in report.comparisons[0].differences)

    def test_unknown_when_broker_unreachable(self, journal):
        fake = make_exec_fake()
        broker = MT5Broker_unconnected(fake)
        journal.record(make_fill_record())
        report = Reconciler(broker, journal).reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.UNKNOWN
        assert not report.clean

    def test_unknown_for_unconfirmed_journal_records(self, journal):
        """ACCEPTED (sent, unverified) records with no position are
        UNKNOWN — absence of evidence is not proof of loss."""
        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        journal.record(
            make_fill_record(status=ExecutionStatus.ACCEPTED, stage=ExecutionStage.SENT)
        )
        report = Reconciler(broker, journal).reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.UNKNOWN

    def test_dry_run_records_are_skipped(self, journal):
        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        journal.record(
            make_fill_record(
                mode=TradingMode.DRY_RUN, status=ExecutionStatus.DRY_RUN,
                stage=ExecutionStage.SIMULATED, order_ticket=None,
                deal_ticket=None, position_ticket=None,
            )
        )
        report = Reconciler(broker, journal).reconcile(now=REF)
        assert report.comparisons == []
        assert report.skipped and "dry-run" in report.skipped[0]

    def test_blocked_records_are_skipped(self, journal):
        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        journal.record(
            make_fill_record(
                status=ExecutionStatus.NOT_ATTEMPTED, stage=ExecutionStage.BLOCKED,
                order_ticket=None, deal_ticket=None, position_ticket=None,
            )
        )
        report = Reconciler(broker, journal).reconcile(now=REF)
        assert report.comparisons == []


def MT5Broker_unconnected(fake):
    from app.brokers.mt5 import MT5Broker

    return MT5Broker(mt5_module=fake)  # never connected -> broker errors


class TestTolerances:
    """§24: documented tolerances hide nothing material."""

    def test_one_tick_sl_rounding_is_matched(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(sl=2645.21)  # one tick off
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MATCHED

    def test_slippage_within_deviation_is_matched(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(price_open=2650.5)  # 30 ticks slippage
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MATCHED

    def test_excessive_slippage_is_mismatch(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(price_open=2652.0)  # 180 ticks
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MISMATCH
        assert any("entry" in d for d in report.comparisons[0].differences)

    def test_volume_within_one_step_is_matched(self, setup):
        fake, broker, reconciler = setup
        fake.positions[0] = make_position(volume=0.08 + 0.01)
        report = reconciler.reconcile(now=REF)
        assert report.comparisons[0].status is ReconciliationStatus.MATCHED


class TestUnrelatedActivityIgnored:
    def test_foreign_magic_positions_are_ignored(self, journal):
        """Positions opened by other EAs/humans never touch our journal."""
        fake = make_exec_fake()
        fake.positions.append(make_position(ticket=777001, magic=12345))
        broker = make_exec_broker(fake)
        journal.record(make_fill_record())
        report = Reconciler(broker, journal).reconcile(now=REF)
        # the foreign position is invisible; the journal record is missing in mt5
        assert report.comparisons[0].status is ReconciliationStatus.MISSING_IN_MT5
        assert all("777001" not in c.detail for c in report.comparisons)


class TestReconciliationGuard:
    def test_clean_report_allows_execution(self, setup):
        _, broker, reconciler = setup
        guard = ReconciliationGuard()
        guard.record(reconciler.reconcile(now=REF))
        assert guard.execution_allowed is True
        assert guard.status == "CLEAN"

    def test_mismatch_halts_execution(self, journal):
        fake = make_exec_fake()  # no position -> MISSING_IN_MT5
        broker = make_exec_broker(fake)
        journal.record(make_fill_record())
        guard = ReconciliationGuard()
        guard.record(Reconciler(broker, journal).reconcile(now=REF))
        assert guard.execution_allowed is False
        assert guard.status == "MISMATCH"

    def test_no_report_yet_allows_execution(self):
        assert ReconciliationGuard().execution_allowed is True
        assert ReconciliationGuard().status == "NOT_RUN"

    def test_acknowledge_is_explicit_and_audited(self, journal):
        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        journal.record(make_fill_record())
        guard = ReconciliationGuard()
        guard.record(Reconciler(broker, journal).reconcile(now=REF))
        assert guard.acknowledge() is True
        assert guard.execution_allowed is True
        assert guard.status == "ACKNOWLEDGED"

    def test_acknowledge_clean_report_is_a_noop(self, setup):
        _, broker, reconciler = setup
        guard = ReconciliationGuard()
        guard.record(reconciler.reconcile(now=REF))
        assert guard.acknowledge() is False  # nothing to acknowledge

    def test_new_report_resets_acknowledgement(self, journal):
        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        journal.record(make_fill_record())
        guard = ReconciliationGuard()
        guard.record(Reconciler(broker, journal).reconcile(now=REF))
        guard.acknowledge()
        # reconcile again, still dirty -> acknowledgement does not persist
        guard.record(Reconciler(broker, journal).reconcile(now=REF))
        assert guard.execution_allowed is False


class TestGuardIntegrationWithService:
    def def_test_placeholder(self):  # keeps pytest happy about class naming
        pass

    def test_mismatch_blocks_next_execution(self, journal):
        """The full §26 loop: execute -> fill disappears -> reconcile ->
        halt -> next execution blocked."""
        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        guard = ReconciliationGuard()
        journal = InMemoryExecutionJournal()
        service = ExecutionService(
            broker,
            config=ExecutionServiceConfig(trading_enabled=True, dry_run=False),
            journal=journal,
            clock=lambda: EXEC_REF_TIME,
            reconciliation_guard=guard,
        )
        proposal = make_proposal()
        decision = make_gate().evaluate(proposal, RiskState(equity=10_000.0), make_account())
        first = service.execute(make_request(proposal, decision), decision)
        assert first.status is ExecutionStatus.FILLED

        # the position closes (SL hit) and disappears from MT5
        fake.positions.clear()
        guard.record(Reconciler(broker, journal).reconcile(now=REF))
        assert guard.execution_allowed is False

        # a DISTINCT request_id, so the idempotency guard cannot mask the
        # reconciliation guard under test
        second_request = make_request(proposal, decision).model_copy(
            update={"request_id": "req-distinct-after-mismatch"}
        )
        second = service.execute(second_request, decision)
        assert second.status is ExecutionStatus.NOT_ATTEMPTED
        assert "reconciliation_halt" in second.message
        assert len(fake.order_sends) == 1  # no second order was sent

        # ... and replaying the ORIGINAL request is refused too (idempotency)
        replay = service.execute(make_request(proposal, decision), decision)
        assert replay.status is ExecutionStatus.NOT_ATTEMPTED
        assert len(fake.order_sends) == 1


class TestEvents:
    def test_clean_report_emits_matched(self, setup):
        from app.core.events import EventBus

        _, _, reconciler = setup
        bus = EventBus()
        reconciler.reconcile(now=REF, event_bus=bus)
        assert [e.type for e in bus.history()] == ["RECONCILIATION_MATCHED"]

    def test_dirty_report_emits_mismatch(self, journal):
        from app.core.events import EventBus

        fake = make_exec_fake()
        broker = make_exec_broker(fake)
        journal.record(make_fill_record())
        bus = EventBus()
        Reconciler(broker, journal).reconcile(now=REF, event_bus=bus)
        assert [e.type for e in bus.history()] == ["RECONCILIATION_MISMATCH"]
