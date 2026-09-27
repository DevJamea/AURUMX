"""Execution journal tests (spec §15/§20/§38)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.enums import Direction, TradingMode
from app.execution import (
    ExecutionRecord,
    ExecutionStage,
    ExecutionStatus,
    InMemoryExecutionJournal,
)
from app.execution.contracts import ExecutionResult


def make_result(**overrides) -> ExecutionResult:
    base = dict(
        request_id="req-0001", proposal_id="prop-0001", risk_decision_id="gate-0001",
        symbol="XAUUSD", direction=Direction.LONG, volume=0.08,
        entry_price=2650.2, stop_loss=2645.2, take_profit=2660.2,
        mode=TradingMode.MT5_DEMO, status=ExecutionStatus.FILLED,
        stage=ExecutionStage.VERIFIED, order_ticket=900001, deal_ticket=900002,
        position_ticket=900001, fill_price=2650.25, filled_volume=0.08,
    )
    base.update(overrides)
    return ExecutionResult(**base)


class TestExecutionRecord:
    def test_from_result_carries_the_correlation_chain(self):
        now = datetime(2026, 1, 5, 12, 30, tzinfo=UTC)
        record = ExecutionRecord.from_result(make_result(), timestamp=now)
        assert record.request_id == "req-0001"
        assert record.proposal_id == "prop-0001"
        assert record.risk_decision_id == "gate-0001"
        assert record.order_ticket == 900001
        assert record.deal_ticket == 900002
        assert record.position_ticket == 900001
        assert record.timestamp == now

    def test_dry_run_flag(self):
        now = datetime(2026, 1, 5, 12, 30, tzinfo=UTC)
        record = ExecutionRecord.from_result(
            make_result(mode=TradingMode.DRY_RUN, status=ExecutionStatus.DRY_RUN,
                        stage=ExecutionStage.SIMULATED, order_ticket=None,
                        deal_ticket=None, position_ticket=None),
            timestamp=now,
        )
        assert record.dry_run is True

    def test_claims_real_fill_only_for_verified_statuses(self):
        now = datetime(2026, 1, 5, 12, 30, tzinfo=UTC)
        assert ExecutionRecord.from_result(make_result(), timestamp=now).claims_real_fill
        for status in (ExecutionStatus.DRY_RUN, ExecutionStatus.ACCEPTED,
                       ExecutionStatus.UNKNOWN, ExecutionStatus.NOT_ATTEMPTED,
                       ExecutionStatus.REJECTED_BY_BROKER):
            record = ExecutionRecord.from_result(
                make_result(status=status), timestamp=now
            )
            assert record.claims_real_fill is False, status

    def test_summary_is_json_safe(self):
        import json

        record = ExecutionRecord.from_result(
            make_result(), timestamp=datetime(2026, 1, 5, 12, 30, tzinfo=UTC)
        )
        blob = json.dumps(record.summary())
        assert "FILLED" in blob
        assert "password" not in blob.lower()


class TestInMemoryJournal:
    def test_record_and_query(self):
        journal = InMemoryExecutionJournal()
        now = datetime(2026, 1, 5, 12, 30, tzinfo=UTC)
        journal.record(ExecutionRecord.from_result(make_result(), timestamp=now))
        journal.record(
            ExecutionRecord.from_result(
                make_result(request_id="req-0002", symbol="XAUUSDm"),
                timestamp=now,
            )
        )
        assert len(journal) == 2
        assert journal.by_request_id("req-0001") is not None
        assert journal.by_request_id("missing") is None
        assert [r.request_id for r in journal.recent(10)] == ["req-0002", "req-0001"]
        assert len(journal.for_symbol("XAUUSD")) == 1
        assert journal.for_symbol("EURUSD") == []

    def test_recent_respects_limit(self):
        journal = InMemoryExecutionJournal()
        now = datetime(2026, 1, 5, 12, 30, tzinfo=UTC)
        for i in range(5):
            journal.record(
                ExecutionRecord.from_result(
                    make_result(request_id=f"req-{i:04d}"), timestamp=now
                )
            )
        assert len(journal.recent(2)) == 2
        assert journal.recent(2)[0].request_id == "req-0004"  # newest first
