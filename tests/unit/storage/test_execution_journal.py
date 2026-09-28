"""SQLiteExecutionJournal: durability, thread-safety, ordering, wiring."""

from __future__ import annotations

import threading

from app.core.enums import Direction, TradingMode
from app.execution import ExecutionRecord, ExecutionStage, ExecutionStatus
from app.storage import (
    SQLiteDecisionJournal,
    SQLiteExecutionJournal,
    build_persistent_journals,
)
from tests.execution.conftest import EXEC_REF_TIME


def _record(request_id: str, status=ExecutionStatus.FILLED, symbol="XAUUSD") -> ExecutionRecord:
    return ExecutionRecord(
        request_id=request_id, proposal_id="p", risk_decision_id="r", symbol=symbol,
        direction=Direction.LONG, volume=0.1, entry_price=2650.0, stop_loss=2640.0,
        take_profit=2670.0, mode=TradingMode.DRY_RUN, status=status,
        stage=ExecutionStage.SENT, dry_run=False, timestamp=EXEC_REF_TIME,
    )


def test_records_survive_reopen(tmp_path):
    db = tmp_path / "e.db"
    journal = SQLiteExecutionJournal(db)
    journal.record(_record("a"))
    journal.close()
    reopened = SQLiteExecutionJournal(db)
    assert reopened.by_request_id("a") is not None
    assert len(reopened) == 1


def test_append_only_same_request_id_keeps_every_row(tmp_path):
    journal = SQLiteExecutionJournal(tmp_path / "e.db")
    journal.record(_record("a"))
    journal.record(_record("a", ExecutionStatus.NOT_ATTEMPTED))
    assert len(journal) == 2
    assert journal.by_request_id("a").status is ExecutionStatus.NOT_ATTEMPTED  # latest
    assert journal.find_attempted("a").status is ExecutionStatus.FILLED  # first real


def test_recent_is_newest_first_even_with_identical_timestamps(tmp_path):
    journal = SQLiteExecutionJournal(tmp_path / "e.db")
    for name in ("a", "b", "c"):
        journal.record(_record(name))
    assert [r.request_id for r in journal.recent(3)] == ["c", "b", "a"]


def test_for_symbol_filters(tmp_path):
    journal = SQLiteExecutionJournal(tmp_path / "e.db")
    journal.record(_record("a", symbol="XAUUSD"))
    journal.record(_record("b", symbol="EURUSD"))
    assert [r.request_id for r in journal.for_symbol("XAUUSD")] == ["a"]


def test_concurrent_writers_are_safe(tmp_path):
    journal = SQLiteExecutionJournal(tmp_path / "e.db")
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(50):
                journal.record(_record(f"t{n}-{i}"))
                journal.find_attempted(f"t{n}-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(journal) == 400


def test_build_persistent_journals_creates_durable_files(tmp_path):
    data_dir = tmp_path / "nested" / "data"
    decisions, executions = build_persistent_journals(data_dir)
    assert isinstance(decisions, SQLiteDecisionJournal)
    assert isinstance(executions, SQLiteExecutionJournal)
    assert (data_dir / "decisions.db").exists()
    assert (data_dir / "executions.db").exists()
