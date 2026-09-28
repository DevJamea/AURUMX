"""Persistence (Phase 3+).

Planned modules: ``database`` (SQLite now, PostgreSQL-ready), ``models``
(decisions, agent results, orders, positions, executions, account snapshots,
risk events, system events, config changes), ``repositories`` (the only code
allowed to touch the database).
"""

from __future__ import annotations

from pathlib import Path

from app.storage.decision_journal import SQLiteDecisionJournal
from app.storage.execution_journal import SQLiteExecutionJournal


def build_persistent_journals(
    data_dir: str | Path,
) -> tuple[SQLiteDecisionJournal, SQLiteExecutionJournal]:
    """The production journals: durable SQLite files under ``data_dir``.

    Used by ``python -m app.control`` so that execution idempotency
    (``duplicate_request``) and the audit trail survive restarts.
    """
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    return (
        SQLiteDecisionJournal(base / "decisions.db"),
        SQLiteExecutionJournal(base / "executions.db"),
    )


__all__ = [
    "SQLiteDecisionJournal",
    "SQLiteExecutionJournal",
    "build_persistent_journals",
]
