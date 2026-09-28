"""Persistence (Phase 3+).

Planned modules: ``database`` (SQLite now, PostgreSQL-ready), ``models``
(decisions, agent results, orders, positions, executions, account snapshots,
risk events, system events, config changes), ``repositories`` (the only code
allowed to touch the database).
"""

from app.storage.execution_journal import SQLiteExecutionJournal

__all__ = ["SQLiteExecutionJournal"]
