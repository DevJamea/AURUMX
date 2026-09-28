"""Durable, transactional execution journal (SQLite, WAL).

Design notes:

* **Append-only.**  Every execution attempt — including blocked attempts and
  rejected duplicates — is a new row, so the audit trail never hides that an
  attempt happened.  ``request_id`` is therefore *not* unique.
* **Idempotency authority.**  ``find_attempted`` returns the first row for a
  ``request_id`` whose status shows the request reached the broker
  (``ATTEMPTED_STATUSES``).  Blocked / DRY_RUN / duplicate-rejection rows do
  not count, so they never stop a later legitimate attempt.  Because the
  data is on disk, this survives process restarts.
* **Thread-safe.**  One connection guarded by one lock (the control server is
  a ``ThreadingHTTPServer``).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from app.execution.journal import ATTEMPTED_STATUSES, ExecutionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    ts              TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    status          TEXT NOT NULL,
    reached_broker  INTEGER NOT NULL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exec_request ON execution_records (request_id);
CREATE INDEX IF NOT EXISTS idx_exec_symbol ON execution_records (symbol);
"""

_ATTEMPTED_VALUES = {status.value for status in ATTEMPTED_STATUSES}


class SQLiteExecutionJournal:
    """Persistent execution journal.  ``path=":memory:"`` for an ephemeral one."""

    def __init__(self, path: str | Path = "data/executions.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # ------------------------------------------------------------------
    def record(self, record: ExecutionRecord) -> None:
        payload = record.model_dump_json()
        reached = 1 if record.status.value in _ATTEMPTED_VALUES else 0
        with self._lock, self._db:  # `with self._db` = one transaction
            self._db.execute(
                "INSERT INTO execution_records "
                "(request_id, ts, symbol, status, reached_broker, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.request_id,
                    record.timestamp.isoformat(),
                    record.symbol,
                    record.status.value,
                    reached,
                    payload,
                ),
            )

    def by_request_id(self, request_id: str) -> ExecutionRecord | None:
        """Most recent record for the request (any status)."""
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM execution_records "
                "WHERE request_id = ? ORDER BY id DESC LIMIT 1",
                (request_id,),
            ).fetchone()
        return ExecutionRecord.model_validate_json(row[0]) if row else None

    def find_attempted(self, request_id: str) -> ExecutionRecord | None:
        """First record for the request that reached the broker, else None."""
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM execution_records "
                "WHERE request_id = ? AND reached_broker = 1 ORDER BY id ASC LIMIT 1",
                (request_id,),
            ).fetchone()
        return ExecutionRecord.model_validate_json(row[0]) if row else None

    def recent(self, limit: int = 50) -> list[ExecutionRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM execution_records ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ExecutionRecord.model_validate_json(row[0]) for row in rows]

    def for_symbol(self, symbol: str, limit: int = 50) -> list[ExecutionRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM execution_records "
                "WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [ExecutionRecord.model_validate_json(row[0]) for row in rows]

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM execution_records").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


__all__ = ["SQLiteExecutionJournal"]
