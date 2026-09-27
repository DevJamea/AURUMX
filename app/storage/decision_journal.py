"""SQLite decision journal (Phase-3 §19; SQLite → PostgreSQL per spec §38).

One row per evaluation, JSON payload for full provenance.  Thread-safe;
suitable for the single-process Phase-3 worker.  The schema is deliberately
flat (decision_id PK + indexed query columns + JSON blob) so migrating to
PostgreSQL later is a transport change, not a redesign.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from app.decision.journal import DecisionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id   TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    decision      TEXT NOT NULL,
    fingerprint   TEXT,
    setup_type    TEXT,
    regime        TEXT,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions (ts);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions (symbol);
"""


class SQLiteDecisionJournal:
    """Persistent journal.  ``path=":memory:"`` for an ephemeral instance."""

    def __init__(self, path: str | Path = "data/decisions.db") -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    def record(self, record: DecisionRecord) -> None:
        payload = record.model_dump(mode="json")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions "
                "(decision_id, ts, symbol, decision, fingerprint, setup_type, regime, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.decision_id,
                    record.timestamp.isoformat(),
                    record.symbol,
                    record.decision.value,
                    record.fingerprint,
                    record.setup_type,
                    record.regime,
                    json.dumps(payload, default=str),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    def _row_to_record(self, row) -> DecisionRecord:
        return DecisionRecord(**json.loads(row[0]))

    def by_decision_id(self, decision_id: str) -> DecisionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def recent(self, limit: int = 50) -> list[DecisionRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM decisions ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def for_symbol(self, symbol: str, limit: int = 50) -> list[DecisionRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM decisions WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self, decision: str | None = None) -> int:
        with self._lock:
            if decision is None:
                row = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE decision = ?", (decision,)
                ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
