"""Durable, transactional execution journal."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from app.execution.journal import ExecutionRecord

class SQLiteExecutionJournal:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS executions (request_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, payload TEXT NOT NULL)")
        self._db.commit()
    def record(self, record: ExecutionRecord) -> None:
        payload = json.dumps(record.model_dump(mode="json"), separators=(",", ":"))
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO executions VALUES (?,?,?)", (record.request_id, record.timestamp.isoformat(), payload))
    def by_request_id(self, request_id: str) -> ExecutionRecord | None:
        row = self._db.execute("SELECT payload FROM executions WHERE request_id=?", (request_id,)).fetchone()
        return ExecutionRecord.model_validate_json(row[0]) if row else None
    def recent(self, limit: int = 50) -> list[ExecutionRecord]:
        rows = self._db.execute("SELECT payload FROM executions ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [ExecutionRecord.model_validate_json(r[0]) for r in rows]
    def for_symbol(self, symbol: str, limit: int = 50) -> list[ExecutionRecord]:
        return [r for r in self.recent(10000) if r.symbol == symbol][:limit]
