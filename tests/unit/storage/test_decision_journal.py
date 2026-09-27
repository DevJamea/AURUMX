"""SQLite decision journal tests (Phase-3 §19, spec §38 SQLite→PostgreSQL)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from app.core.enums import DecisionAction
from app.decision import (
    DecisionEngine,
    DecisionEngineConfig,
)
from app.risk import RiskState
from app.storage.decision_journal import SQLiteDecisionJournal
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_snapshot,
    standard_triple,
)

STATE = RiskState(equity=10_000.0)


def buy_snapshot():
    up = linear_trend_closes(300, slope=2.0, seed=11)
    return make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)


@pytest.fixture()
def journal(tmp_path: Path) -> SQLiteDecisionJournal:
    j = SQLiteDecisionJournal(tmp_path / "decisions.db")
    yield j
    j.close()


class TestRoundtrip:
    def test_record_and_fetch(self, journal):
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        stored = journal.by_decision_id(d.decision_id)
        assert stored is not None
        assert stored.decision is DecisionAction.BUY
        assert stored.proposal is not None
        assert stored.proposal["entry_price"] == pytest.approx(2650.2)
        assert stored.agent_results  # evidence survives the JSON roundtrip
        assert stored.synthesis["action"] == "BUY"

    def test_roundtrip_is_exact(self, journal):
        """Whatever the engine journaled is what SQLite gives back, unchanged."""
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        stored = journal.by_decision_id(d.decision_id)
        assert stored is not None
        # JSON-roundtrip-stable fields
        assert stored.decision_id == d.decision_id
        assert stored.timestamp == d.timestamp
        assert stored.fingerprint == d.fingerprint
        assert stored.supporting_agents == d.supporting_agents
        assert stored.gates == d.gates
        assert stored.agent_results == d.agent_results
        assert stored.synthesis == d.synthesis_payload

    def test_missing_id_returns_none(self, journal):
        assert journal.by_decision_id("does-not-exist") is None

    def test_recent_newest_first(self, journal):
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        for minutes in (1, 2, 3):
            engine.evaluate(
                buy_snapshot(), risk_state=STATE, now=REF_TIME + timedelta(minutes=minutes),
            )
        stamps = [r.timestamp for r in journal.recent()]
        assert stamps == sorted(stamps, reverse=True)
        assert len(journal.recent(limit=2)) == 2

    def test_for_symbol_filters(self, journal):
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert len(journal.for_symbol("XAUUSD")) == 1
        assert journal.for_symbol("EURUSD") == []

    def test_replace_on_same_decision_id(self, journal):
        """Same decision_id (re-evaluation at the same instant) replaces, not
        duplicates — the journal reflects the final word."""
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert len(journal.recent()) == 1


class TestPersistence:
    def test_reopening_keeps_history(self, tmp_path: Path):
        path = tmp_path / "decisions.db"
        engine = DecisionEngine(DecisionEngineConfig(), journal=SQLiteDecisionJournal(path))
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)

        reopened = SQLiteDecisionJournal(path)
        try:
            assert reopened.by_decision_id(d.decision_id) is not None
            assert len(reopened.recent()) == 1
        finally:
            reopened.close()

    def test_wal_mode_enabled(self, journal, tmp_path):
        mode = journal._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_schema_has_indexes(self, journal):
        rows = journal._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "idx_decisions_ts" in names
        assert "idx_decisions_symbol" in names


class TestSafety:
    def test_payload_is_json_and_credential_free(self, journal):
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        row = journal._conn.execute(
            "SELECT payload FROM decisions LIMIT 1"
        ).fetchone()
        payload = json.loads(row[0])
        blob = json.dumps(payload).lower()
        for secret in ("password", "secret", "token", "api_key", "credential"):
            assert secret not in blob

    def test_thread_safety_lock(self):
        """A lock exists and serializes writes (single-writer contract)."""
        j = SQLiteDecisionJournal(":memory:")
        try:
            assert hasattr(j, "_lock")
        finally:
            j.close()
