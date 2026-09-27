"""Decision journal tests (Phase-3 §19): every evaluation answerable, no secrets."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from app.core.enums import DecisionAction, SessionState
from app.decision import (
    DecisionEngine,
    DecisionEngineConfig,
    DecisionRecord,
    InMemoryDecisionJournal,
)
from app.decision.engine import Decision
from app.risk import RiskState
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_snapshot,
    standard_triple,
)


def buy_snapshot():
    up = linear_trend_closes(300, slope=2.0, seed=11)
    return make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)


STATE = RiskState(equity=10_000.0)


class TestJournalContract:
    def test_every_decision_kind_is_recordable(self):
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)

        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)  # BUY
        engine.evaluate(
            buy_snapshot(),
            risk_state=RiskState(equity=10_000.0, open_positions=1),  # ABORT
            now=REF_TIME,
        )
        engine.evaluate(
            make_snapshot(
                standard_triple(h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11),
                created_at=REF_TIME, session_state=SessionState.CLOSED,
            ),  # HOLD
            risk_state=STATE, now=REF_TIME,
        )
        assert len(journal) == 3
        decisions = {r.decision for r in journal.recent()}
        assert decisions == {DecisionAction.BUY, DecisionAction.ABORT, DecisionAction.HOLD}

    def test_engine_passes_journal_automatic(self):
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        stored = journal.by_decision_id(d.decision_id)
        assert stored is not None
        assert stored.decision is DecisionAction.BUY
        assert stored.timestamp == d.timestamp

    def test_without_journal_nothing_breaks(self):
        engine = DecisionEngine(DecisionEngineConfig())
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.BUY


class TestFullProvenance:
    """The journal must answer "why did the bot decide this" — years later."""

    def test_buy_record_answers_why(self):
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        r = journal.by_decision_id(d.decision_id)

        assert r.regime == "TREND_UP"
        assert r.symbol == "XAUUSD"
        assert r.proposal is not None
        assert r.proposal["entry_price"] == pytest.approx(2650.2)
        assert r.proposal["stop_loss"] == pytest.approx(2645.49, abs=0.02)
        assert r.proposal["take_profit"] == pytest.approx(2659.61, abs=0.02)
        assert r.proposal["suggested_volume"] == pytest.approx(0.10)
        assert r.risk_sizing is not None
        assert r.risk_sizing["risk_amount"] == pytest.approx(50.0)
        assert r.fingerprint == d.proposal.fingerprint
        assert r.setup_type is not None
        # agent evidence is embedded — no re-run needed to explain
        names = {a["agent"] for a in r.agent_results}
        assert {"trend", "momentum", "structure"} <= names
        assert r.agent_results[0]["direction"] in {"BUY", "SELL", "NEUTRAL"}
        assert r.synthesis is not None and r.synthesis["action"] == "BUY"
        assert r.supporting_agents == d.supporting_agents
        # gates + config snapshot let an auditor reproduce the threshold logic
        assert r.gates and r.config_snapshot
        assert r.market_snapshot_ref  # snapshot summary attached

    def test_abort_record_carries_reasons(self):
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(
            buy_snapshot(),
            risk_state=RiskState(equity=10_000.0, open_positions=1),
            now=REF_TIME,
        )
        r = journal.recent(1)[0]
        assert r.decision is DecisionAction.ABORT
        assert r.rejection_reasons == ["position_limit"]
        assert r.proposal is None

    def test_record_is_json_serializable(self):
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        r = journal.recent(1)[0]
        dumped = json.dumps(r.model_dump(mode="json"))
        assert len(dumped) > 1000  # a real evidence trail, not a stub

    def test_no_credentials_in_records(self):
        """Records are meant for dashboards/exports — they must never contain
        secrets. Scan the serialized record for credential-shaped keys."""
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        blob = json.dumps(journal.recent(1)[0].model_dump(mode="json")).lower()
        for secret in ("password", "secret", "token", "api_key", "apikey", "login", "credential", "account_number"):
            assert secret not in blob, f"credential-shaped key '{secret}' in journal record"

    def test_snapshot_ref_is_a_summary_not_raw_data(self):
        """The record references the snapshot (summary), not megabytes of candles."""
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        r = journal.recent(1)[0]
        blob = json.dumps(r.market_snapshot_ref)
        assert len(blob) < 2000  # a summary, not the raw series
        # candle COUNTS are fine; OHLC arrays are not
        assert "tick_volume" not in blob
        assert "open" not in blob


class TestH4RenormalizationRecording:
    """Hardening §2: when renormalization is explicitly enabled, the journal
    must record the missing timeframe and the effective weights in force."""

    def _renorm_decision(self, journal):
        from app.core.enums import TimeFrame
        from app.decision.config import TimeframePolicy

        engine = DecisionEngine(
            DecisionEngineConfig(
                timeframe=TimeframePolicy(allow_missing_h4_renormalization=True)
            ),
            journal=journal,
        )
        series = standard_triple(
            h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11,
        )
        del series[TimeFrame.H4]
        return engine.evaluate(
            make_snapshot(series, created_at=REF_TIME), risk_state=STATE, now=REF_TIME,
        )

    def test_journal_records_missing_timeframe_and_weights(self):
        journal = InMemoryDecisionJournal()
        d = self._renorm_decision(journal)
        assert d.decision is DecisionAction.BUY
        r = journal.by_decision_id(d.decision_id)
        assert r is not None
        # the fact is journaled as a warning...
        assert any("H4" in w and "renormalized" in w for w in r.warnings)
        # ...and structurally: missing timeframe + effective weights
        assert r.alignment_detail["renormalized"] is True
        assert r.alignment_detail["missing"] == ["H4"]
        assert r.alignment_detail["effective_weights"]["H1"] == pytest.approx(0.642857, abs=1e-6)
        assert r.alignment_detail["effective_weights"]["M15"] == pytest.approx(0.357143, abs=1e-6)
        # the policy in force is part of the config snapshot
        assert r.config_snapshot["timeframe"]["allow_missing_h4_renormalization"] is True

    def test_strict_mode_hold_is_journaled_with_reason(self):
        from app.core.enums import TimeFrame

        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        series = standard_triple(
            h1_closes=linear_trend_closes(300, slope=2.0, seed=11), h1_seed=11,
        )
        del series[TimeFrame.H4]
        d = engine.evaluate(
            make_snapshot(series, created_at=REF_TIME), risk_state=STATE, now=REF_TIME,
        )
        assert d.decision is DecisionAction.HOLD
        r = journal.by_decision_id(d.decision_id)
        assert r.rejection_reasons == ["missing_primary_context"]
        assert any("allow_missing_h4_renormalization" in w for w in r.warnings)


class TestInMemoryJournal:
    def test_recent_is_newest_first(self):
        journal = InMemoryDecisionJournal()
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        for minutes in (1, 2, 3):
            engine.evaluate(
                buy_snapshot(), risk_state=STATE, now=REF_TIME + timedelta(minutes=minutes),
            )
        stamps = [r.timestamp for r in journal.recent()]
        assert stamps == sorted(stamps, reverse=True)

    def test_for_symbol_filters(self):
        journal = InMemoryDecisionJournal()
        for decision_id, symbol in (("x1", "XAUUSD"), ("x2", "EURUSD")):
            journal.record(
                DecisionRecord.from_decision(
                    Decision.model_construct(
                        decision_id=decision_id, timestamp=REF_TIME,
                        symbol=symbol, decision=DecisionAction.HOLD,
                    )
                )
            )
        assert [r.symbol for r in journal.for_symbol("XAUUSD")] == ["XAUUSD"]

    def test_bounded_memory(self):
        journal = InMemoryDecisionJournal(maxlen=3)
        engine = DecisionEngine(DecisionEngineConfig(), journal=journal)
        for i in range(5):
            engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME + timedelta(minutes=i))
        assert len(journal) == 3

    def test_by_decision_id_missing_returns_none(self):
        assert InMemoryDecisionJournal().by_decision_id("nope") is None
