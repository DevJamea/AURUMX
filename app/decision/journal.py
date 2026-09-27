"""Decision journal contract (Phase-3 §19).

The point of the journal is answerability: not just *did* the bot make
money, but **why** did it decide this — every agent output, the synthesis,
the gates that fired, the thresholds in force, the proposal it produced (or
why none).  Records are JSON-safe and credential-free by construction.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.core.enums import DecisionAction
from app.decision.engine import Decision


class DecisionRecord(BaseModel):
    """One journaled evaluation — complete provenance, no secrets."""

    model_config = ConfigDict(allow_inf_nan=False)

    decision_id: str
    timestamp: datetime
    symbol: str
    decision: DecisionAction
    regime: str | None = None
    volatility: str | None = None
    session_state: str | None = None
    spread_points: float | None = None
    conflict_score: float = 0.0
    alignment_score: float | None = None
    supporting_agents: list[str] = []
    opposing_agents: list[str] = []
    # ---- full evidence ---------------------------------------------------
    agent_results: list[dict] = []
    synthesis: dict | None = None
    gates: list[dict] = []
    reasons: list[str] = []
    warnings: list[str] = []
    rejection_reasons: list[str] = []
    # ---- proposal provenance ----------------------------------------------
    proposal: dict | None = None
    risk_sizing: dict | None = None
    fingerprint: str | None = None
    setup_type: str | None = None
    # ---- environment provenance -------------------------------------------
    market_snapshot_ref: dict = {}
    config_snapshot: dict = {}

    @classmethod
    def from_decision(
        cls, decision: Decision, *, agent_results: list[dict] | None = None,
        synthesis: dict | None = None, config_snapshot: dict | None = None,
        snapshot_ref: dict | None = None,
    ) -> DecisionRecord:
        return cls(
            decision_id=decision.decision_id,
            timestamp=decision.timestamp,
            symbol=decision.symbol,
            decision=decision.decision,
            regime=decision.regime.value if decision.regime else None,
            volatility=decision.volatility.value if decision.volatility else None,
            session_state=decision.session_state.value if decision.session_state else None,
            spread_points=decision.spread_points,
            conflict_score=decision.conflict_score,
            alignment_score=decision.alignment_score,
            supporting_agents=list(decision.supporting_agents),
            opposing_agents=list(decision.opposing_agents),
            agent_results=agent_results or [],
            synthesis=synthesis,
            gates=list(decision.gates),
            reasons=list(decision.reasons),
            warnings=list(decision.warnings),
            rejection_reasons=list(decision.rejection_reasons),
            proposal=decision.proposal.model_dump(mode="json") if decision.proposal else None,
            risk_sizing=(
                decision.proposal.sizing.model_dump(mode="json") if decision.proposal else None
            ),
            fingerprint=decision.fingerprint,
            setup_type=decision.setup_type,
            market_snapshot_ref=snapshot_ref or {},
            config_snapshot=config_snapshot or {},
        )


class DecisionJournal(Protocol):
    """Storage contract.  Implementations: in-memory (tests/backtest),
    SQLite (app/storage/decision_journal.py), PostgreSQL later."""

    def record(self, record: DecisionRecord) -> None: ...

    def by_decision_id(self, decision_id: str) -> DecisionRecord | None: ...

    def recent(self, limit: int = 50) -> list[DecisionRecord]: ...

    def for_symbol(self, symbol: str, limit: int = 50) -> list[DecisionRecord]: ...


class InMemoryDecisionJournal:
    """Bounded in-memory journal (tests, backtests, diagnostics)."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._records: deque[DecisionRecord] = deque(maxlen=maxlen)

    def record(self, record: DecisionRecord) -> None:
        self._records.append(record)

    def by_decision_id(self, decision_id: str) -> DecisionRecord | None:
        for record in reversed(self._records):
            if record.decision_id == decision_id:
                return record
        return None

    def recent(self, limit: int = 50) -> list[DecisionRecord]:
        return list(self._records)[-limit:][::-1]

    def for_symbol(self, symbol: str, limit: int = 50) -> list[DecisionRecord]:
        return [r for r in self.recent(10_000) if r.symbol == symbol][:limit]

    def __len__(self) -> int:
        return len(self._records)
