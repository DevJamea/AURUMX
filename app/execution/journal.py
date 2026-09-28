"""Execution journal (Phase-5 spec §15/§20/§38).

Every execution attempt — blocked, dry-run or live — leaves exactly one
``ExecutionRecord`` carrying the full correlation chain and the *actual*
outcome.  The journal never upgrades a status: ``FILLED`` only ever appears
when the broker state was independently verified.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import Direction, TradingMode
from app.execution.contracts import ExecutionRequest, ExecutionResult, ExecutionStage, ExecutionStatus

#: statuses that mean the request actually reached (or may have reached) the
#: broker.  Only these make a ``request_id`` "already attempted": blocked
#: (NOT_ATTEMPTED), DRY_RUN and duplicate-rejection records never touched the
#: broker, so they must not stop a later real attempt (e.g. DRY_RUN -> DEMO
#: for the same approved proposal, or "trading was disabled, now enabled").
ATTEMPTED_STATUSES: frozenset[ExecutionStatus] = frozenset(
    {
        ExecutionStatus.FILLED,
        ExecutionStatus.PARTIALLY_FILLED,
        ExecutionStatus.UNKNOWN,
        ExecutionStatus.SEND_FAILED,
        ExecutionStatus.REJECTED_BY_BROKER,
        ExecutionStatus.CHECK_FAILED,
        ExecutionStatus.ACCEPTED,
    }
)


class ExecutionRecord(BaseModel):
    """One journaled execution attempt — complete provenance, no secrets."""

    model_config = ConfigDict(allow_inf_nan=False)

    # ---- correlation chain (spec §38) ----------------------------------
    request_id: str
    proposal_id: str
    risk_decision_id: str
    # ---- trade parameters (as requested) ---------------------------------
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    # ---- mode + outcome ---------------------------------------------------
    mode: TradingMode
    status: ExecutionStatus
    stage: ExecutionStage
    dry_run: bool
    # ---- broker verdict ---------------------------------------------------
    retcode: int | None = None
    retcode_description: str = ""
    category: str = ""
    # ---- correlation chain continuation (None = not supplied) -------------
    order_ticket: int | None = None
    deal_ticket: int | None = None
    position_ticket: int | None = None
    # ---- economics --------------------------------------------------------
    fill_price: float | None = None
    filled_volume: float | None = None
    message: str = ""
    reasons: list[str] = Field(default_factory=list)
    timestamp: datetime

    @classmethod
    def from_result(cls, result: ExecutionResult, *, timestamp: datetime) -> ExecutionRecord:
        return cls(
            request_id=result.request_id,
            proposal_id=result.proposal_id,
            risk_decision_id=result.risk_decision_id,
            symbol=result.symbol,
            direction=result.direction,
            volume=result.volume,
            entry_price=result.entry_price,
            stop_loss=result.stop_loss,
            take_profit=result.take_profit,
            mode=result.mode,
            status=result.status,
            stage=result.stage,
            dry_run=result.mode is TradingMode.DRY_RUN,
            retcode=result.retcode,
            retcode_description=result.retcode_description,
            category=result.category,
            order_ticket=result.order_ticket,
            deal_ticket=result.deal_ticket,
            position_ticket=result.position_ticket,
            fill_price=result.fill_price,
            filled_volume=result.filled_volume,
            message=result.message,
            reasons=list(result.reasons),
            timestamp=timestamp,
        )

    @property
    def reached_broker(self) -> bool:
        """True when this attempt reached (or may have reached) the broker."""
        return self.status in ATTEMPTED_STATUSES

    @property
    def claims_real_fill(self) -> bool:
        """True only for statuses backed by verified broker state (§20)."""
        return self.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED)

    def summary(self) -> dict:
        data = self.model_dump(mode="json")
        return data


class ExecutionJournal(Protocol):
    """Storage contract.  Implementations: in-memory (tests/control plane);
    SQLite/PostgreSQL later — same shape as the decision journal."""

    def record(self, record: ExecutionRecord) -> None: ...

    def by_request_id(self, request_id: str) -> ExecutionRecord | None:
        """The most recent record for ``request_id`` (any status)."""
        ...

    def find_attempted(self, request_id: str) -> ExecutionRecord | None:
        """The first record for ``request_id`` that reached the broker
        (status in ``ATTEMPTED_STATUSES``), else None — the idempotency
        authority."""
        ...

    def recent(self, limit: int = 50) -> list[ExecutionRecord]: ...

    def for_symbol(self, symbol: str, limit: int = 50) -> list[ExecutionRecord]: ...


class InMemoryExecutionJournal:
    """Bounded in-memory journal (tests, control plane, diagnostics)."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._records: deque[ExecutionRecord] = deque(maxlen=maxlen)

    def record(self, record: ExecutionRecord) -> None:
        self._records.append(record)

    def by_request_id(self, request_id: str) -> ExecutionRecord | None:
        for record in reversed(self._records):
            if record.request_id == request_id:
                return record
        return None

    def find_attempted(self, request_id: str) -> ExecutionRecord | None:
        for record in self._records:
            if record.request_id == request_id and record.reached_broker:
                return record
        return None

    def recent(self, limit: int = 50) -> list[ExecutionRecord]:
        return list(self._records)[-limit:][::-1]

    def for_symbol(self, symbol: str, limit: int = 50) -> list[ExecutionRecord]:
        return [r for r in self.recent(10_000) if r.symbol == symbol][:limit]

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "ATTEMPTED_STATUSES",
    "ExecutionJournal",
    "ExecutionRecord",
    "ExecutionRequest",
    "InMemoryExecutionJournal",
]
