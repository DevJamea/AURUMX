"""Phase-5 execution contracts (spec §7A.1/§7A.2, §11, §16, §38).

The execution layer converts an **already approved** proposal into a
controlled broker request and reports what actually happened.  Everything
here is deterministic data:

* ``ExecutionRequest`` is derived verbatim from a ``TradeProposal`` — the
  execution layer never invents or "normalizes" trade parameters (if the
  approved values are not executable, the request is rejected, not fixed).
* IDs are derived by hashing (like the Phase-4 ``gate_decision_id``) — no
  uuid/random/wall clock anywhere.
* ``ExecutionResult`` distinguishes *how far the pipeline got*
  (``stage``) from *what the outcome was* (``status``).  Unknown state is
  never collapsed into success: ``UNKNOWN`` is a first-class outcome.

The correlation chain every record carries (spec §38):

    proposal_id -> risk_decision_id -> request_id -> order_ticket
                -> deal_ticket -> position_ticket

Broker identifiers are only ever *reported*, never fabricated — a field the
broker did not supply stays ``None``.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from app.core.enums import AgentDirection, Direction, TradingMode
from app.core.models import _FiniteModel

if TYPE_CHECKING:  # pragma: no cover
    from app.decision.proposal import TradeProposal

#: Dedicated MetaTrader "magic number" identifying AURUMX activity
#: (spec §43).  Value: 0x41555258 = ASCII "AURX" big-endian (1096110680),
#: a positive int32 that reconciliation can filter on to distinguish
#: AURUMX trades from unrelated manual/EA activity on the same account.
AURUMX_MAGIC = 0x41555258

#: Default max price slippage for market orders, in points.
DEFAULT_DEVIATION_POINTS = 20


class ExecutionStatus(StrEnum):
    """Outcome of one execution attempt (spec §7A.2 — exact set).

    Failure is never collapsed into success:

    * ``NOT_ATTEMPTED``   — blocked before any broker call (risk decision
      not approved, mismatched proposal, trading disabled, validation
      failure, reconciliation halt, MT5 unavailable).
    * ``DRY_RUN``        — full pipeline, simulated fill, no order sent.
    * ``CHECK_FAILED``   — broker ``order_check`` refused; nothing sent.
    * ``SEND_FAILED``    — the send attempt itself errored (exception).
    * ``REJECTED_BY_BROKER`` — the broker returned a definite rejection
      retcode (incl. invalid request/volume/stops, market closed, no
      money, requote).
    * ``ACCEPTED``       — broker confirmed the request (no state verified
      yet) — only an intermediate state in journal terms.
    * ``FILLED``         — broker confirmed AND the resulting MT5 state
      (position) was independently verified.
    * ``PARTIALLY_FILLED`` — verified, but the filled volume is smaller
      than requested.
    * ``UNKNOWN``        — the state could not be established confidently
      (missing retcode, ambiguous/timeout responses, unverified fill).
    """

    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    DRY_RUN = "DRY_RUN"
    CHECK_FAILED = "CHECK_FAILED"
    SEND_FAILED = "SEND_FAILED"
    REJECTED_BY_BROKER = "REJECTED_BY_BROKER"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal_failure(self) -> bool:
        """Definite failures (as opposed to blocked/unknown/success)."""
        return self in (self.CHECK_FAILED, self.SEND_FAILED, self.REJECTED_BY_BROKER)

    @property
    def is_verified_success(self) -> bool:
        """Only statuses backed by actual broker state (spec §20)."""
        return self in (self.FILLED, self.PARTIALLY_FILLED)

    @property
    def is_blocked(self) -> bool:
        return self is self.NOT_ATTEMPTED


class ExecutionStage(StrEnum):
    """How far the pipeline reached (spec §20: the journal must
    distinguish REQUESTED / CHECKED / SENT / ... states)."""

    REQUESTED = "REQUESTED"    #: accepted for processing
    BLOCKED = "BLOCKED"        #: stopped before any broker call
    SIMULATED = "SIMULATED"    #: dry-run terminal stage
    CHECKED = "CHECKED"        #: broker order_check stage reached
    SENT = "SENT"              #: order_send returned a verdict
    VERIFIED = "VERIFIED"      #: MT5 state independently confirmed


class ExecutionRequest(_FiniteModel):
    """The strongly validated execution intent for ONE approved proposal.

    Built via :meth:`from_proposal` — the constructor is deliberately not
    the friendly path.  Carries no hidden strategy information: only what
    a broker needs plus the correlation chain back to the decision.
    """

    symbol: str = Field(min_length=1)
    direction: Direction  # LONG / SHORT only (NEUTRAL rejected)
    volume: float = Field(gt=0)
    #: indicative entry price (the proposal's validated entry); the actual
    #: market fill may differ within `deviation_points`
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    deviation_points: int = Field(default=DEFAULT_DEVIATION_POINTS, ge=0, le=10_000)
    #: MT5 filling mode (ORDER_FILLING_*).  None = let the terminal default
    #: apply; if the broker rejects with INVALID_FILL the operator must set
    #: it explicitly (no automatic retry, spec §42).
    type_filling: int | None = None

    # ---- correlation chain (spec §38) ---------------------------------
    proposal_id: str = Field(min_length=1)
    risk_decision_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    #: deterministic derived id (no uuid/random/clock)
    request_id: str = Field(min_length=1)

    comment: str = Field(default="", max_length=31)
    magic: int = Field(default=AURUMX_MAGIC)

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("symbol must not be empty")
        return cleaned

    @field_validator("direction")
    @classmethod
    def _executable_direction(cls, value: Direction) -> Direction:
        if value is Direction.NEUTRAL:
            raise ValueError("direction NEUTRAL is not executable")
        return value

    @classmethod
    def from_proposal(
        cls,
        proposal: TradeProposal,
        *,
        risk_decision_id: str,
        deviation_points: int = DEFAULT_DEVIATION_POINTS,
        type_filling: int | None = None,
        magic: int = AURUMX_MAGIC,
    ) -> ExecutionRequest:
        """Derive the request VERBATIM from the proposal (spec §12/§16).

        Symbol, direction, volume, entry, SL and TP are copied exactly —
        any inconsistency between the proposal and the request is a bug,
        and the service re-verifies the identity chain against the risk
        decision before executing.
        """
        if not risk_decision_id:
            raise ValueError("risk_decision_id is required (approval must be explicit)")
        if proposal.direction not in (AgentDirection.BUY, AgentDirection.SELL):
            raise ValueError(f"proposal direction {proposal.direction} is not executable")
        direction = Direction.LONG if proposal.direction is AgentDirection.BUY else Direction.SHORT
        return cls(
            symbol=proposal.symbol,
            direction=direction,
            volume=proposal.suggested_volume,
            entry_price=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            deviation_points=deviation_points,
            type_filling=type_filling,
            proposal_id=proposal.decision_id,
            risk_decision_id=risk_decision_id,
            fingerprint=proposal.fingerprint,
            request_id=derive_request_id(
                proposal_id=proposal.decision_id,
                risk_decision_id=risk_decision_id,
                fingerprint=proposal.fingerprint,
                symbol=proposal.symbol,
                direction=direction.value,
                volume=proposal.suggested_volume,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
            ),
            comment=f"AURUMX {proposal.decision_id[:12]}",
            magic=magic,
        )

    def matches_proposal(self, proposal: TradeProposal) -> list[str]:
        """Spec §16 consistency: symbol/direction/volume/SL/TP/identity may
        not change between proposal and request.  Returns violations."""
        expected_direction = (
            Direction.LONG if proposal.direction is AgentDirection.BUY else Direction.SHORT
        )
        issues: list[str] = []
        if self.symbol != proposal.symbol:
            issues.append(f"symbol changed: {self.symbol} != {proposal.symbol}")
        if self.direction is not expected_direction:
            issues.append(f"direction changed: {self.direction} != {expected_direction}")
        if self.volume != proposal.suggested_volume:
            issues.append(f"volume changed: {self.volume} != {proposal.suggested_volume}")
        if self.entry_price != proposal.entry_price:
            issues.append(f"entry changed: {self.entry_price} != {proposal.entry_price}")
        if self.stop_loss != proposal.stop_loss:
            issues.append(f"stop_loss changed: {self.stop_loss} != {proposal.stop_loss}")
        if self.take_profit != proposal.take_profit:
            issues.append(f"take_profit changed: {self.take_profit} != {proposal.take_profit}")
        if self.proposal_id != proposal.decision_id:
            issues.append("proposal identity changed")
        return issues


class ExecutionResult(_FiniteModel):
    """What actually happened (structured, never optimistic)."""

    request_id: str
    proposal_id: str
    risk_decision_id: str
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    mode: TradingMode
    status: ExecutionStatus
    stage: ExecutionStage

    # ---- broker verdict -------------------------------------------------
    retcode: int | None = None
    retcode_description: str = ""
    category: str = ""

    # ---- correlation chain continuation (None = broker did not supply) --
    order_ticket: int | None = None
    deal_ticket: int | None = None
    position_ticket: int | None = None

    # ---- economics -------------------------------------------------------
    fill_price: float | None = None
    filled_volume: float | None = None

    message: str = ""
    reasons: list[str] = Field(default_factory=list)
    timestamp: str = ""  # ISO-8601 UTC, stamped by the service clock

    @property
    def ok(self) -> bool:
        """Success = verified fill (or a clean dry-run simulation)."""
        return self.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED) or (
            self.status is ExecutionStatus.DRY_RUN
        )

    @property
    def blocked(self) -> bool:
        return self.status is ExecutionStatus.NOT_ATTEMPTED

    @property
    def order_was_sent(self) -> bool:
        """True only if order_send was actually invoked for this request."""
        return self.stage in (ExecutionStage.SENT, ExecutionStage.VERIFIED) and (
            self.status
            not in (ExecutionStatus.NOT_ATTEMPTED, ExecutionStatus.CHECK_FAILED, ExecutionStatus.DRY_RUN)
        )

    def summary(self) -> dict:
        """JSON-safe view (journal / API / logs — no secrets by construction)."""
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "risk_decision_id": self.risk_decision_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "volume": self.volume,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "mode": self.mode.value,
            "status": self.status.value,
            "stage": self.stage.value,
            "retcode": self.retcode,
            "retcode_description": self.retcode_description,
            "category": self.category,
            "order_ticket": self.order_ticket,
            "deal_ticket": self.deal_ticket,
            "position_ticket": self.position_ticket,
            "fill_price": self.fill_price,
            "filled_volume": self.filled_volume,
            "message": self.message,
            "reasons": list(self.reasons),
            "timestamp": self.timestamp,
        }


def derive_request_id(
    *,
    proposal_id: str,
    risk_decision_id: str,
    fingerprint: str,
    symbol: str,
    direction: str,
    volume: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> str:
    """Deterministic request id (mirrors the Phase-4 gate id style:
    sha256 over the canonical inputs, truncated)."""
    canonical = (
        f"{proposal_id}|{risk_decision_id}|{fingerprint}|{symbol}|{direction}"
        f"|{volume!r}|{entry_price!r}|{stop_loss!r}|{take_profit!r}"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


__all__ = [
    "AURUMX_MAGIC",
    "DEFAULT_DEVIATION_POINTS",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStage",
    "ExecutionStatus",
    "derive_request_id",
]
