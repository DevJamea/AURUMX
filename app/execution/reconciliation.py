"""Reconciliation (Phase-5 spec §22-§26): AURUMX journal vs actual MT5 state.

The reconciler compares what the execution journal claims against what the
broker actually shows (open positions carrying the AURUMX magic number):

    MATCHED           journal record and MT5 position agree within the
                      documented tolerances
    MISSING_IN_MT5    journal claims a (verified) fill but no corresponding
                      position exists
    MISSING_IN_JOURNAL an AURUMX-magic position exists that no journal
                      record accounts for
    MISMATCH          both sides exist but disagree beyond tolerance
    UNKNOWN           evidence is insufficient to classify

**Documented tolerances (spec §24):**

* direction — exact match, no tolerance;
* magic     — exact match, no tolerance (positions without the AURUMX
              magic number are ignored entirely — unrelated activity);
* volume    — within one ``volume_step`` (representation differences);
* SL / TP   — within one tick (``tick_size``): brokers legitimately round
              protective levels to the grid when applying them;
* entry     — within ``entry_slippage_ticks`` × tick (default 50 ticks):
              market fills legitimately differ from the indicative price;
              anything larger is a mismatch, not slippage.

The reconciler NEVER repairs state, never invents journal entries and never
adopts unknown MT5 positions (spec §25).  A non-clean report engages the
``ReconciliationGuard``, which blocks further execution until the operator
explicitly resolves/re-runs reconciliation (audited) — never silently
(spec §26).

Known limitation (documented): Phase 5 reconciles against *current* state
(open positions).  A position that was closed normally (SL/TP hit) no
longer appears and its journal record reads MISSING_IN_MT5; resolution
requires the explicit operator acknowledgement (deal history arrives with
a later phase).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.brokers.interface import BrokerInterface
from app.core.enums import TradingMode
from app.core.exceptions import BrokerError
from app.core.logging import get_logger
from app.core.models import Position
from app.execution.contracts import AURUMX_MAGIC, ExecutionStatus
from app.execution.events import RECONCILIATION_MATCHED, RECONCILIATION_MISMATCH
from app.execution.journal import ExecutionJournal, ExecutionRecord

log = get_logger("execution.reconciliation")

#: journal statuses that should correspond to something in MT5
_SENT_STATUSES = (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED)
#: statuses that were sent but never confirmed — absence is inconclusive
_UNCONFIRMED_STATUSES = (ExecutionStatus.ACCEPTED, ExecutionStatus.UNKNOWN)


class ReconciliationStatus(StrEnum):
    MATCHED = "MATCHED"
    MISSING_IN_MT5 = "MISSING_IN_MT5"
    MISSING_IN_JOURNAL = "MISSING_IN_JOURNAL"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class ExecutionComparison(BaseModel):
    """One journal-record-vs-broker-state comparison."""

    model_config = ConfigDict(allow_inf_nan=False)

    request_id: str | None = None
    proposal_id: str | None = None
    journal_status: ExecutionStatus | None = None
    position_ticket: int | None = None
    status: ReconciliationStatus
    differences: list[str] = Field(default_factory=list)
    detail: str = ""

    def summary(self) -> dict:
        return self.model_dump(mode="json")


class ReconciliationReport(BaseModel):
    """The full comparison result + counts."""

    model_config = ConfigDict(allow_inf_nan=False)

    generated_at: datetime
    comparisons: list[ExecutionComparison] = Field(default_factory=list)
    #: journal records not compared (dry-run / blocked — nothing in MT5)
    skipped: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return bool(self.comparisons) and all(
            c.status is ReconciliationStatus.MATCHED for c in self.comparisons
        )

    @property
    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for comparison in self.comparisons:
            tally[comparison.status.value] = tally.get(comparison.status.value, 0) + 1
        return tally

    def summary(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "clean": self.clean,
            "counts": self.counts,
            "comparisons": [c.summary() for c in self.comparisons],
            "skipped": list(self.skipped),
        }


class ReconciliationGuard:
    """Blocks execution while a reconciliation mismatch is unresolved
    (spec §26).  Resolution is explicit and audited — never silent."""

    def __init__(self) -> None:
        self._report: ReconciliationReport | None = None
        self._acknowledged: bool = False

    @property
    def report(self) -> ReconciliationReport | None:
        return self._report

    @property
    def status(self) -> str:
        if self._report is None:
            return "NOT_RUN"
        if self._report.clean:
            return "CLEAN"
        return "ACKNOWLEDGED" if self._acknowledged else "MISMATCH"

    @property
    def execution_allowed(self) -> bool:
        if self._report is None:
            return True  # nothing reconciled yet — normal pre-trade state
        return self._report.clean or self._acknowledged

    def record(self, report: ReconciliationReport) -> None:
        self._report = report
        self._acknowledged = False

    def acknowledge(self) -> bool:
        """Operator acknowledgement of a non-clean report (audited by the
        caller).  Returns whether the acknowledgement took effect."""
        if self._report is not None and not self._report.clean:
            self._acknowledged = True
            return True
        return False


class Reconciler:
    """Compares the execution journal against actual broker state."""

    def __init__(
        self,
        broker: BrokerInterface,
        journal: ExecutionJournal,
        *,
        entry_slippage_ticks: float = 50.0,
        magic: int = AURUMX_MAGIC,
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._entry_slippage_ticks = entry_slippage_ticks
        self._magic = magic

    def reconcile(
        self,
        *,
        now: datetime | None = None,
        event_bus=None,
        clock: Callable[[], datetime] | None = None,
    ) -> ReconciliationReport:
        stamp = now or (clock() if clock else datetime.now(UTC))
        comparisons: list[ExecutionComparison] = []
        skipped: list[str] = []

        try:
            broker_positions = self._broker.get_positions()
        except BrokerError as exc:
            return ReconciliationReport(
                generated_at=stamp,
                comparisons=[
                    ExecutionComparison(
                        status=ReconciliationStatus.UNKNOWN,
                        detail=f"broker state unavailable: {exc}",
                    )
                ],
            )

        own_positions = [p for p in broker_positions if p.magic == self._magic]
        claimed_tickets: set[int] = set()

        for record in self._journal.recent(limit=10_000)[::-1]:
            if record.mode is TradingMode.DRY_RUN:
                skipped.append(f"{record.request_id} (dry-run)")
                continue
            if record.status not in (*_SENT_STATUSES, *_UNCONFIRMED_STATUSES):
                skipped.append(f"{record.request_id} ({record.status.value})")
                continue
            comparisons.append(
                self._compare_record(record, own_positions, claimed_tickets)
            )

        # reverse direction: AURUMX positions the journal cannot explain
        for position in own_positions:
            if position.ticket not in claimed_tickets:
                comparisons.append(
                    ExecutionComparison(
                        status=ReconciliationStatus.MISSING_IN_JOURNAL,
                        position_ticket=position.ticket,
                        differences=[
                            f"MT5 position {position.ticket} ({position.symbol} "
                            f"{position.direction.value} {position.volume}) has no "
                            "journal record"
                        ],
                        detail="position not adopted — operator review required",
                    )
                )

        report = ReconciliationReport(
            generated_at=stamp, comparisons=comparisons, skipped=skipped
        )
        if event_bus is not None:
            event = RECONCILIATION_MATCHED if report.clean else RECONCILIATION_MISMATCH
            event_bus.emit(
                event, "reconciliation", clean=report.clean, counts=report.counts
            )
        log.info(
            "reconciliation finished",
            event="RECONCILIATION_DONE",
            clean=report.clean,
            counts=report.counts,
        )
        return report

    # ------------------------------------------------------------------
    def _compare_record(
        self,
        record: ExecutionRecord,
        own_positions: list[Position],
        claimed_tickets: set[int],
    ) -> ExecutionComparison:
        candidates = [p for p in own_positions if p.symbol == record.symbol]

        match = None
        if record.position_ticket is not None:
            match = next((p for p in candidates if p.ticket == record.position_ticket), None)
        if match is None and record.order_ticket is not None:
            match = next((p for p in candidates if p.ticket == record.order_ticket), None)

        if match is None:
            if record.status in _SENT_STATUSES:
                return ExecutionComparison(
                    request_id=record.request_id,
                    proposal_id=record.proposal_id,
                    journal_status=record.status,
                    status=ReconciliationStatus.MISSING_IN_MT5,
                    differences=[
                        "journal records a verified fill but no corresponding "
                        "open position exists in MT5"
                    ],
                    detail=(
                        "position may have closed normally (SL/TP) — Phase 5 "
                        "reconciles current state only; requires operator review"
                    ),
                )
            # ACCEPTED/UNKNOWN journal records: absence is inconclusive
            return ExecutionComparison(
                request_id=record.request_id,
                proposal_id=record.proposal_id,
                journal_status=record.status,
                status=ReconciliationStatus.UNKNOWN,
                detail="journal record unconfirmed and no matching position found",
            )

        claimed_tickets.add(match.ticket)
        differences = self._field_differences(record, match)
        status = ReconciliationStatus.MISMATCH if differences else ReconciliationStatus.MATCHED
        return ExecutionComparison(
            request_id=record.request_id,
            proposal_id=record.proposal_id,
            journal_status=record.status,
            position_ticket=match.ticket,
            status=status,
            differences=differences,
        )

    def _field_differences(
        self, record: ExecutionRecord, position: Position
    ) -> list[str]:
        differences: list[str] = []
        expected_direction = record.direction
        if position.direction is not expected_direction:
            differences.append(
                f"direction: journal {expected_direction.value} vs MT5 {position.direction.value}"
            )

        spec = self._spec(record.symbol)
        volume_step = spec.volume_step if spec else 0.0
        if volume_step and abs(position.volume - record.volume) > volume_step:
            differences.append(f"volume: journal {record.volume} vs MT5 {position.volume}")

        tick = spec.tick_size if spec else 0.0
        if tick:
            # epsilon guards against float representation (0.010000000002 > 0.01)
            sl_tp_tolerance = tick + 1e-9  # one tick (documented)
            if record.stop_loss and (
                position.price_sl is None
                or abs(position.price_sl - record.stop_loss) > sl_tp_tolerance
            ):
                differences.append(
                    f"stop_loss: journal {record.stop_loss} vs MT5 {position.price_sl}"
                )
            if record.take_profit and (
                position.price_tp is None
                or abs(position.price_tp - record.take_profit) > sl_tp_tolerance
            ):
                differences.append(
                    f"take_profit: journal {record.take_profit} vs MT5 {position.price_tp}"
                )
            entry_tolerance = self._entry_slippage_ticks * tick + 1e-9
            if record.fill_price is not None and position.price_open and (
                abs(position.price_open - record.fill_price) > entry_tolerance
            ):
                differences.append(
                    f"entry: journal {record.fill_price} vs MT5 {position.price_open}"
                )
        return differences

    def _spec(self, symbol: str):
        try:
            return self._broker.get_symbol(symbol)
        except BrokerError:
            return None


__all__ = [
    "ExecutionComparison",
    "ReconciliationGuard",
    "ReconciliationReport",
    "ReconciliationStatus",
    "Reconciler",
]
