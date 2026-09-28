"""ExecutionService — the controlled bridge from approval to broker (5B/5C).

Contract (spec §11/§6)::

    execute(execution_request, approved_risk_decision) -> ExecutionResult

Nothing else in the system can reach a broker execution method: the
service itself is the only caller of ``BrokerInterface.place_market_order``
(and only after its own fail-closed gates).  ``order_check``/``order_send``
themselves live *inside* the MT5 adapter, nowhere else.

Pipeline (identical for DRY_RUN and MT5 — spec §13; the ONLY difference is
that the dry-run path stops before touching the broker's execution API):

    1. risk-decision verification   (APPROVED + identity chain, else BLOCK)
    2. mode gate                    (trading_enabled, reconciliation halt)
    3. local validation             (gold symbol, broker spec, volume,
                                      geometry — independent of the Risk
                                      Gate: defense in depth, spec §9)
    4a. DRY_RUN: simulated result   (no order_check/order_send, no
                                      fabricated tickets, journaled DRY_RUN)
    4b. MT5: account verification (DEMO only — real accounts are blocked
                                      unconditionally) -> adapter (check ->
                                      send -> retcode verdict) -> state
                                      verification (position must exist for
                                      FILLED) -> journal

Fail-closed everywhere (spec §21): unknown state is never success.  There
are NO automatic retries (spec §42) — one execution attempt per call.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

from app.brokers.interface import BrokerInterface
from app.core.enums import Direction, RiskAction, TradingMode
from app.core.exceptions import BrokerError
from app.core.logging import get_logger
from app.core.models import MarketOrderRequest, OrderResult
from app.execution.contracts import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStage,
    ExecutionStatus,
)
from app.execution.events import (
    EXECUTION_CHECKED,
    EXECUTION_FAILED,
    EXECUTION_FILLED,
    EXECUTION_REJECTED,
    EXECUTION_REQUESTED,
    EXECUTION_SENT,
    EXECUTION_SIMULATED,
    EXECUTION_UNKNOWN,
)
from app.execution.journal import ExecutionRecord, InMemoryExecutionJournal
from app.market.symbol_discovery import is_gold_symbol
from app.risk import RiskDecision

if TYPE_CHECKING:  # pragma: no cover
    from app.core.events import EventBus

log = get_logger("execution.service")

#: rel-tolerance for tick-grid alignment (mirrors the Phase-4 gate)
_GRID_REL_TOL = 1e-6


class ReconciliationGuardProtocol(Protocol):
    """Minimal surface the service needs from the reconciliation layer."""

    @property
    def execution_allowed(self) -> bool: ...


class ExecutionServiceConfig(BaseModel):
    """Safe defaults: nothing executes unless explicitly enabled."""

    model_config = ConfigDict(allow_inf_nan=False)

    trading_enabled: bool = False
    dry_run: bool = True
    deviation_points: int = 20
    type_filling: int | None = None
    #: verification tolerance for SL/TP: broker-side rounding to the tick
    #: grid is legitimate; anything larger is a mismatch (documented §24)
    sl_tp_tolerance_ticks: float = 1.0
    #: verification tolerance for fill price: the requested deviation plus
    #: a small buffer — a fill outside it is flagged, not trusted
    fill_slippage_ticks: float = 50.0


def _on_grid(value: float, tick_size: float) -> bool:
    if tick_size <= 0:
        return False
    return abs(value / tick_size - round(value / tick_size)) <= _GRID_REL_TOL


class ExecutionService:
    """The ONLY caller of broker execution methods."""

    name = "execution"

    def __init__(
        self,
        broker: BrokerInterface,
        *,
        config: ExecutionServiceConfig | None = None,
        journal=None,
        event_bus: EventBus | None = None,
        clock: Callable[[], datetime] | None = None,
        reconciliation_guard: ReconciliationGuardProtocol | None = None,
    ) -> None:
        self._broker = broker
        self._config = config or ExecutionServiceConfig()
        self._journal = journal if journal is not None else InMemoryExecutionJournal()
        self._bus = event_bus
        self._clock = clock or (lambda: datetime.now(UTC))
        self._guard = reconciliation_guard

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> ExecutionServiceConfig:
        return self._config

    @property
    def journal(self):
        return self._journal

    @property
    def mode(self) -> TradingMode:
        """The operating mode derived from the safe config (spec §37)."""
        if not self._config.trading_enabled:
            return TradingMode.READ_ONLY
        if self._config.dry_run:
            return TradingMode.DRY_RUN
        # Non-dry-run requires the config-level confirmation phrase
        # (enforced by AppConfig) — and this layer still verifies the
        # account is DEMO at execution time; real accounts never pass.
        return TradingMode.MT5_DEMO

    def force_dry_run(self) -> None:
        """Control-plane safety ratchet: switch to DRY_RUN (one-way — the
        control plane may only DE-escalate; re-enabling demo execution
        requires an explicit configuration change + restart)."""
        if not self._config.dry_run:
            self._config.dry_run = True
            log.warning(
                "execution forced into DRY_RUN by control plane",
                event="EXECUTION_DRY_RUN_FORCED",
            )

    # ------------------------------------------------------------------
    # the execution entry point
    # ------------------------------------------------------------------
    def execute(
        self, request: ExecutionRequest, decision: RiskDecision | None
    ) -> ExecutionResult:
        """Execute ONE approved request.  Never raises for controlled
        failures — the outcome is always a structured ExecutionResult."""
        self._emit(EXECUTION_REQUESTED, request)
        now = self._clock()

        # Idempotency is checked before every other gate.  The journal is the
        # authority, so this also survives process restarts.
        prior = self._journal.by_request_id(request.request_id)
        if prior is not None:
            duplicate = prior.model_copy(update={
                "status": ExecutionStatus.NOT_ATTEMPTED,
                "message": "duplicate_request: request_id has already been attempted",
                "reasons": ["duplicate_request"],
                "timestamp": now,
            })
            return self._finish(request, duplicate, now, event=EXECUTION_REJECTED)

        # ---- 1. risk-decision verification (spec §11) ---------------------
        reason = self._verify_decision(request, decision)
        if reason is not None:
            return self._finish(
                request, self._blocked(request, reason), now,
                event=EXECUTION_REJECTED,
            )

        # ---- 2. mode gates -------------------------------------------------
        if not self._config.trading_enabled:
            return self._finish(
                request, self._blocked(request, "trading_disabled: TRADING_ENABLED is false"),
                now, event=EXECUTION_REJECTED,
            )
        if self._guard is not None and not self._guard.execution_allowed:
            return self._finish(
                request,
                self._blocked(
                    request,
                    "reconciliation_halt: unresolved reconciliation mismatch — "
                    "execution blocked until the state is resolved",
                ),
                now, event=EXECUTION_REJECTED,
            )

        # ---- 3. local validation (defense in depth, spec §9/§10) ----------
        try:
            reasons = self._validate_request(request)
        except BrokerError as exc:
            return self._finish(
                request, self._blocked(request, f"mt5_unavailable: {exc}"), now,
                event=EXECUTION_FAILED,
            )
        if reasons:
            return self._finish(
                request, self._blocked(request, "; ".join(reasons)), now,
                event=EXECUTION_REJECTED,
            )

        # ---- 4a. DRY_RUN: same pipeline, simulated terminal -----------------
        if self._config.dry_run:
            result = ExecutionResult(
                request_id=request.request_id,
                proposal_id=request.proposal_id,
                risk_decision_id=request.risk_decision_id,
                symbol=request.symbol,
                direction=request.direction,
                volume=request.volume,
                entry_price=request.entry_price,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                mode=TradingMode.DRY_RUN,
                status=ExecutionStatus.DRY_RUN,
                stage=ExecutionStage.SIMULATED,
                fill_price=request.entry_price,
                filled_volume=request.volume,
                message="dry-run: full pipeline executed, no order was sent",
            )
            return self._finish(request, result, now, event=EXECUTION_SIMULATED)

        # ---- 4b. MT5 path (5C): demo-guarded real execution ----------------
        return self._execute_mt5(request, now)

    # ------------------------------------------------------------------
    # stage 1: approval verification
    # ------------------------------------------------------------------
    @staticmethod
    def _verify_decision(
        request: ExecutionRequest, decision: RiskDecision | None
    ) -> str | None:
        if decision is None:
            return "risk_decision_missing: an APPROVED RiskDecision is required"
        if decision.action is not RiskAction.APPROVED:
            return (
                f"risk_decision_not_approved: gate decision was "
                f"{decision.action.value}"
                + (f" ({'; '.join(decision.reasons[:3])})" if decision.reasons else "")
            )
        if not decision.gate_decision_id:
            return "risk_decision_id_missing: the decision carries no gate id"
        if decision.gate_decision_id != request.risk_decision_id:
            return "risk_decision_id_mismatch: request does not reference this decision"
        if decision.proposal_id != request.proposal_id:
            return (
                f"proposal_id_mismatch: decision approved {decision.proposal_id!r}, "
                f"request references {request.proposal_id!r}"
            )
        if decision.fingerprint and decision.fingerprint != request.fingerprint:
            return "fingerprint_mismatch: request does not match the approved proposal"
        return None

    # ------------------------------------------------------------------
    # stage 3: independent local validation (never trust, never repair)
    # ------------------------------------------------------------------
    def _validate_request(self, request: ExecutionRequest) -> list[str]:
        reasons: list[str] = []

        # gold-only protection: reuse Phase-1, no second implementation
        if not is_gold_symbol(request.symbol):
            reasons.append(f"symbol_not_gold: {request.symbol} is not a gold symbol")

        spec = self._broker.get_symbol(request.symbol)
        if spec is None:
            reasons.append(f"symbol_unknown: no broker metadata for {request.symbol}")
            return reasons  # nothing else can be checked without a spec

        report = spec.validate()
        if not report.ok:
            reasons.append(f"symbol_metadata_invalid: {report.summary()}")
            return reasons

        # ---- volume vs the broker's actual constraints (spec §9) ----------
        volume = request.volume
        if volume < spec.volume_min:
            reasons.append(f"volume_below_minimum: {volume} < volume_min {spec.volume_min}")
        if volume > spec.volume_max:
            reasons.append(f"volume_above_maximum: {volume} > volume_max {spec.volume_max}")
        if spec.volume_step > 0 and not _on_grid(volume - spec.volume_min, spec.volume_step):
            reasons.append(
                f"volume_off_step: {volume} is not aligned to step {spec.volume_step}"
            )

        # ---- geometry (spec §10): reject, never repair --------------------
        entry, sl, tp = request.entry_price, request.stop_loss, request.take_profit
        for name, value in (("entry", entry), ("sl", sl), ("tp", tp)):
            if not math.isfinite(value) or value <= 0:
                reasons.append(f"{name}_invalid: {value}")
            elif not _on_grid(value, spec.tick_size):
                reasons.append(f"{name}_off_tick_grid: {value} (tick {spec.tick_size})")
        if reasons:
            return reasons

        if request.direction is Direction.LONG:
            if not sl < entry < tp:
                reasons.append(f"geometry_invalid: BUY requires SL < entry < TP (sl={sl}, entry={entry}, tp={tp})")
        else:
            if not tp < entry < sl:
                reasons.append(f"geometry_invalid: SELL requires TP < entry < SL (sl={sl}, entry={entry}, tp={tp})")

        min_distance = max(spec.stops_level_points, spec.freeze_level_points) * spec.point
        if min_distance > 0:
            if abs(entry - sl) < min_distance:
                reasons.append(
                    f"stop_loss_too_close: |entry-sl| {abs(entry - sl):.4f} < broker minimum {min_distance:.4f}"
                )
            if abs(entry - tp) < min_distance:
                reasons.append(
                    f"take_profit_too_close: |entry-tp| {abs(entry - tp):.4f} < broker minimum {min_distance:.4f}"
                )
        return reasons

    # ------------------------------------------------------------------
    # stage 4b: MT5 execution (DEMO only)
    # ------------------------------------------------------------------
    def _execute_mt5(self, request: ExecutionRequest, now: datetime) -> ExecutionResult:
        # ---- account verification: never infer DEMO from config (§31) -----
        try:
            account = self._broker.get_account()
        except BrokerError as exc:
            return self._finish(
                request, self._blocked(request, f"mt5_unavailable: {exc}"), now,
                event=EXECUTION_FAILED,
            )
        if account.is_real:
            return self._finish(
                request,
                self._blocked(
                    request,
                    f"real_account_blocked: account {account.login} is REAL — "
                    "AURUMX demo execution refuses real accounts",
                ),
                now, event=EXECUTION_REJECTED,
            )
        if not account.is_demo:
            return self._finish(
                request,
                self._blocked(
                    request,
                    f"account_type_unknown: trade mode {account.trade_mode.value} is not "
                    "verified DEMO — failing closed",
                ),
                now, event=EXECUTION_REJECTED,
            )

        # ---- adapter call: the ONLY broker execution call site ------------
        broker_request = MarketOrderRequest(
            symbol=request.symbol,
            direction=request.direction,
            volume=request.volume,
            sl=request.stop_loss,
            tp=request.take_profit,
            deviation_points=request.deviation_points,
            type_filling=request.type_filling,
            comment=request.comment,
            magic=request.magic,
        )
        try:
            order_result = self._broker.place_market_order(broker_request)
        except Exception as exc:  # noqa: BLE001 - fail closed (spec §21)
            return self._finish(
                request,
                ExecutionResult(
                    **self._common_fields(request),
                    mode=self.mode,
                    status=ExecutionStatus.SEND_FAILED,
                    stage=ExecutionStage.SENT,
                    message=f"broker call raised: {exc}",
                    reasons=["broker_exception"],
                ),
                now, event=EXECUTION_FAILED,
            )

        if order_result.phase == "send":
            # order_check passed and order_send was reached
            self._emit(EXECUTION_CHECKED, request)
            self._emit(EXECUTION_SENT, request)

        result = self._map_broker_result(request, order_result, now)
        return result

    # ------------------------------------------------------------------
    # verdict mapping + state verification
    # ------------------------------------------------------------------
    def _map_broker_result(
        self, request: ExecutionRequest, order: OrderResult, now: datetime
    ) -> ExecutionResult:
        common = self._common_fields(request)

        if order.accepted:
            verified, detail, position_ticket = self._verify_fill(request, order)
            filled_volume = order.volume if order.volume and order.volume > 0 else request.volume
            partial = (
                order.category == "partial"
                or filled_volume < request.volume - self._volume_tolerance(request)
            )
            if partial:
                status = ExecutionStatus.PARTIALLY_FILLED if verified else ExecutionStatus.UNKNOWN
                stage = ExecutionStage.VERIFIED if verified else ExecutionStage.SENT
            else:
                status = ExecutionStatus.FILLED if verified else ExecutionStatus.UNKNOWN
                stage = ExecutionStage.VERIFIED if verified else ExecutionStage.SENT
            if not verified:
                # accepted but unverified is NOT a fill (spec §19/§20)
                event = EXECUTION_UNKNOWN
            else:
                event = EXECUTION_FILLED
            return self._finish(
                request,
                ExecutionResult(
                    **common,
                    mode=self.mode,
                    status=status,
                    stage=stage,
                    retcode=order.retcode,
                    retcode_description=order.retcode_description,
                    category=order.category,
                    order_ticket=order.ticket,
                    deal_ticket=order.deal_ticket,
                    position_ticket=position_ticket,
                    fill_price=order.price,
                    filled_volume=filled_volume,
                    message=(f"{order.message}; state verification: {detail}"),
                    reasons=[] if verified else [f"unverified: {detail}"],
                ),
                now, event=event,
            )

        # ---- not accepted: classify by phase + category --------------------
        if order.phase == "check":
            status, stage = ExecutionStatus.CHECK_FAILED, ExecutionStage.CHECKED
            event = EXECUTION_FAILED
        elif order.category == "error":
            # the send attempt itself errored (spec §41: broker exception
            # -> FAILED).  No retry: the first attempt may have succeeded.
            status, stage = ExecutionStatus.SEND_FAILED, ExecutionStage.SENT
            event = EXECUTION_FAILED
        elif order.category in ("timeout", "unknown", "connection", "placed_unconfirmed"):
            status, stage = ExecutionStatus.UNKNOWN, ExecutionStage.SENT
            event = EXECUTION_UNKNOWN
        else:
            status, stage = ExecutionStatus.REJECTED_BY_BROKER, ExecutionStage.SENT
            event = EXECUTION_REJECTED
        return self._finish(
            request,
            ExecutionResult(
                **common,
                mode=self.mode,
                status=status,
                stage=stage,
                retcode=order.retcode,
                retcode_description=order.retcode_description,
                category=order.category,
                message=order.message,
                reasons=[order.category or "broker_rejected"],
            ),
            now, event=event,
        )

    def _verify_fill(
        self, request: ExecutionRequest, order: OrderResult
    ) -> tuple[bool, str, int | None]:
        """Confirm the executed trade actually exists in MT5 state (§19).
        FILLED is only ever recorded on the strength of this check (§20)."""
        try:
            positions = self._broker.get_positions(request.symbol)
            spec = self._broker.get_symbol(request.symbol)
        except BrokerError as exc:
            return False, f"state check unavailable: {exc}", None

        own = [p for p in positions if p.magic == request.magic and p.symbol == request.symbol]
        if not own:
            return False, "no open position with the AURUMX magic number", None

        match = None
        if order.ticket is not None:
            match = next((p for p in own if p.ticket == order.ticket), None)
        if match is None:
            # fallback (netting accounts aggregate positions, so the ticket
            # may not match): same direction and volume within one step
            volume_tolerance = self._volume_tolerance(request)
            match = next(
                (
                    p for p in own
                    if p.direction is request.direction
                    and abs(p.volume - request.volume) <= volume_tolerance
                ),
                None,
            )

        if match is None:
            return False, "no position matches the executed order", None

        tick = spec.tick_size if spec else 0.01
        volume_tolerance = self._volume_tolerance(request)
        # a partial fill legitimately reports less than requested — verify
        # against what the broker says was filled, not what was asked for
        filled = order.volume if order.volume and order.volume > 0 else request.volume
        diffs: list[str] = []
        if match.direction is not request.direction:
            diffs.append(f"direction {match.direction.value}")
        if abs(match.volume - filled) > volume_tolerance:
            diffs.append(f"volume {match.volume} vs filled {filled}")
        sl_tp_tolerance = self._config.sl_tp_tolerance_ticks * tick + 1e-9
        if request.stop_loss and (
            match.price_sl is None or abs(match.price_sl - request.stop_loss) > sl_tp_tolerance
        ):
            diffs.append(f"stop_loss {match.price_sl} vs requested {request.stop_loss}")
        if request.take_profit and (
            match.price_tp is None or abs(match.price_tp - request.take_profit) > sl_tp_tolerance
        ):
            diffs.append(f"take_profit {match.price_tp} vs requested {request.take_profit}")

        if diffs:
            return False, "position mismatch: " + "; ".join(diffs), match.ticket
        return True, "position verified against broker state", match.ticket

    def _volume_tolerance(self, request: ExecutionRequest) -> float:
        try:
            spec = self._broker.get_symbol(request.symbol)
        except BrokerError:
            return 0.0
        return (spec.volume_step if spec else 0.0) or 0.0

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _common_fields(request: ExecutionRequest) -> dict:
        return dict(
            request_id=request.request_id,
            proposal_id=request.proposal_id,
            risk_decision_id=request.risk_decision_id,
            symbol=request.symbol,
            direction=request.direction,
            volume=request.volume,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
        )

    def _blocked(self, request: ExecutionRequest, reason: str) -> ExecutionResult:
        return ExecutionResult(
            **self._common_fields(request),
            mode=self.mode,  # the mode in force (nothing was attempted)
            status=ExecutionStatus.NOT_ATTEMPTED,
            stage=ExecutionStage.BLOCKED,
            message=reason,
            reasons=[reason.split(":")[0]],
        )

    def _finish(
        self, request: ExecutionRequest, result: ExecutionResult,
        now: datetime, *, event: str,
    ) -> ExecutionResult:
        stamped = result.model_copy(update={"timestamp": now.isoformat()})
        self._journal.record(ExecutionRecord.from_result(stamped, timestamp=now))
        self._emit(event, request, result=stamped)
        log.info(
            "execution attempt finished",
            event=f"EXECUTION_{stamped.status.value}",
            request_id=stamped.request_id,
            proposal_id=stamped.proposal_id,
            risk_decision_id=stamped.risk_decision_id,
            status=stamped.status.value,
            stage=stamped.stage.value,
            retcode=stamped.retcode,
            order_ticket=stamped.order_ticket,
            deal_ticket=stamped.deal_ticket,
            position_ticket=stamped.position_ticket,
            mode=stamped.mode.value,
            dry_run=stamped.mode is TradingMode.DRY_RUN,
        )
        return stamped

    def _emit(self, event_type: str, request: ExecutionRequest, **extra) -> None:
        if self._bus is None:
            return
        payload: dict = {
            "request_id": request.request_id,
            "proposal_id": request.proposal_id,
            "risk_decision_id": request.risk_decision_id,
            "symbol": request.symbol,
        }
        result = extra.pop("result", None)
        if result is not None:
            payload["result"] = result.summary()
        payload.update(extra)
        self._bus.emit(event_type, "execution", **payload)


__all__ = ["ExecutionService", "ExecutionServiceConfig"]
