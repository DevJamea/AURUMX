"""EngineRuntime — the composition the control plane talks to (5E).

Wires the full stack with the production seams:

    MT5Broker (or any BrokerInterface)
      -> MarketDataService -> DecisionEngine -> HardRiskGate
      -> ExecutionService -> Reconciler (+ guard)
      -> journals (decision + execution) + EventBus

Safety properties:

* the runtime starts SAFE — whatever ``AppConfig`` says, the control
  plane's default is stopped + halts from the persistent stores;
* ``evaluate_cycle`` ANALYZES ONLY (journals the decision + the risk
  verdict).  It never executes;
* ``execute_approved`` executes at most the LAST gate-approved proposal,
  one order, and only when the engine is started, no operator halt is
  active, and the execution service's own gates pass (defense in depth);
* the account type shown by ``status()`` comes from the BROKER, never
  from configuration (spec §31).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from threading import RLock

from app.brokers.interface import BrokerInterface
from app.control.errors import EngineNotStartedError, ExecutionRefused
from app.control.state import EngineControl
from app.core.enums import SystemState
from app.core.events import EventBus
from app.core.exceptions import AurumXError, BrokerError
from app.core.logging import get_logger
from app.core.models import AccountSnapshot
from app.decision import DecisionEngine, DecisionEngineConfig, InMemoryDecisionJournal
from app.decision.journal import DecisionJournal, DecisionRecord
from app.execution import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ExecutionServiceConfig,
    InMemoryExecutionJournal,
    Reconciler,
    ReconciliationGuard,
    ReconciliationReport,
)
from app.execution.events import CONTROL_DRY_RUN_FORCED, CONTROL_RECONCILIATION_ACKNOWLEDGED
from app.execution.journal import ExecutionJournal
from app.market.data_service import MarketDataService
from app.risk import AccountState, HardRiskGate, RiskGateConfig, RiskState

log = get_logger("control.runtime")


class EngineRuntime:
    """Owns every component; the control API is a thin JSON veneer over it."""

    def __init__(
        self,
        config,
        broker: BrokerInterface,
        *,
        event_bus: EventBus | None = None,
        clock: Callable[[], datetime] | None = None,
        decision_journal: DecisionJournal | None = None,
        execution_journal: ExecutionJournal | None = None,
        gate: HardRiskGate | None = None,
        execution_service: ExecutionService | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.bus = event_bus or EventBus()
        self.clock = clock or (lambda: datetime.now(UTC))

        self.decision_journal = decision_journal or InMemoryDecisionJournal()
        self.execution_journal = execution_journal or InMemoryExecutionJournal()

        self.data_service = MarketDataService(
            broker,
            symbol=config.symbol,
            timeframes=list(config.timeframes),
            candle_count=config.candle_count,
            max_tick_age_seconds=config.max_tick_age_seconds,
            candle_freshness_multiplier=config.candle_freshness_multiplier,
            clock=self.clock,
        )
        self.engine = DecisionEngine(DecisionEngineConfig(), journal=self.decision_journal)

        if gate is not None:
            self.gate = gate
        else:
            gate_config = RiskGateConfig.from_app_config(config).model_copy(
                update={"max_total_exposure": config.max_total_exposure_usd}
            )
            self.gate = HardRiskGate(gate_config, symbol_specs={})

        self.reconciler = Reconciler(broker, self.execution_journal)
        self.guard = ReconciliationGuard()

        if execution_service is not None:
            self.execution_service = execution_service
        else:
            self.execution_service = ExecutionService(
                broker,
                config=ExecutionServiceConfig(
                    trading_enabled=config.trading_enabled,
                    dry_run=config.dry_run,
                ),
                journal=self.execution_journal,
                event_bus=self.bus,
                clock=self.clock,
                reconciliation_guard=self.guard,
            )

        self.control = EngineControl(event_bus=self.bus)

        self.system_state: SystemState = SystemState.STARTING
        self._last_decision: DecisionRecord | None = None
        self._last_proposal = None
        self._last_risk_decision = None
        self._symbol: str | None = None
        self._account: AccountSnapshot | None = None
        # ThreadingHTTPServer dispatches requests concurrently; all stateful
        # trading operations share one re-entrant critical section.
        self._operation_lock = RLock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> dict:
        """Connect the broker, resolve the symbol, register the verified
        symbol spec with the risk gate, and read the ACTUAL account type."""
        self.broker.connect()
        self.data_service.connect()
        spec = self.data_service.resolve_symbol()
        self._symbol = spec.name
        self.gate.register_symbol(spec)
        self._account = self._read_account()
        self.system_state = SystemState.CONNECTED
        log.info(
            "runtime connected",
            event="RUNTIME_CONNECTED",
            symbol=self._symbol,
            account_type=self._account.trade_mode.value if self._account else "unknown",
        )
        return self.status()

    def _read_account(self) -> AccountSnapshot | None:
        try:
            return self.broker.get_account()
        except BrokerError:
            return None

    # ------------------------------------------------------------------
    # evidence builders (the honest bridge into the Phase-4 gate inputs)
    # ------------------------------------------------------------------
    def _risk_inputs(self) -> tuple[RiskState, AccountState]:
        account = self._read_account()
        if account is None:
            raise BrokerError("no account information available — cannot build evidence")

        positions = self.broker.get_positions(self._symbol)
        orders = self.broker.get_orders(self._symbol)
        tick = self.broker.get_tick(self._symbol)

        spec = self.broker.get_symbol(self._symbol) if self._symbol else None
        notional = 0.0
        if spec is not None:
            for position in positions:
                notional += (
                    position.volume
                    * (position.price_open / spec.tick_size)
                    * spec.tick_value
                )

        spread_points = None
        if tick is not None and spec is not None and spec.point > 0:
            spread_points = (tick.ask - tick.bid) / spec.point

        trade_allowed = self.broker.is_trading_allowed()
        halts = self.control.halt_flags()

        risk_state = RiskState(
            equity=account.equity,
            # NOTE: no loss accumulators exist yet (Phase 6 owns trade
            # history); 0 = "no losses recorded", documented limitation.
            daily_loss=0.0,
            consecutive_losses=0,
            open_positions=len(positions),
            pending_orders=len(orders),
        )
        account_state = AccountState(
            equity=account.equity,
            balance=account.balance,
            margin=account.margin,
            margin_free=account.margin_free,
            margin_level=account.margin_level,
            leverage=account.leverage,
            open_positions=len(positions),
            pending_orders=len(orders),
            open_positions_notional=notional,
            spread_points=spread_points,
            trade_allowed=bool(trade_allowed),  # None -> False (fail closed)
            **halts,
        )
        return risk_state, account_state

    # ------------------------------------------------------------------
    # the analysis cycle (never executes)
    # ------------------------------------------------------------------
    def evaluate_cycle(self) -> dict:
        with self._operation_lock:
            return self._evaluate_cycle_locked()

    def _evaluate_cycle_locked(self) -> dict:
        if not self.control.started:
            raise EngineNotStartedError("engine is stopped — POST /control/start first")

        self.system_state = SystemState.ANALYZING
        try:
            snapshot = self.data_service.get_snapshot()
            risk_state, account_state = self._risk_inputs()

            self.system_state = SystemState.RISK_CHECK
            decision = self.engine.evaluate(
                snapshot, risk_state=risk_state, now=self.clock()
            )
            self._last_decision = self.decision_journal.by_decision_id(
                decision.decision_id
            )
            self._last_proposal = decision.proposal

            if decision.proposal is not None:
                self._last_risk_decision = self.gate.evaluate(
                    decision.proposal, risk_state, account_state
                )
            else:
                self._last_risk_decision = None
        finally:
            self.system_state = SystemState.CONNECTED

        return {
            "decision": self.decision_summary(),
            "risk": self.risk_summary(),
        }

    # ------------------------------------------------------------------
    # the single, explicit execution path
    # ------------------------------------------------------------------
    def execute_approved(self) -> ExecutionResult:
        with self._operation_lock:
            return self._execute_approved_locked()

    def _execute_approved_locked(self) -> ExecutionResult:
        # Approval is single-use, including blocked and exceptional attempts.
        proposal = self._last_proposal
        risk_decision = self._last_risk_decision
        self._last_proposal = None
        self._last_risk_decision = None
        if proposal is not None and proposal.is_expired(self.clock()):
            raise ExecutionRefused("proposal_expired: approved proposal has expired")
        if not self.control.started:
            raise EngineNotStartedError("engine is stopped — POST /control/start first")
        halts = self.control.halt_flags()
        if halts.get("emergency_stop_active") or halts.get("kill_switch_active"):
            self.system_state = SystemState.EMERGENCY_STOP
            raise ExecutionRefused(
                "operator halt is active — execution refused "
                "(reset via POST /control/reset_kill_switch)"
            )
        if proposal is None or risk_decision is None:
            raise ExecutionRefused("no proposal to execute — run a cycle first")
        if not risk_decision.approved:
            raise ExecutionRefused(
                f"last risk decision was {risk_decision.action.value} — "
                "only APPROVED proposals are executable"
            )

        request = ExecutionRequest.from_proposal(
            proposal,
            risk_decision_id=risk_decision.gate_decision_id,
            deviation_points=self.execution_service.config.deviation_points,
            type_filling=self.execution_service.config.type_filling,
        )
        self.system_state = SystemState.EXECUTING
        try:
            return self.execution_service.execute(request, risk_decision)
        finally:
            self.system_state = SystemState.CONNECTED

    # ------------------------------------------------------------------
    # reconciliation
    # ------------------------------------------------------------------
    def reconcile(self) -> ReconciliationReport:
        report = self.reconciler.reconcile(now=self.clock(), event_bus=self.bus)
        self.guard.record(report)
        if not report.clean:
            log.error(
                "reconciliation mismatch — execution halted until resolved",
                event="RECONCILIATION_HALT",
                counts=report.counts,
            )
        return report

    def acknowledge_reconciliation(self, reason: str = "operator") -> bool:
        acknowledged = self.guard.acknowledge()
        if acknowledged:
            self.bus.emit(
                CONTROL_RECONCILIATION_ACKNOWLEDGED, "control", reason=reason
            )
        return acknowledged

    # ------------------------------------------------------------------
    # control operations
    # ------------------------------------------------------------------
    def force_dry_run(self) -> None:
        """Safety ratchet: the control plane can only DE-escalate to
        DRY_RUN; re-enabling demo execution requires a config change and
        a restart (audited)."""
        self.execution_service.force_dry_run()
        self.bus.emit(CONTROL_DRY_RUN_FORCED, "control")

    # ------------------------------------------------------------------
    # status views (all credential-free)
    # ------------------------------------------------------------------
    def status(self) -> dict:
        connected = False
        account_type = "unknown"
        account_login = None
        try:
            connected = self.broker.is_connected
            account = self._read_account()
            if account is not None:
                account_type = account.trade_mode.value
                account_login = account.login
        except (BrokerError, AurumXError):
            pass

        tick = None
        if connected and self._symbol:
            try:
                tick = self.broker.get_tick(self._symbol)
            except BrokerError:
                tick = None

        return {
            "system_state": self.system_state.value,
            "started": self.control.started,
            "mt5_connected": connected,
            "account_type": account_type,  # from the BROKER, never config (§31)
            "account_login": account_login,
            "symbol": self._symbol,
            "mode": self.execution_service.mode.value,
            "trading_enabled": self.execution_service.config.trading_enabled,
            "dry_run": self.execution_service.config.dry_run,
            "kill_switch_active": self.control.halt_flags()["kill_switch_active"],
            "emergency_stop_active": self.control.halt_flags()["emergency_stop_active"],
            "reconciliation": self.guard.status,
            "bid": tick.bid if tick else None,
            "ask": tick.ask if tick else None,
            "spread": round(tick.spread, 4) if tick else None,
            "timestamp": self.clock().isoformat(),
        }

    def market(self) -> dict:
        if not self._symbol:
            return {"error": "not connected"}
        try:
            snapshot = self.data_service.get_snapshot()
        except (BrokerError, AurumXError) as exc:
            return {"error": str(exc), "symbol": self._symbol}
        return {
            "symbol": snapshot.symbol.name if snapshot.symbol else self._symbol,
            "tick": {
                "bid": snapshot.tick.tick.bid if snapshot.tick.tick else None,
                "ask": snapshot.tick.tick.ask if snapshot.tick.tick else None,
                "spread_points": snapshot.tick.spread_points,
                "time": (
                    snapshot.tick.tick.time.isoformat() if snapshot.tick.tick else None
                ),
            },
            "trading_data_ok": snapshot.trading_data_ok,
            "session": snapshot.session_state.value if snapshot.session_state else None,
        }

    def agents(self) -> dict:
        if self._last_decision is None:
            return {"agents": []}
        return {"agents": self._last_decision.agent_results}

    def decision_summary(self) -> dict:
        if self._last_decision is None:
            return {"decision": None}
        record = self._last_decision
        return {
            "decision_id": record.decision_id,
            "decision": record.decision.value,
            "symbol": record.symbol,
            "timestamp": record.timestamp.isoformat(),
            "regime": record.regime,
            "reasons": record.reasons[:5],
            "proposal": record.proposal is not None,
        }

    def risk_summary(self) -> dict:
        if self._last_risk_decision is None:
            return {"risk_decision": None}
        decision = self._last_risk_decision
        return {
            "action": decision.action.value,
            "gate_decision_id": decision.gate_decision_id,
            "risk_amount": decision.risk_amount,
            "failed_checks": decision.failed_checks,
            "reasons": decision.reasons[:5],
            "warnings": decision.warnings[:5],
        }

    def execution_summary(self, limit: int = 20) -> dict:
        records = self.execution_journal.recent(limit)
        return {"executions": [r.summary() for r in records]}

    def positions(self) -> dict:
        try:
            positions = self.broker.get_positions()
        except (BrokerError, AurumXError) as exc:
            return {"positions": [], "error": str(exc)}
        return {
            "positions": [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "direction": p.direction.value,
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "sl": p.price_sl,
                    "tp": p.price_tp,
                    "profit": p.profit,
                    "magic": p.magic,
                    "is_aurumx": p.magic == self.execution_service_config_magic(),
                }
                for p in positions
            ]
        }

    def reconciliation_summary(self) -> dict:
        report = self.guard.report
        return {
            "status": self.guard.status,
            "execution_allowed": self.guard.execution_allowed,
            "last_report": report.summary() if report is not None else None,
        }

    def events(self, limit: int = 50) -> dict:
        return {"events": [e.to_dict() for e in self.bus.history()[-limit:]]}

    @staticmethod
    def execution_service_config_magic() -> int:
        from app.execution.contracts import AURUMX_MAGIC

        return AURUMX_MAGIC
