"""Operator control state: start/stop + the persistent halts (5E).

The halts reuse the Phase-4 kill-switch models
(``KillSwitchState``/``EmergencyStopState`` + ``KillSwitchStore``) — the
control plane is the "future GUI/VPS persistence layer" those models were
built for: one store per halt (kill switch, emergency stop), so the two
stay independent exactly as the Phase-4 contract requires.

Every transition is audited on the event bus; nothing here can send an
order.
"""

from __future__ import annotations

from app.core.events import EventBus
from app.core.logging import get_logger
from app.execution.events import (
    CONTROL_EMERGENCY_STOP,
    CONTROL_KILL_SWITCH_RESET,
    CONTROL_STARTED,
    CONTROL_STOPPED,
)
from app.risk import (
    EmergencyStopState,
    HaltStatus,
    InMemoryKillSwitchStore,
    KillSwitchState,
    KillSwitchStore,
    halt_flags,
)

log = get_logger("control.state")


class EngineControl:
    """Start/stop flag + the two operator halts, all audited."""

    def __init__(
        self,
        kill_switch_store: KillSwitchStore | None = None,
        emergency_stop_store: KillSwitchStore | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._kill_store: KillSwitchStore = kill_switch_store or InMemoryKillSwitchStore()
        self._stop_store: KillSwitchStore = emergency_stop_store or InMemoryKillSwitchStore()
        self._bus = event_bus
        self.started: bool = False

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.started = True
        self._emit(CONTROL_STARTED, started=True)
        log.info("engine started by operator", event="CONTROL_STARTED")

    def stop(self) -> None:
        self.started = False
        self._emit(CONTROL_STOPPED, started=False)
        log.info("engine stopped by operator", event="CONTROL_STOPPED")

    # ------------------------------------------------------------------
    # halts (persistent — a restart cannot silently re-enable trading)
    # ------------------------------------------------------------------
    def engage_kill_switch(self, reason: str = "operator") -> None:
        self._kill_store.save(KillSwitchState(status=HaltStatus.ACTIVE, reason=reason))
        self._emit(CONTROL_EMERGENCY_STOP, halt="kill_switch", reason=reason)
        log.warning("kill switch engaged", event="CONTROL_KILL_SWITCH_ACTIVE", reason=reason)

    def engage_emergency_stop(self, reason: str = "operator") -> None:
        self._stop_store.save(
            EmergencyStopState(status=HaltStatus.ACTIVE, reason=reason)
        )
        self._emit(CONTROL_EMERGENCY_STOP, halt="emergency_stop", reason=reason)
        log.warning("EMERGENCY STOP engaged", event="CONTROL_EMERGENCY_STOP", reason=reason)

    def reset_kill_switch(self) -> dict:
        """Explicit operator reset of BOTH halt states (audited).  This is
        the only way out of an emergency stop — and it is logged."""
        self._kill_store.save(KillSwitchState(status=HaltStatus.STANDBY))
        self._stop_store.save(EmergencyStopState(status=HaltStatus.STANDBY))
        self._emit(CONTROL_KILL_SWITCH_RESET)
        log.info(
            "kill switch / emergency stop reset by operator",
            event="CONTROL_KILL_SWITCH_RESET",
        )
        return self.halt_flags()

    # ------------------------------------------------------------------
    # state views
    # ------------------------------------------------------------------
    def kill_switch(self) -> KillSwitchState:
        state = self._kill_store.load()
        return state if isinstance(state, KillSwitchState) else KillSwitchState()

    def emergency_stop(self) -> EmergencyStopState:
        state = self._stop_store.load()
        return state if isinstance(state, EmergencyStopState) else EmergencyStopState()

    def halt_flags(self) -> dict[str, bool]:
        """The documented Phase-4 bridge onto the RiskGate's inputs."""
        return halt_flags(
            kill_switch=self.kill_switch(), emergency_stop=self.emergency_stop()
        )

    def summary(self) -> dict:
        return {
            "started": self.started,
            **self.halt_flags(),
            "kill_switch": self.kill_switch().summary(),
            "emergency_stop": self.emergency_stop().summary(),
        }

    def _emit(self, event_type: str, **payload) -> None:
        if self._bus is not None:
            self._bus.emit(event_type, "control", **payload)
