"""Kill switch + emergency stop state (Phase 4, spec §24/§4/§5).

Deterministic, storage-independent STATE MODELS for the two operator
halts, plus a persistence interface (Protocol) suitable for a future
GUI/VPS control layer.  Phase 4 deliberately ships no I/O: the control
layer that flips these states and persists them across restarts arrives
later; when it does, it only has to implement ``KillSwitchStore``.

Distinction (docs/RISK_GATE.md):

* **kill switch** — the global, config-level halt (``KILL_SWITCH_ACTIVE``);
* **emergency stop** — the persistent operator halt
  (``EMERGENCY_STOP``).

Both map onto the ``AccountState`` boolean flags the RiskGate consumes
(``kill_switch_active`` / ``emergency_stop_active``) — the gate itself
never touches storage, so it stays pure and backtest-safe.  Both are
caller-supplied catastrophic states: the gate NEVER invents automatic
emergency conditions (that would be an undocumented safety behavior).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from app.core.enums import RiskAction


class HaltStatus(StrEnum):
    """Lifecycle of a halt.

    STANDBY        not engaged — normal operation
    ACTIVE         engaged by an operator (or startup config); halts trading
    TRIGGERED      engaged automatically by a documented trigger source;
                   halts trading AND requires a manual reset
    RESET_REQUIRED engaged and latched: it stays engaged until an operator
                   explicitly resets it (a restart must NOT clear it —
                   persistence is the control layer's job)
    """

    STANDBY = "STANDBY"
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    RESET_REQUIRED = "RESET_REQUIRED"


class _HaltState(BaseModel):
    """Shared shape of a halt state (see KillSwitchState/EmergencyStopState)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    status: HaltStatus = HaltStatus.STANDBY
    reason: str = ""
    #: who/what engaged the halt ("operator", "config", "auto:<name>")
    source: str = "operator"
    #: when the halt was engaged — set by the control layer (informational;
    #: the RiskGate never reads the clock, so this never affects decisions)
    activated_at: datetime | None = None

    @property
    def engaged(self) -> bool:
        """True whenever the halt is anything other than STANDBY."""
        return self.status is not HaltStatus.STANDBY

    @property
    def requires_reset(self) -> bool:
        """True when the halt latches until an operator resets it."""
        return self.status in (HaltStatus.TRIGGERED, HaltStatus.RESET_REQUIRED)

    def gate_action(self) -> RiskAction | None:
        """The gate outcome this state forces (None = no halt)."""
        return RiskAction.EMERGENCY_STOP if self.engaged else None

    def summary(self) -> dict:
        return {
            "status": self.status.value,
            "engaged": self.engaged,
            "reason": self.reason,
            "source": self.source,
        }


class KillSwitchState(_HaltState):
    """The global kill switch (config-level halt).

    Future control layer: persist with a ``KillSwitchStore``; map onto the
    gate input via ``AccountState(kill_switch_active=state.engaged, ...)``.
    """

    source: str = "config"


class EmergencyStopState(_HaltState):
    """The persistent operator emergency stop.

    Must survive restarts (spec: persistent emergency stop) — the store
    interface below is where the future GUI/VPS layer plugs in.
    """

    source: str = "operator"


class KillSwitchStore(Protocol):
    """Persistence interface for a future GUI/VPS control layer.

    Contract: ``load`` returns the persisted state (STANDBY when nothing
    was ever persisted); ``save`` durably replaces it.  Implementations
    may be file/Redis/DB backed — the gate neither knows nor cares.
    """

    def load(self) -> _HaltState: ...

    def save(self, state: _HaltState) -> None: ...


class InMemoryKillSwitchStore:
    """Reference implementation (tests, backtests, single-process use).

    Not durable across restarts — production persistence arrives with the
    control layer.  Deterministic and thread-hostile by design (the future
    control layer owns synchronization).
    """

    def __init__(self) -> None:
        self._state: _HaltState | None = None

    def load(self) -> _HaltState:
        return self._state or KillSwitchState()

    def save(self, state: _HaltState) -> None:
        self._state = state


def halt_flags(
    kill_switch: KillSwitchState | None = None,
    emergency_stop: EmergencyStopState | None = None,
) -> dict:
    """Map halt states onto the AccountState flags the gate consumes.

    The documented bridge between the (future) control layer and the gate
    inputs: ``AccountState(**halt_flags(ks, es), ...)``.
    """
    ks = kill_switch or KillSwitchState()
    es = emergency_stop or EmergencyStopState()
    return {
        "kill_switch_active": ks.engaged,
        "emergency_stop_active": es.engaged,
    }
