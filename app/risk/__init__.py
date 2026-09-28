"""Risk layer: Phase-3 sizing/state + the Phase-4 hard safety barrier.

* ``sizing``       — deterministic risk-based position sizing (no martingale,
                     no loss-recovery, no DCA: past losses are not an input);
* ``state``        — RiskState (engine-side risk accumulators) and
                     AccountState (execution-side account/safety evidence);
* ``gate``         — the RiskGate CONTRACT (Protocol, RiskDecision,
                     RiskCheck, REQUIRED_CHECKS);
* ``engine``       — HardRiskGate, the Phase-4 implementation: independent,
                     fail-closed, pure (no MT5/network/wall clock/IO);
* ``kill_switch``  — kill-switch / emergency-stop STATE models + the
                     persistence interface for a future control layer;
* ``events``       — internal risk events (RISK_APPROVED / RISK_REJECTED /
                     EMERGENCY_STOP / KILL_SWITCH_ACTIVE) — data objects
                     only, never transmitted by this layer.

The gate decides APPROVED / REJECTED / EMERGENCY_STOP.  It never modifies
a proposal and never executes anything (Phase 5 owns execution).
"""

from app.risk.engine import ADDITIONAL_CHECKS, IMPLEMENTED_CHECKS, HardRiskGate, RiskGateConfig
from app.risk.events import RiskEvent, RiskEventType
from app.risk.gate import (
    REQUIRED_CHECKS,
    CheckSeverity,
    CheckStatus,
    RiskCheck,
    RiskDecision,
    RiskGate,
)
from app.risk.kill_switch import (
    EmergencyStopState,
    HaltStatus,
    InMemoryKillSwitchStore,
    KillSwitchState,
    KillSwitchStore,
    halt_flags,
)
from app.risk.sizing import RiskSizing, calculate_position_size
from app.risk.state import AccountState, RiskState

__all__ = [
    "ADDITIONAL_CHECKS", "IMPLEMENTED_CHECKS", "REQUIRED_CHECKS",
    "AccountState", "CheckSeverity", "CheckStatus", "EmergencyStopState",
    "HardRiskGate", "HaltStatus", "InMemoryKillSwitchStore", "KillSwitchState",
    "KillSwitchStore", "RiskCheck", "RiskDecision", "RiskEvent",
    "RiskEventType", "RiskGate", "RiskGateConfig", "RiskSizing", "RiskState",
    "calculate_position_size", "halt_flags",
]

