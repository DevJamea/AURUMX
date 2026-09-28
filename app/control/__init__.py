"""Local Windows control plane (Phase 5E).

    Windows GUI (static page, zero MT5 access)
        -> Local Control API (localhost, JSON)
            -> EngineRuntime (the engine composition)
                -> HardRiskGate / ExecutionService / Reconciler
                    -> MT5Broker (the only MetaTrader5 importer)

The control plane is an *interface to the engine*, not an alternative
execution path: there is no HTTP endpoint that sends an order.  Control
operations can only START/STOP the engine, force DRY_RUN, engage/reset
operator halts and resolve reconciliation — every one of them audited on
the event bus.
"""

from app.control.errors import ControlError, EngineNotStartedError, ExecutionRefused
from app.control.runtime import EngineRuntime
from app.control.state import EngineControl

__all__ = [
    "ControlError",
    "EngineControl",
    "EngineNotStartedError",
    "EngineRuntime",
    "ExecutionRefused",
]
