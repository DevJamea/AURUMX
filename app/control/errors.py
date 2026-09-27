"""Control-plane errors (fail-closed, never silent)."""

from __future__ import annotations

from app.core.exceptions import AurumXError


class ControlError(AurumXError):
    """A control operation was refused."""


class EngineNotStartedError(ControlError):
    """The operation requires a started engine (POST /control/start first)."""


class ExecutionRefused(ControlError):
    """Execution was refused before reaching the execution service —
    e.g. an operator halt is active or there is no approved proposal.
    (The execution service performs its own independent gates afterwards.)"""
