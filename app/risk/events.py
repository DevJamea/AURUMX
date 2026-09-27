"""Structured risk events for future observability (Phase 4, spec §25).

Internal data objects ONLY: no Telegram, no HTTP, no websockets.  A later
observability layer (API/websocket/telegram) may subscribe through an
``event_sink`` callback on the gate and forward these objects — the gate
itself never transmits anything.

The events carry no timestamp by design: the gate is wall-clock-free, so
the receiving layer stamps them at observation time.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.core.enums import RiskAction


class RiskEventType(Enum):
    """One event type per gate outcome (plus the kill-switch variant)."""

    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


class RiskEvent(BaseModel):
    """A single gate evaluation, as an observability data object."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    type: RiskEventType
    action: RiskAction
    proposal_id: str
    fingerprint: str | None = None
    gate_decision_id: str = ""
    reasons: list[str] = []

    @classmethod
    def from_decision(cls, decision, event_type: RiskEventType) -> RiskEvent:
        """Build the event for a RiskDecision (pure transformation)."""
        return cls(
            type=event_type,
            action=decision.action,
            proposal_id=decision.proposal_id,
            fingerprint=decision.fingerprint,
            gate_decision_id=decision.gate_decision_id,
            reasons=list(decision.reasons),
        )

    def summary(self) -> dict:
        return {
            "type": self.type.value,
            "action": self.action.value,
            "proposal_id": self.proposal_id,
            "fingerprint": self.fingerprint,
            "gate_decision_id": self.gate_decision_id,
            "reasons": list(self.reasons),
        }
