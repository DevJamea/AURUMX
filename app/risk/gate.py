"""Phase-4 RiskGate CONTRACT — interface only, NOT implemented (hardening §4).

Boundary (docs/DECISIONS.md §14):

    DecisionEngine          (Phase 3 — decision correctness)
          ↓
    TradeProposal           (pure data; nothing has been sent anywhere)
          ↓
    RiskGate                (Phase 4 — INDEPENDENT safety barrier)
          ↓
    RiskDecision            (APPROVED / REJECTED / EMERGENCY_STOP)
          ↓
    execution service       (Phase 5 — only for APPROVED proposals)

The division of responsibility is absolute:

    Phase 3 = decision correctness   (is this a good trade to propose?)
    Phase 4 = independent safety barrier (is this trade SAFE to execute?)

Phase 4 must re-verify every safety-relevant property ITSELF and must never
assume the decision engine already checked something.  The engine's
risk-state gates exist to avoid *proposing* unsafe trades; the RiskGate
exists to stop *executing* them.  Two independent layers, deliberately
duplicated by design (defense in depth).

This module defines only the pure contract Phase 4 must satisfy:

* ``RiskCheck`` / ``RiskDecision`` — the audit-trail data models;
* ``RiskGate`` — the evaluation Protocol (signature fixed:
  ``evaluate(proposal, risk_state, account_state) -> RiskDecision``);
* ``REQUIRED_CHECKS`` — the named checks every implementation MUST perform
  independently (see docstring).

There is deliberately NO implementation here, and no Phase-3 code may call
into one: ``app/decision`` never imports this module (enforced by boundary
tests).  The gate is pure — no MT5, no network, no wall clock, no I/O — so
the same gate can evaluate backtest proposals unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import RiskAction
from app.decision.proposal import TradeProposal
from app.risk.state import AccountState, RiskState

#: Named checks every Phase-4 RiskGate implementation MUST perform
#: independently of the decision engine (hardening §5).  An implementation
#: may add checks; it may never skip one of these.  Each check must be able
#: to REJECT a dangerous proposal on its own evidence.
REQUIRED_CHECKS: tuple[str, ...] = (
    # --- monetary risk -----------------------------------------------------
    "max_risk_per_trade",           # proposal risk <= configured % of equity
    "max_total_exposure",           # open exposure + proposal <= limit
    "daily_loss_limit",             # today's realized+unrealized loss budget
    "consecutive_loss_protection",  # losing-streak stand-down
    # --- exposure counts ---------------------------------------------------
    "max_open_positions",           # open position count limit
    "max_pending_orders",           # pending order count limit
    # --- market conditions -------------------------------------------------
    "max_spread",                   # current quoted spread vs limit
    # --- instrument & order sanity ------------------------------------------
    "symbol_restriction",           # gold-only: proposal.symbol allowed
    "volume_limits",                # volume_min <= vol <= volume_max, step-aligned
    "sl_presence",                  # every proposal carries a usable stop loss
    "tp_validity",                  # TP present and geometrically valid
    # --- system safety ------------------------------------------------------
    "emergency_stop",               # persistent operator emergency stop
    "kill_switch",                  # global kill switch (config level)
    "account_safety",               # account state allows trading (margin,
                                    # trade_allowed, sane balance/equity)
)


class RiskCheck(BaseModel):
    """One named safety check's outcome — the audit trail of the barrier.

    A check that cannot be evaluated (missing data) is a FAILED check:
    the gate is fail-closed by construction.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    name: str
    passed: bool
    detail: str = ""


class RiskDecision(BaseModel):
    """The RiskGate's complete answer for one proposal.

    The gate NEVER modifies a proposal — it answers APPROVED (the proposal
    passes unchanged), REJECTED (named reasons), or EMERGENCY_STOP (an
    operator halt is active; no proposal is executable while it is).
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    action: RiskAction
    #: decision_id of the evaluated TradeProposal (provenance link)
    proposal_id: str
    fingerprint: str | None = None
    checks: list[RiskCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.action is RiskAction.APPROVED

    def summary(self) -> dict:
        """JSON-safe summary for logs / journal / dashboard."""
        return {
            "action": self.action.value,
            "proposal_id": self.proposal_id,
            "fingerprint": self.fingerprint,
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": [c.name for c in self.checks if not c.passed],
            "reasons": list(self.reasons),
        }


@runtime_checkable
class RiskGate(Protocol):
    """The Phase-4 safety barrier contract (NOT implemented in Phase 3).

    Implementations must:

    * be pure functions of their inputs (no wall clock, no network, no
      broker calls, no global mutable state) — backtest-safe;
    * perform every check in ``REQUIRED_CHECKS`` independently, on the
      evidence carried by ``proposal`` + ``risk_state`` + ``account_state``
      plus the implementation's own configuration;
    * be fail-closed: unknown/missing input data fails the corresponding
      check (never assume "probably fine");
    * return APPROVED only when every check passed; EMERGENCY_STOP when an
      operator halt (emergency stop / kill switch) is active; REJECTED with
      named failed checks otherwise;
    * never modify the proposal, never place/modify/close anything.
    """

    def evaluate(
        self,
        proposal: TradeProposal,
        risk_state: RiskState,
        account_state: AccountState,
    ) -> RiskDecision: ...
