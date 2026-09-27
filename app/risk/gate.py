"""RiskGate contract + models (Phase 4 implements it in app/risk/engine.py).

Boundary (docs/RISK_GATE.md, docs/DECISIONS.md §14):

    DecisionEngine          Phase 3 — decision correctness
          ↓
    TradeProposal           pure data; nothing has been sent anywhere
          ↓
    HardRiskGate            Phase 4 — INDEPENDENT safety barrier
          ↓
    RiskDecision            APPROVED / REJECTED / EMERGENCY_STOP
          ↓
    execution service       Phase 5 — only for APPROVED proposals

The division of responsibility is absolute:

    Phase 3 = decision correctness   (is this a good trade to propose?)
    Phase 4 = independent safety barrier (is this trade SAFE to execute?)

The gate re-verifies every safety-relevant property ITSELF and never
assumes the decision engine already checked something.  It is pure —
no MT5, no network, no wall clock, no I/O — so the same gate evaluates
backtest proposals unchanged.  It NEVER modifies a proposal: it answers.

This module defines the contract (models + Protocol + REQUIRED_CHECKS).
The implementation is ``HardRiskGate`` in ``app/risk/engine.py``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import RiskAction
from app.risk.state import AccountState, RiskState

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids import cycles)
    from app.decision.proposal import TradeProposal

#: Named checks every RiskGate implementation MUST perform independently
#: of the decision engine (hardening §5).  An implementation may add
#: checks; it may never skip one of these.  Each check must be able to
#: REJECT a dangerous proposal on its own evidence.
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


class CheckStatus(Enum):
    """Outcome of one safety check.

    PASS          the check evaluated and the evidence is within limits
    FAIL          the check evaluated and the evidence violates a limit
                  (or required evidence is missing — fail closed)
    WARN          evaluated and within limits, but close enough to a limit
                  to be worth surfacing (never blocks approval on its own)
    NOT_EVALUATED evaluation did not run (halted by a higher-precedence
                  state, or the check's policy is disabled)
    """

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    NOT_EVALUATED = "NOT_EVALUATED"


class CheckSeverity(Enum):
    """How blocking a check is.

    CRITICAL  a FAIL rejects the proposal (every REQUIRED_CHECK is critical)
    ADVISORY  informational only; a FAIL/WARN never blocks on its own
    """

    CRITICAL = "CRITICAL"
    ADVISORY = "ADVISORY"


class RiskCheck(BaseModel):
    """One named safety check's outcome — the audit trail of the barrier.

    Fail-closed by construction: a check whose evidence is missing is a
    FAILED check (``reason`` explains what evidence was absent).  Checks
    never repair anything — they observe and report.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    name: str
    status: CheckStatus
    severity: CheckSeverity = CheckSeverity.CRITICAL
    reason: str = ""
    #: the observed value the check compared against the limit
    observed_value: float | str | None = None
    #: the limit in force when the check ran
    limit: float | str | None = None

    @property
    def passed(self) -> bool:
        """PASS and WARN both clear a check (WARN is advisory)."""
        return self.status in (CheckStatus.PASS, CheckStatus.WARN)


class RiskDecision(BaseModel):
    """The RiskGate's complete answer for one proposal.

    The gate NEVER modifies a proposal — it answers APPROVED (the proposal
    passes unchanged), REJECTED (named failed checks), or EMERGENCY_STOP
    (an operator halt is active; no proposal is executable while it is).
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    action: RiskAction
    #: decision_id of the evaluated TradeProposal (provenance link)
    proposal_id: str
    fingerprint: str | None = None
    #: deterministic id derived from the inputs (never a UUID/wall clock)
    gate_decision_id: str = ""
    checks: list[RiskCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: independently computed monetary risk of the proposal (never the
    #: proposal's own claim — the proposal is evidence, not authority)
    risk_amount: float | None = None
    #: exposure breakdown: current / proposed / total / limit
    exposure: dict | None = None
    #: the safety-relevant inputs that were in force (config snapshot)
    config_snapshot: dict = Field(default_factory=dict)
    #: halt states observed at evaluation time
    kill_switch_active: bool = False
    emergency_stop_active: bool = False

    @property
    def approved(self) -> bool:
        return self.action is RiskAction.APPROVED

    @property
    def passed_checks(self) -> list[str]:
        return [c.name for c in self.checks if c.status is CheckStatus.PASS]

    @property
    def failed_checks(self) -> list[str]:
        """Checks that evaluated to FAIL (NOT_EVALUATED is not a failure —
        it means a higher-precedence state stopped the evaluation)."""
        return [c.name for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def not_evaluated_checks(self) -> list[str]:
        return [c.name for c in self.checks if c.status is CheckStatus.NOT_EVALUATED]

    def summary(self) -> dict:
        """JSON-safe summary for logs / journal / dashboard."""
        return {
            "action": self.action.value,
            "gate_decision_id": self.gate_decision_id,
            "proposal_id": self.proposal_id,
            "fingerprint": self.fingerprint,
            "passed": [c.name for c in self.checks if c.status is CheckStatus.PASS],
            "warned": [c.name for c in self.checks if c.status is CheckStatus.WARN],
            "failed": self.failed_checks,
            "not_evaluated": self.not_evaluated_checks,
            "risk_amount": self.risk_amount,
            "exposure": self.exposure,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "kill_switch_active": self.kill_switch_active,
            "emergency_stop_active": self.emergency_stop_active,
        }


@runtime_checkable
class RiskGate(Protocol):
    """The safety-barrier contract (implemented by HardRiskGate).

    Implementations must:

    * be pure functions of their inputs (no wall clock, no network, no
      broker calls, no global mutable state) — backtest-safe;
    * perform every check in ``REQUIRED_CHECKS`` independently, on the
      evidence carried by ``proposal`` + ``risk_state`` + ``account_state``
      plus the implementation's own configuration;
    * be fail-closed: unknown/missing input data fails the corresponding
      check (never assume "probably fine");
    * return APPROVED only when every critical check passed; EMERGENCY_STOP
      when an operator halt (emergency stop / kill switch) is active;
      REJECTED with named failed checks otherwise;
    * never modify the proposal, never place/modify/close anything.
    """

    def evaluate(
        self,
        proposal: TradeProposal,
        risk_state: RiskState,
        account_state: AccountState,
    ) -> RiskDecision: ...
