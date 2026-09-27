"""HardRiskGate — the Phase-4 implementation of the RiskGate contract.

The gate is an INDEPENDENT, fail-closed safety barrier.  It re-derives
every safety-relevant number from the evidence triple
(``TradeProposal``, ``RiskState``, ``AccountState``) plus its own
configuration — it never trusts the proposal's claims about its own risk,
never repairs an invalid proposal, and never modifies one.

Purity: no MT5, no network, no wall clock, no I/O, no randomness.  The
same inputs always produce the same ``RiskDecision`` (the decision id is
derived deterministically from the inputs).  The gate is therefore usable
unchanged inside backtests.

Outcome precedence (docs/RISK_GATE.md §Precedence):

    1. EMERGENCY_STOP   — emergency-stop flag active
    2. KILL_SWITCH      — kill-switch flag active (same action, its own
                          event/reason; emergency stop is reported first
                          when both are active)
    3. TRADING_DISABLED — trading_enabled=false (safe default)
    4. ACCOUNT_NOT_ALLOWED — trade_allowed=false / unsafe account evidence
    5. SAFETY_CHECK_FAILURE — any critical check FAIL (all failures in
                          this tier are evaluated and reported together)
    6. APPROVED         — every critical check passed

Tiers 1-4 short-circuit (evaluation stops; remaining checks are recorded
NOT_EVALUATED — never hidden).  Tier 5 runs every remaining check so the
audit trail shows ALL independent failures, not just the first.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AgentDirection, RiskAction
from app.core.models import SymbolSpec
from app.decision.levels import min_stop_distance
from app.market.symbol_discovery import is_gold_symbol
from app.risk.events import RiskEvent, RiskEventType
from app.risk.gate import (
    REQUIRED_CHECKS,
    CheckSeverity,
    CheckStatus,
    RiskCheck,
    RiskDecision,
)
from app.risk.state import AccountState, RiskState

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids import cycles)
    from app.decision.proposal import TradeProposal

#: checks beyond the frozen contract list (spec §3 items 1/6/7 + §18)
ADDITIONAL_CHECKS: tuple[str, ...] = (
    "trading_enabled",   # trading globally enabled (safe default: False)
    "direction_valid",   # only BUY / SELL may ever be executable
    "entry_valid",       # entry price finite and positive
    "margin_safety",     # optional minimum margin-level policy
)

#: full ordered check list — the order is the evaluation order
IMPLEMENTED_CHECKS: tuple[str, ...] = (
    "emergency_stop",
    "kill_switch",
    "trading_enabled",
    "account_safety",
    "symbol_restriction",
    "direction_valid",
    "entry_valid",
    "sl_presence",
    "tp_validity",
    "volume_limits",
    "max_risk_per_trade",
    "max_total_exposure",
    "daily_loss_limit",
    "consecutive_loss_protection",
    "max_open_positions",
    "max_pending_orders",
    "max_spread",
    "margin_safety",
)

assert set(REQUIRED_CHECKS) <= set(IMPLEMENTED_CHECKS)

#: relative floating-point tolerance for limit comparisons.  A value at
#: exactly the limit passes; a value above the limit by more than this
#: tiny relative slack fails.  Never rounds risk UP into approval.
_REL_TOL = 1e-9


class RiskGateConfig(BaseModel):
    """Every gate threshold — the single home of Phase-4 tunables.

    Safe defaults (fail-closed): ``trading_enabled`` is False and the
    exposure limit is unset — a default-constructed gate rejects
    everything.  Operators must affirmatively enable trading and configure
    ``max_total_exposure``.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    #: trading globally enabled — NEVER default-true (spec §19)
    trading_enabled: bool = False
    #: informational mirror of the app-level DRY_RUN flag; approval is not
    #: execution, so this does not affect gating (Phase 5 consumes it)
    dry_run: bool = True
    #: max risk per trade as a fraction of equity (percent)
    max_risk_per_trade_pct: float | None = Field(default=0.5, gt=0, le=100.0)
    #: monetary override — when set, used instead of the percentage
    max_risk_per_trade_amount: float | None = Field(default=None, gt=0)
    #: max total exposure (monetary).  REQUIRED: None -> the exposure
    #: check fails closed ("limit not configured")
    max_total_exposure: float | None = Field(default=None, gt=0)
    #: monetary daily loss limit; None -> pct fallback below
    daily_loss_limit: float | None = Field(default=None, gt=0)
    #: daily loss limit as percent of equity (used when no monetary limit
    #: is configured anywhere).  Units are never mixed: monetary vs
    #: monetary, percent applies to equity evidence.
    daily_loss_limit_pct: float | None = Field(default=2.0, gt=0, le=100.0)
    #: consecutive closed losses tolerated before stand-down
    max_consecutive_losses: int = Field(default=3, ge=1)
    #: exposure count limits
    max_open_positions: int = Field(default=1, ge=1)
    max_pending_orders: int = Field(default=2, ge=0)
    #: max quoted spread in points.  REQUIRED: None -> fail closed
    max_spread_points: float | None = Field(default=50.0, gt=0)
    #: optional margin-level floor in percent (spec §18).  None = policy
    #: off -> the margin_safety check is NOT_EVALUATED (advisory)
    min_free_margin_percent: float | None = Field(default=None, gt=0, le=1000.0)

    @classmethod
    def from_app_config(cls, config) -> RiskGateConfig:
        """Build from the runtime AppConfig (Phase-1 fields; safety flags
        keep their safe defaults unless the app affirms them)."""
        kwargs: dict = {}
        for source, target in (
            ("trading_enabled", "trading_enabled"),
            ("dry_run", "dry_run"),
            ("risk_per_trade_pct", "max_risk_per_trade_pct"),
            ("max_daily_loss_pct", "daily_loss_limit_pct"),
            ("max_open_positions", "max_open_positions"),
            ("max_pending_orders", "max_pending_orders"),
            ("max_spread_points", "max_spread_points"),
        ):
            value = getattr(config, source, None)
            if value is not None:
                kwargs[target] = value
        if getattr(config, "max_consecutive_losses", None) is not None:
            kwargs["max_consecutive_losses"] = config.max_consecutive_losses
        # NOTE: max_total_exposure is deliberately NOT defaulted from the
        # app config — it must be configured explicitly (fail closed).
        return cls(**kwargs)

    def risk_snapshot(self) -> dict:
        """The safety-relevant inputs in force (journaled on every decision)."""
        return self.model_dump(mode="json")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _on_grid(price: float, tick_size: float) -> bool:
    """Price lies on the broker's tick grid (small relative tolerance)."""
    if tick_size <= 0:
        return True  # no grid declared — nothing to check against
    return abs(price / tick_size - round(price / tick_size)) <= 1e-6


def _at_or_below(value: float, limit: float) -> bool:
    """value <= limit within a tiny relative tolerance (never rounds up)."""
    return value <= limit + abs(limit) * _REL_TOL


class HardRiskGate:
    """The independent safety barrier (implements app.risk.gate.RiskGate).

    Construct with the verified Phase-1 ``SymbolSpec`` registry — the
    caller (worker) supplies the broker-verified instrument metadata the
    gate needs for geometry/risk/volume math.  A proposal whose symbol has
    no registered spec is rejected (missing evidence, fail closed).
    """

    def __init__(
        self,
        config: RiskGateConfig | None = None,
        *,
        symbol_specs: Mapping[str, SymbolSpec] | None = None,
        event_sink: Callable[[RiskEvent], None] | None = None,
    ) -> None:
        self.config = config or RiskGateConfig()
        self._symbol_specs: dict[str, SymbolSpec] = dict(symbol_specs or {})
        self._event_sink = event_sink

    # ------------------------------------------------------------------
    # the contract
    # ------------------------------------------------------------------
    def evaluate(
        self,
        proposal: TradeProposal,
        risk_state: RiskState,
        account_state: AccountState,
    ) -> RiskDecision:
        """Evaluate one proposal.  Pure, deterministic, never mutates."""
        checks: list[RiskCheck] = []
        warnings: list[str] = []
        action = RiskAction.APPROVED
        stop_reason: str | None = None

        def halt(check_name: str, reason: str, outcome: RiskAction) -> None:
            """Record the blocking check and stop evaluating (tiers 1-4)."""
            nonlocal action, stop_reason
            checks.append(
                RiskCheck(name=check_name, status=CheckStatus.FAIL, reason=reason)
            )
            action = outcome
            stop_reason = reason

        # ---- tier 1/2: operator halts override EVERYTHING ----------------
        # (a passing halt check means "no halt active")
        if account_state.emergency_stop_active:
            halt("emergency_stop", "operator emergency stop is active", RiskAction.EMERGENCY_STOP)
        elif account_state.kill_switch_active:
            halt("kill_switch", "global kill switch is active", RiskAction.EMERGENCY_STOP)
        else:
            checks.append(RiskCheck(name="emergency_stop", status=CheckStatus.PASS))
            checks.append(RiskCheck(name="kill_switch", status=CheckStatus.PASS))

            # ---- tier 3: global trading enable ----------------------------
            if not self.config.trading_enabled:
                halt(
                    "trading_enabled",
                    "trading is not enabled (risk.trading_enabled=false)",
                    RiskAction.REJECTED,
                )
            else:
                checks.append(
                    RiskCheck(name="trading_enabled", status=CheckStatus.PASS)
                )

                # ---- tier 4: account permission / sanity ------------------
                account_check = self._check_account_safety(risk_state, account_state)
                checks.append(account_check)
                if account_check.status is CheckStatus.FAIL:
                    action = RiskAction.REJECTED
                    stop_reason = account_check.reason

        # ---- tier 5: independent safety checks (all evaluated) -------------
        risk_amount: float | None = None
        exposure: dict | None = None
        if action is RiskAction.APPROVED:
            spec = self._symbol_specs.get(proposal.symbol)
            context = _CheckContext(
                proposal=proposal, risk_state=risk_state, account_state=account_state,
                spec=spec, config=self.config,
            )
            for name in IMPLEMENTED_CHECKS[4:]:  # after account_safety
                check = getattr(self, f"_check_{name}")(context)
                checks.append(check)
                if check.status is CheckStatus.WARN:
                    warnings.append(f"{name}: {check.reason}")
                elif check.status is CheckStatus.FAIL and check.severity is CheckSeverity.CRITICAL:
                    action = RiskAction.REJECTED
            risk_amount = context.risk_amount
            exposure = context.exposure

        # ---- checks that never ran (halted) are recorded, not hidden -------
        evaluated = {c.name for c in checks}
        for name in IMPLEMENTED_CHECKS:
            if name not in evaluated:
                checks.append(
                    RiskCheck(
                        name=name,
                        status=CheckStatus.NOT_EVALUATED,
                        reason=f"evaluation stopped: {stop_reason or 'earlier failure'}",
                    )
                )

        reasons = self._reasons_for(action, checks)
        decision = RiskDecision(
            action=action,
            proposal_id=proposal.decision_id,
            fingerprint=proposal.fingerprint,
            gate_decision_id=self._decision_id(proposal, action, checks),
            checks=checks,
            reasons=reasons,
            warnings=warnings,
            risk_amount=risk_amount,
            exposure=exposure,
            config_snapshot=self.config.risk_snapshot(),
            kill_switch_active=account_state.kill_switch_active,
            emergency_stop_active=account_state.emergency_stop_active,
        )
        self._emit(decision)
        return decision

    # ------------------------------------------------------------------
    # tier 4: account evidence
    # ------------------------------------------------------------------
    def _check_account_safety(
        self, risk_state: RiskState, account_state: AccountState
    ) -> RiskCheck:
        if not account_state.trade_allowed:
            return RiskCheck(
                name="account_safety", status=CheckStatus.FAIL,
                reason="trading_not_allowed: account/terminal does not allow trading",
                observed_value="trade_allowed=false", limit="trade_allowed=true",
            )
        # equity cross-check: where both sources report equity they must agree
        if (
            risk_state.equity is not None and account_state.equity is not None
            and risk_state.equity > 0 and account_state.equity > 0
        ):
            spread = abs(risk_state.equity - account_state.equity)
            ref = max(risk_state.equity, account_state.equity)
            if spread > ref * 0.005:  # 0.5% documented inconsistency budget
                return RiskCheck(
                    name="account_safety", status=CheckStatus.FAIL,
                    reason=(
                        "inconsistent equity evidence: risk_state "
                        f"{risk_state.equity:.2f} vs account_state {account_state.equity:.2f}"
                    ),
                    observed_value=round(spread, 2), limit=round(ref * 0.005, 2),
                )
        # position-count cross-check when the account reports counts
        for field_name, risk_field in (
            ("open_positions", "open_positions"),
            ("pending_orders", "pending_orders"),
        ):
            account_count = getattr(account_state, field_name)
            risk_count = getattr(risk_state, risk_field)
            if account_count is not None and account_count != risk_count:
                return RiskCheck(
                    name="account_safety", status=CheckStatus.FAIL,
                    reason=(
                        f"inconsistent {field_name} evidence: risk_state "
                        f"{risk_count} vs account_state {account_count}"
                    ),
                    observed_value=account_count, limit=risk_count,
                )
        return RiskCheck(name="account_safety", status=CheckStatus.PASS)

    # ------------------------------------------------------------------
    # tier 5: independent safety checks
    # ------------------------------------------------------------------
    def _check_symbol_restriction(self, ctx: _CheckContext) -> RiskCheck:
        if not is_gold_symbol(ctx.proposal.symbol):
            return RiskCheck(
                name="symbol_restriction", status=CheckStatus.FAIL,
                reason=f"symbol_not_allowed: {ctx.proposal.symbol} is not a gold symbol",
                observed_value=ctx.proposal.symbol, limit="gold-only (XAUUSD/GOLD family)",
            )
        if ctx.spec is None:
            return RiskCheck(
                name="symbol_restriction", status=CheckStatus.FAIL,
                reason=(
                    f"no verified symbol metadata registered for "
                    f"{ctx.proposal.symbol} (missing evidence, fail closed)"
                ),
                observed_value=ctx.proposal.symbol, limit="registered SymbolSpec required",
            )
        return RiskCheck(name="symbol_restriction", status=CheckStatus.PASS)

    def _check_direction_valid(self, ctx: _CheckContext) -> RiskCheck:
        if ctx.proposal.direction not in (AgentDirection.BUY, AgentDirection.SELL):
            return RiskCheck(
                name="direction_valid", status=CheckStatus.FAIL,
                reason=f"direction_not_executable: {ctx.proposal.direction}",
                observed_value=str(ctx.proposal.direction), limit="BUY|SELL",
            )
        return RiskCheck(name="direction_valid", status=CheckStatus.PASS)

    def _check_entry_valid(self, ctx: _CheckContext) -> RiskCheck:
        entry = ctx.proposal.entry_price
        if not _finite(entry) or entry <= 0:
            return RiskCheck(
                name="entry_valid", status=CheckStatus.FAIL,
                reason=f"entry price invalid: {entry}",
                observed_value=str(entry), limit="finite > 0",
            )
        return RiskCheck(name="entry_valid", status=CheckStatus.PASS)

    def _check_sl_presence(self, ctx: _CheckContext) -> RiskCheck:
        p = ctx.proposal
        sl = p.stop_loss
        if not _finite(sl) or sl <= 0:
            return RiskCheck(
                name="sl_presence", status=CheckStatus.FAIL,
                reason=f"stop loss missing/invalid: {sl}", observed_value=str(sl), limit="finite > 0",
            )
        if p.direction is AgentDirection.BUY and not sl < p.entry_price:
            return RiskCheck(
                name="sl_presence", status=CheckStatus.FAIL,
                reason=f"SL {sl} not below entry {p.entry_price} for BUY",
                observed_value=sl, limit=f"< {p.entry_price}",
            )
        if p.direction is AgentDirection.SELL and not sl > p.entry_price:
            return RiskCheck(
                name="sl_presence", status=CheckStatus.FAIL,
                reason=f"SL {sl} not above entry {p.entry_price} for SELL",
                observed_value=sl, limit=f"> {p.entry_price}",
            )
        if abs(sl - p.entry_price) <= 0:
            return RiskCheck(
                name="sl_presence", status=CheckStatus.FAIL,
                reason="zero-distance stop loss", observed_value=sl, limit="> 0 distance",
            )
        if ctx.spec is not None:
            minimum = min_stop_distance(ctx.spec)
            distance = abs(p.entry_price - sl)
            if distance < minimum * (1 - _REL_TOL):
                return RiskCheck(
                    name="sl_presence", status=CheckStatus.FAIL,
                    reason=f"SL distance {distance:.2f} below broker minimum {minimum:.2f}",
                    observed_value=round(distance, 4), limit=minimum,
                )
            if not _on_grid(sl, ctx.spec.tick_size):
                return RiskCheck(
                    name="sl_presence", status=CheckStatus.FAIL,
                    reason=f"SL {sl} not on tick grid {ctx.spec.tick_size}",
                    observed_value=sl, limit=ctx.spec.tick_size,
                )
        return RiskCheck(name="sl_presence", status=CheckStatus.PASS)

    def _check_tp_validity(self, ctx: _CheckContext) -> RiskCheck:
        p = ctx.proposal
        tp = p.take_profit
        if not _finite(tp) or tp <= 0:
            return RiskCheck(
                name="tp_validity", status=CheckStatus.FAIL,
                reason=f"take profit missing/invalid: {tp}", observed_value=str(tp), limit="finite > 0",
            )
        if p.direction is AgentDirection.BUY and not tp > p.entry_price:
            return RiskCheck(
                name="tp_validity", status=CheckStatus.FAIL,
                reason=f"TP {tp} not above entry {p.entry_price} for BUY",
                observed_value=tp, limit=f"> {p.entry_price}",
            )
        if p.direction is AgentDirection.SELL and not tp < p.entry_price:
            return RiskCheck(
                name="tp_validity", status=CheckStatus.FAIL,
                reason=f"TP {tp} not below entry {p.entry_price} for SELL",
                observed_value=tp, limit=f"< {p.entry_price}",
            )
        if abs(tp - p.entry_price) <= 0:
            return RiskCheck(
                name="tp_validity", status=CheckStatus.FAIL,
                reason="zero-distance take profit", observed_value=tp, limit="> 0 distance",
            )
        if ctx.spec is not None:
            minimum = min_stop_distance(ctx.spec)
            distance = abs(p.entry_price - tp)
            if distance < minimum * (1 - _REL_TOL):
                return RiskCheck(
                    name="tp_validity", status=CheckStatus.FAIL,
                    reason=f"TP distance {distance:.2f} below broker minimum {minimum:.2f}",
                    observed_value=round(distance, 4), limit=minimum,
                )
            if not _on_grid(tp, ctx.spec.tick_size):
                return RiskCheck(
                    name="tp_validity", status=CheckStatus.FAIL,
                    reason=f"TP {tp} not on tick grid {ctx.spec.tick_size}",
                    observed_value=tp, limit=ctx.spec.tick_size,
                )
        return RiskCheck(name="tp_validity", status=CheckStatus.PASS)

    def _check_volume_limits(self, ctx: _CheckContext) -> RiskCheck:
        volume = ctx.proposal.suggested_volume
        if not _finite(volume) or volume <= 0:
            return RiskCheck(
                name="volume_limits", status=CheckStatus.FAIL,
                reason=f"volume invalid: {volume}", observed_value=str(volume), limit="> 0",
            )
        if ctx.spec is None:
            return RiskCheck(  # already failed in symbol_restriction; keep explicit
                name="volume_limits", status=CheckStatus.FAIL,
                reason="no symbol metadata — cannot verify volume limits",
                observed_value=volume, limit="SymbolSpec required",
            )
        spec = ctx.spec
        if volume < spec.volume_min * (1 - _REL_TOL):
            return RiskCheck(
                name="volume_limits", status=CheckStatus.FAIL,
                reason=f"volume {volume} below broker minimum {spec.volume_min}",
                observed_value=volume, limit=spec.volume_min,
            )
        if not _at_or_below(volume, spec.volume_max):
            return RiskCheck(
                name="volume_limits", status=CheckStatus.FAIL,
                reason=f"volume {volume} above broker maximum {spec.volume_max}",
                observed_value=volume, limit=spec.volume_max,
            )
        if spec.volume_step > 0 and not _on_grid(volume, spec.volume_step):
            return RiskCheck(
                name="volume_limits", status=CheckStatus.FAIL,
                reason=f"volume {volume} not aligned to step {spec.volume_step}",
                observed_value=volume, limit=spec.volume_step,
            )
        return RiskCheck(name="volume_limits", status=CheckStatus.PASS)

    def _check_max_risk_per_trade(self, ctx: _CheckContext) -> RiskCheck:
        """INDEPENDENT risk calculation — the proposal's own numbers are
        evidence, never authority:

            loss_per_lot = |entry - SL| / tick_size * tick_value
            actual_risk  = volume * loss_per_lot
        """
        p = ctx.proposal
        if ctx.spec is None or ctx.spec.tick_size <= 0 or ctx.spec.tick_value <= 0:
            return RiskCheck(
                name="max_risk_per_trade", status=CheckStatus.FAIL,
                reason="symbol metadata missing/invalid — cannot compute risk",
                observed_value=None, limit="tick_size & tick_value required",
            )
        loss_per_lot = abs(p.entry_price - p.stop_loss) / ctx.spec.tick_size * ctx.spec.tick_value
        actual_risk = p.suggested_volume * loss_per_lot
        if not math.isfinite(actual_risk):
            # an earlier check (entry/SL/volume) already FAILed for the bad
            # input; the risk calc still must never emit a non-finite number
            return RiskCheck(
                name="max_risk_per_trade", status=CheckStatus.FAIL,
                reason="risk calculation produced a non-finite value (invalid inputs)",
                observed_value=None, limit=None,
            )
        ctx.risk_amount = actual_risk

        if self.config.max_risk_per_trade_amount is not None:
            limit = self.config.max_risk_per_trade_amount
            limit_label = "configured amount"
        elif self.config.max_risk_per_trade_pct is not None:
            equity = ctx.equity_evidence()
            if equity is None:
                return RiskCheck(
                    name="max_risk_per_trade", status=CheckStatus.FAIL,
                    reason="equity evidence missing — cannot compute the risk budget",
                    observed_value=round(actual_risk, 2), limit="equity required",
                )
            limit = equity * self.config.max_risk_per_trade_pct / 100.0
            limit_label = f"{self.config.max_risk_per_trade_pct}% of equity {equity:.2f}"
        else:
            return RiskCheck(
                name="max_risk_per_trade", status=CheckStatus.FAIL,
                reason="risk limit not configured (fail closed)",
                observed_value=round(actual_risk, 2), limit="risk limit required",
            )

        if not _at_or_below(actual_risk, limit):
            return RiskCheck(
                name="max_risk_per_trade", status=CheckStatus.FAIL,
                reason=f"actual risk {actual_risk:.2f} exceeds limit ({limit_label})",
                observed_value=round(actual_risk, 2), limit=round(limit, 2),
            )
        if limit > 0 and actual_risk >= limit * 0.9:
            return RiskCheck(  # within limits, close to the budget — advisory
                name="max_risk_per_trade", status=CheckStatus.WARN, severity=CheckSeverity.ADVISORY,
                reason=f"risk {actual_risk:.2f} at >=90% of the limit ({limit_label})",
                observed_value=round(actual_risk, 2), limit=round(limit, 2),
            )
        return RiskCheck(
            name="max_risk_per_trade", status=CheckStatus.PASS,
            observed_value=round(actual_risk, 2), limit=round(limit, 2),
        )

    def _check_max_total_exposure(self, ctx: _CheckContext) -> RiskCheck:
        """Exposure = current open notional (AccountState evidence) plus the
        proposal's notional (volume × entry / tick_size × tick_value — the
        contract value implied by the verified symbol metadata)."""
        p = ctx.proposal
        if ctx.spec is None or ctx.spec.tick_size <= 0:
            return RiskCheck(
                name="max_total_exposure", status=CheckStatus.FAIL,
                reason="symbol metadata missing — cannot compute exposure",
            )
        if ctx.account_state.open_positions_notional is None:
            return RiskCheck(
                name="max_total_exposure", status=CheckStatus.FAIL,
                reason="current exposure evidence missing (open_positions_notional)",
                observed_value=None, limit="evidence required (never assumed zero)",
            )
        if self.config.max_total_exposure is None:
            return RiskCheck(
                name="max_total_exposure", status=CheckStatus.FAIL,
                reason="max_total_exposure not configured (fail closed)",
                observed_value=None, limit="must be configured",
            )
        current = ctx.account_state.open_positions_notional
        proposed = p.suggested_volume * (p.entry_price / ctx.spec.tick_size) * ctx.spec.tick_value
        total = current + proposed
        if not math.isfinite(total):
            return RiskCheck(
                name="max_total_exposure", status=CheckStatus.FAIL,
                reason="exposure calculation produced a non-finite value (invalid inputs)",
                observed_value=None, limit=None,
            )
        limit = self.config.max_total_exposure
        ctx.exposure = {
            "current": round(current, 2),
            "proposed": round(proposed, 2),
            "total": round(total, 2),
            "limit": limit,
        }
        if not _at_or_below(total, limit):
            return RiskCheck(
                name="max_total_exposure", status=CheckStatus.FAIL,
                reason=f"total exposure {total:.2f} exceeds limit {limit:.2f}",
                observed_value=round(total, 2), limit=limit,
            )
        if total >= limit * 0.9:
            return RiskCheck(
                name="max_total_exposure", status=CheckStatus.WARN, severity=CheckSeverity.ADVISORY,
                reason=f"total exposure {total:.2f} at >=90% of the limit {limit:.2f}",
                observed_value=round(total, 2), limit=limit,
            )
        return RiskCheck(
            name="max_total_exposure", status=CheckStatus.PASS,
            observed_value=round(total, 2), limit=limit,
        )

    def _check_daily_loss_limit(self, ctx: _CheckContext) -> RiskCheck:
        """Units: ABSOLUTE MONETARY loss (RiskState.daily_loss, positive =
        loss today).  Limit resolution: RiskState.daily_loss_limit ->
        config.daily_loss_limit -> config.daily_loss_limit_pct × equity.
        Percent is only ever applied to equity evidence — units are never
        mixed."""
        state = ctx.risk_state
        if state.daily_loss_limit is not None:
            limit, label = state.daily_loss_limit, "risk_state.daily_loss_limit"
        elif self.config.daily_loss_limit is not None:
            limit, label = self.config.daily_loss_limit, "configured amount"
        elif self.config.daily_loss_limit_pct is not None:
            equity = ctx.equity_evidence()
            if equity is None:
                return RiskCheck(
                    name="daily_loss_limit", status=CheckStatus.FAIL,
                    reason="equity evidence missing — cannot derive the daily loss limit",
                    observed_value=round(state.daily_loss, 2), limit="equity required",
                )
            limit = equity * self.config.daily_loss_limit_pct / 100.0
            label = f"{self.config.daily_loss_limit_pct}% of equity {equity:.2f}"
        else:
            return RiskCheck(
                name="daily_loss_limit", status=CheckStatus.FAIL,
                reason="daily loss limit not configured anywhere (fail closed)",
                observed_value=round(state.daily_loss, 2), limit="limit required",
            )
        if state.daily_loss >= limit - abs(limit) * _REL_TOL:
            return RiskCheck(
                name="daily_loss_limit", status=CheckStatus.FAIL,
                reason=f"daily loss {state.daily_loss:.2f} reached limit ({label})",
                observed_value=round(state.daily_loss, 2), limit=round(limit, 2),
            )
        return RiskCheck(
            name="daily_loss_limit", status=CheckStatus.PASS,
            observed_value=round(state.daily_loss, 2), limit=round(limit, 2),
        )

    def _check_consecutive_loss_protection(self, ctx: _CheckContext) -> RiskCheck:
        losses = ctx.risk_state.consecutive_losses
        limit = self.config.max_consecutive_losses
        if losses >= limit:
            return RiskCheck(
                name="consecutive_loss_protection", status=CheckStatus.FAIL,
                reason=f"consecutive losses {losses} reached stand-down limit {limit}",
                observed_value=losses, limit=limit,
            )
        return RiskCheck(
            name="consecutive_loss_protection", status=CheckStatus.PASS,
            observed_value=losses, limit=limit,
        )

    def _check_max_open_positions(self, ctx: _CheckContext) -> RiskCheck:
        open_positions = ctx.risk_state.open_positions
        limit = self.config.max_open_positions
        if open_positions >= limit:
            return RiskCheck(
                name="max_open_positions", status=CheckStatus.FAIL,
                reason=f"open positions {open_positions} at/above limit {limit}",
                observed_value=open_positions, limit=limit,
            )
        return RiskCheck(
            name="max_open_positions", status=CheckStatus.PASS,
            observed_value=open_positions, limit=limit,
        )

    def _check_max_pending_orders(self, ctx: _CheckContext) -> RiskCheck:
        pending = ctx.risk_state.pending_orders
        limit = self.config.max_pending_orders
        if pending >= limit:
            return RiskCheck(
                name="max_pending_orders", status=CheckStatus.FAIL,
                reason=f"pending orders {pending} at/above limit {limit}",
                observed_value=pending, limit=limit,
            )
        return RiskCheck(
            name="max_pending_orders", status=CheckStatus.PASS,
            observed_value=pending, limit=limit,
        )

    def _check_max_spread(self, ctx: _CheckContext) -> RiskCheck:
        spread = ctx.account_state.spread_points
        if spread is None:
            return RiskCheck(
                name="max_spread", status=CheckStatus.FAIL,
                reason="spread evidence missing (spread_points) — never assumed zero",
                observed_value=None, limit="evidence required",
            )
        if self.config.max_spread_points is None:
            return RiskCheck(
                name="max_spread", status=CheckStatus.FAIL,
                reason="spread limit not configured (fail closed)",
                observed_value=spread, limit="must be configured",
            )
        if not _at_or_below(spread, self.config.max_spread_points):
            return RiskCheck(
                name="max_spread", status=CheckStatus.FAIL,
                reason=f"spread {spread} points exceeds limit {self.config.max_spread_points}",
                observed_value=spread, limit=self.config.max_spread_points,
            )
        return RiskCheck(
            name="max_spread", status=CheckStatus.PASS,
            observed_value=spread, limit=self.config.max_spread_points,
        )

    def _check_margin_safety(self, ctx: _CheckContext) -> RiskCheck:
        """Optional policy (spec §18): minimum margin level in percent.
        Policy off -> NOT_EVALUATED (advisory).  Policy on + evidence
        missing -> FAIL (fail closed).  No broker margin formulas are
        invented here — the caller supplies the reported margin level."""
        if self.config.min_free_margin_percent is None:
            return RiskCheck(
                name="margin_safety", status=CheckStatus.NOT_EVALUATED,
                severity=CheckSeverity.ADVISORY,
                reason="margin policy not configured",
            )
        level = ctx.account_state.margin_level
        if level is None:
            return RiskCheck(
                name="margin_safety", status=CheckStatus.FAIL,
                reason="margin level evidence missing but policy requires it",
                observed_value=None, limit=self.config.min_free_margin_percent,
            )
        if level < self.config.min_free_margin_percent * (1 - _REL_TOL):
            return RiskCheck(
                name="margin_safety", status=CheckStatus.FAIL,
                reason=f"margin level {level:.1f}% below minimum {self.config.min_free_margin_percent}%",
                observed_value=round(level, 2), limit=self.config.min_free_margin_percent,
            )
        return RiskCheck(
            name="margin_safety", status=CheckStatus.PASS,
            observed_value=round(level, 2), limit=self.config.min_free_margin_percent,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reasons_for(action: RiskAction, checks: list[RiskCheck]) -> list[str]:
        if action is RiskAction.APPROVED:
            return ["all safety checks passed"]
        failures = [c for c in checks if c.status is CheckStatus.FAIL]
        return [f"{c.name}: {c.reason}" for c in failures]

    @staticmethod
    def _decision_id(proposal: TradeProposal, action: RiskAction, checks: list[RiskCheck]) -> str:
        """Deterministic id: sha256 of the inputs' decisive content."""
        failed = ",".join(c.name for c in checks if c.status is CheckStatus.FAIL)
        seed = f"{proposal.decision_id}|{proposal.fingerprint}|{action.value}|{failed}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def _emit(self, decision: RiskDecision) -> None:
        """Internal observability only — never a network/Telegram send."""
        if self._event_sink is None:
            return
        if decision.action is RiskAction.EMERGENCY_STOP:
            event_type = (
                RiskEventType.KILL_SWITCH_ACTIVE
                if decision.kill_switch_active and not decision.emergency_stop_active
                else RiskEventType.EMERGENCY_STOP
            )
        elif decision.action is RiskAction.APPROVED:
            event_type = RiskEventType.RISK_APPROVED
        else:
            event_type = RiskEventType.RISK_REJECTED
        self._event_sink(RiskEvent.from_decision(decision, event_type))


class _CheckContext:
    """Mutable per-evaluation scratchpad (never touches the proposal)."""

    def __init__(
        self,
        *,
        proposal: TradeProposal,
        risk_state: RiskState,
        account_state: AccountState,
        spec: SymbolSpec | None,
        config: RiskGateConfig,
    ) -> None:
        self.proposal = proposal
        self.risk_state = risk_state
        self.account_state = account_state
        self.spec = spec
        self.config = config
        self.risk_amount: float | None = None
        self.exposure: dict | None = None

    def equity_evidence(self) -> float | None:
        """Equity from RiskState or AccountState; both present must agree
        (the account_safety check already failed them out if they don't)."""
        if self.risk_state.equity is not None and self.risk_state.equity > 0:
            return self.risk_state.equity
        if self.account_state.equity is not None and self.account_state.equity > 0:
            return self.account_state.equity
        return None
