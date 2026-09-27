"""HardRiskGate tests — the Phase-4 safety barrier (spec §2-§16, §20-§23, §27).

Every matrix point from the Phase-4 specification is pinned with an exact
action and check name, so operators can grep decisions for these strings.
"""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, RiskAction
from app.risk import (
    AccountState,
    CheckSeverity,
    CheckStatus,
    HardRiskGate,
    RiskGateConfig,
    RiskState,
)
from app.risk.events import RiskEvent, RiskEventType
from tests.unit.agents.scenarios import gold_symbol_spec
from tests.unit.risk.conftest import make_account, make_gate, make_proposal


def smuggle(field: str, value):
    """Build a proposal with a value the model's own validation would
    reject (model_copy skips validation) — proving the GATE independently
    catches what a future model regression might let through."""
    return make_proposal().model_copy(update={field: value})


class TestApprovePath:
    def test_valid_buy_approves(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED
        assert d.approved
        assert d.failed_checks == []
        # independently computed risk: 5.0 / 0.01 * 1.0 * 0.08 = $40
        assert d.risk_amount == pytest.approx(40.0)
        # exposure breakdown is recorded
        assert d.exposure["proposed"] == pytest.approx(2650.2 / 0.01 * 0.08, rel=1e-6)
        assert d.exposure["total"] == pytest.approx(d.exposure["proposed"])
        assert d.exposure["limit"] == 1_000_000.0

    def test_approved_decision_shows_every_check(self, gate, proposal, risk_state, account):
        """§21: an approved decision lists all safety checks that ran."""
        d = gate.evaluate(proposal, risk_state, account)
        names = [c.name for c in d.checks]
        from app.risk.engine import IMPLEMENTED_CHECKS

        assert set(names) == set(IMPLEMENTED_CHECKS)  # all 18, nothing hidden
        assert len(names) == len(set(names))
        assert all(c.status is not CheckStatus.FAIL for c in d.checks)
        # margin policy is off -> advisory NOT_EVALUATED, never blocking
        margin = next(c for c in d.checks if c.name == "margin_safety")
        assert margin.status is CheckStatus.NOT_EVALUATED
        assert margin.severity is CheckSeverity.ADVISORY

    def test_valid_sell_approves(self, proposal, risk_state, account):
        gate = make_gate()
        sell = make_proposal(
            direction=AgentDirection.SELL, entry_price=2650.0,
            stop_loss=2655.0, take_profit=2640.0,
        )
        d = gate.evaluate(sell, risk_state, account)
        assert d.action is RiskAction.APPROVED
        assert d.risk_amount == pytest.approx(40.0)


class TestKillSwitch:
    def test_active_kill_switch_overrides_valid_proposal(self, proposal, risk_state):
        gate = make_gate()
        account = make_account(kill_switch_active=True)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.EMERGENCY_STOP
        assert d.kill_switch_active is True
        assert d.reasons == ["kill_switch: global kill switch is active"]

    def test_inactive_kill_switch_ignores(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED
        assert d.kill_switch_active is False

    def test_kill_switch_stops_evaluation(self, proposal, risk_state):
        """Everything else is broken too — the halt still wins and the
        remaining checks are recorded NOT_EVALUATED, never hidden."""
        gate = make_gate()
        broken = make_proposal(symbol="EURUSD", suggested_volume=999.0)
        account = make_account(kill_switch_active=True, trade_allowed=False)
        d = gate.evaluate(broken, RiskState(equity=10_000.0, daily_loss=999.0), account)
        assert d.action is RiskAction.EMERGENCY_STOP
        assert d.failed_checks == ["kill_switch"]
        assert len(d.not_evaluated_checks) == len(d.checks) - 1


class TestEmergencyStop:
    def test_active_emergency_stop(self, proposal, risk_state):
        gate = make_gate()
        account = make_account(emergency_stop_active=True)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.EMERGENCY_STOP
        assert d.emergency_stop_active is True
        assert d.reasons == ["emergency_stop: operator emergency stop is active"]

    def test_inactive_emergency_stop(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_emergency_stop_outranks_kill_switch_in_reporting(self, proposal, risk_state):
        """Both active -> EMERGENCY_STOP outcome, emergency reason first."""
        gate = make_gate()
        account = make_account(emergency_stop_active=True, kill_switch_active=True)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.EMERGENCY_STOP
        assert d.failed_checks == ["emergency_stop"]  # emergency consumed the halt


class TestTradingPermission:
    def test_safe_default_rejects_everything(self, proposal, risk_state, account):
        """§19: a default-constructed gate has trading_enabled=False."""
        gate = HardRiskGate(symbol_specs={"XAUUSD": gold_symbol_spec()})
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert d.failed_checks == ["trading_enabled"]
        assert "not enabled" in d.reasons[0]

    def test_trade_disallowed_rejects(self, proposal, risk_state):
        gate = make_gate()
        account = make_account(trade_allowed=False)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert d.failed_checks == ["account_safety"]
        assert "trading_not_allowed" in d.reasons[0]

    def test_trade_allowed_proceeds(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_permission_absence_is_not_permission(self):
        """AccountState defaults trade_allowed=False — building an account
        without affirmatively enabling trading cannot approve."""
        assert AccountState().trade_allowed is False


class TestSymbolProtection:
    @pytest.mark.parametrize("symbol", ["XAUUSD", "XAUUSDm", "XAUUSD.a", "GOLD", "GOLDm", "XAU/USD"])
    def test_gold_symbols_pass(self, symbol, risk_state, account):
        specs = {symbol: gold_symbol_spec().model_copy(update={"name": symbol})}
        gate = HardRiskGate(
            RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0),
            symbol_specs=specs,
        )
        d = gate.evaluate(make_proposal(symbol=symbol), risk_state, account)
        assert d.action is RiskAction.APPROVED, d.reasons

    @pytest.mark.parametrize("symbol", ["EURUSD", "GBPUSD", "XAGUSD", "XPTUSD", "XPDUSD", "GOLDEN", "GC=F"])
    def test_non_gold_symbols_never_pass(self, symbol, risk_state, account):
        specs = {symbol: gold_symbol_spec().model_copy(update={"name": symbol})}
        gate = HardRiskGate(
            RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0),
            symbol_specs=specs,
        )
        d = gate.evaluate(make_proposal(symbol=symbol), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "symbol_restriction" in d.failed_checks
        assert "not a gold symbol" in d.reasons[0]

    def test_gold_symbol_without_registered_spec_fails_closed(self, risk_state, account):
        """Gold-looking symbol but no verified metadata -> missing evidence."""
        gate = HardRiskGate(RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0))
        d = gate.evaluate(make_proposal(), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "symbol_restriction" in d.failed_checks
        assert "no verified symbol metadata" in d.reasons[0]


class TestDirection:
    def test_buy_and_sell_are_executable(self, gate, risk_state, account):
        for direction, sl, tp in (
            (AgentDirection.BUY, 2645.2, 2660.2),
            (AgentDirection.SELL, 2655.0, 2640.0),
        ):
            p = make_proposal(direction=direction, stop_loss=sl, take_profit=tp)
            if direction is AgentDirection.SELL:
                p = make_proposal(direction=direction, entry_price=2650.0,
                                  stop_loss=2655.0, take_profit=2640.0)
            d = gate.evaluate(p, risk_state, account)
            assert d.action is RiskAction.APPROVED, d.reasons

    def test_neutral_never_executable(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(direction=AgentDirection.NEUTRAL), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "direction_valid" in d.failed_checks
        assert "direction_not_executable" in d.reasons[0]

    @pytest.mark.parametrize("action_name", ["HOLD", "ABORT"])
    def test_hold_abort_decisions_never_executable(self, action_name, gate, risk_state, account):
        """HOLD/ABORT are DecisionAction values and can never legitimately
        reach a proposal — but if a future bug let one through
        (model_construct bypasses validation), the gate still rejects it."""
        from app.core.enums import DecisionAction

        invalid = make_proposal().model_construct(
            **{**make_proposal().model_dump(),
               "direction": getattr(DecisionAction, action_name)}
        )
        d = gate.evaluate(invalid, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "direction_valid" in d.failed_checks


class TestGeometry:
    def test_valid_buy_passes(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_invalid_buy_sl_above_entry(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(stop_loss=2655.0), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "sl_presence" in d.failed_checks

    def test_invalid_buy_tp_below_entry(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(take_profit=2645.0), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "tp_validity" in d.failed_checks

    def test_invalid_sell_sl_below_entry(self, gate, risk_state, account):
        sell = make_proposal(direction=AgentDirection.SELL, entry_price=2650.0,
                             stop_loss=2645.0, take_profit=2640.0)
        d = gate.evaluate(sell, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "sl_presence" in d.failed_checks

    def test_invalid_sell_tp_above_entry(self, gate, risk_state, account):
        sell = make_proposal(direction=AgentDirection.SELL, entry_price=2650.0,
                             stop_loss=2655.0, take_profit=2660.0)
        d = gate.evaluate(sell, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "tp_validity" in d.failed_checks

    @pytest.mark.parametrize("field", ["entry_price", "stop_loss", "take_profit"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -5.0])
    def test_non_finite_or_non_positive_prices_rejected(
        self, field, bad, gate, risk_state, account
    ):
        """The proposal model itself rejects these — the gate must too,
        independently (smuggled past model validation on purpose)."""
        d = gate.evaluate(smuggle(field, bad), risk_state, account)
        assert d.action is RiskAction.REJECTED
        blocking = {"entry_price": "entry_valid", "stop_loss": "sl_presence",
                    "take_profit": "tp_validity"}[field]
        assert blocking in d.failed_checks

    def test_zero_distance_sl_rejected(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(stop_loss=2650.2), risk_state, account)
        assert d.action is RiskAction.REJECTED

    def test_below_broker_minimum_distance_rejected(self, gate, risk_state, account):
        """stops_level 20 points -> 0.20 minimum; SL 0.10 away is invalid."""
        d = gate.evaluate(make_proposal(stop_loss=2650.1), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "sl_presence" in d.failed_checks
        assert "below broker minimum" in d.reasons[0]

    def test_off_tick_grid_rejected(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(stop_loss=2645.205), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "not on tick grid" in d.reasons[0]

    def test_gate_never_repairs_geometry(self, gate, risk_state, account):
        """§16: the gate rejects invalid geometry; it does not fix it."""
        bad = make_proposal(stop_loss=2655.0)
        before = bad.model_dump_json()
        d = gate.evaluate(bad, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert bad.model_dump_json() == before
        assert bad.stop_loss == 2655.0  # untouched


class TestRiskPerTrade:
    def test_below_maximum_passes(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)  # $40 of $50 budget
        assert d.action is RiskAction.APPROVED
        assert d.risk_amount == pytest.approx(40.0)

    def test_exactly_maximum_passes_within_tolerance(self, gate, risk_state, account):
        """0.10 lots -> $50 risk = exactly the 0.5% budget -> passes (WARN)."""
        d = gate.evaluate(make_proposal(suggested_volume=0.10), risk_state, account)
        assert d.action is RiskAction.APPROVED
        assert d.risk_amount == pytest.approx(50.0)
        check = next(c for c in d.checks if c.name == "max_risk_per_trade")
        assert check.status is CheckStatus.WARN  # at >=90% of the budget

    def test_above_maximum_rejected(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(suggested_volume=0.50), risk_state, account)  # $250
        assert d.action is RiskAction.REJECTED
        assert "max_risk_per_trade" in d.failed_checks
        assert d.risk_amount == pytest.approx(250.0)  # the independent number

    def test_proposal_lie_about_risk_is_caught(self, gate, risk_state, account):
        """§10: the proposal claims risk_amount=$40 but its own geometry says
        $500 — the gate computes independently and rejects."""
        lying = make_proposal(suggested_volume=1.0)
        assert lying.sizing.risk_amount == pytest.approx(40.0)  # the claim
        d = gate.evaluate(lying, risk_state, account)
        assert d.risk_amount == pytest.approx(500.0)  # the truth
        assert d.action is RiskAction.REJECTED
        assert "exceeds limit" in d.reasons[0]

    def test_missing_equity_evidence_fails_closed(self, proposal, account):
        gate = make_gate()
        d = gate.evaluate(proposal, RiskState(), account.model_copy(update={"equity": None}))
        assert d.action is RiskAction.REJECTED
        assert "equity evidence missing" in d.reasons[0]

    def test_monetary_limit_override(self, proposal, risk_state, account):
        gate = make_gate(max_risk_per_trade_amount=30.0)  # $40 risk > $30
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        gate = make_gate(max_risk_per_trade_amount=100.0)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_risk_never_rounded_upward_into_approval(self, gate, risk_state, account):
        """$50.01 vs a $50.00 budget must reject (no tolerance abuse)."""
        d = gate.evaluate(
            make_proposal(suggested_volume=0.10002), risk_state, account
        )
        assert d.risk_amount > 50.0
        assert d.action is RiskAction.REJECTED


class TestExposure:
    def test_below_limit_passes(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)  # ~26.5k of 1M
        assert d.action is RiskAction.APPROVED
        assert d.exposure["total"] == pytest.approx(d.exposure["proposed"])

    def test_exactly_at_limit_passes(self, proposal, risk_state):
        gate = make_gate()  # limit 1_000_000
        proposed = 2650.2 / 0.01 * 0.08  # 21,201.6
        account = make_account(open_positions_notional=1_000_000.0 - proposed)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED
        check = next(c for c in d.checks if c.name == "max_total_exposure")
        assert check.status is CheckStatus.WARN  # at the limit

    def test_above_limit_rejected(self, proposal, risk_state):
        gate = make_gate()
        account = make_account(open_positions_notional=1_000_000.0)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "max_total_exposure" in d.failed_checks
        assert d.exposure["total"] > d.exposure["limit"]

    def test_missing_exposure_evidence_fails_closed(self, gate, proposal, risk_state):
        account = make_account(open_positions_notional=None)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "exposure evidence missing" in d.reasons[0]

    def test_unconfigured_limit_fails_closed(self, proposal, risk_state, account):
        gate = HardRiskGate(
            RiskGateConfig(trading_enabled=True),  # no max_total_exposure
            symbol_specs={"XAUUSD": gold_symbol_spec()},
        )
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "max_total_exposure" in d.failed_checks
        assert "not configured" in d.reasons[0]


class TestDailyLoss:
    def test_below_limit_passes(self, gate, proposal, account):
        state = RiskState(equity=10_000.0, daily_loss=199.99)  # limit 200 (2%)
        d = gate.evaluate(proposal, state, account)
        assert d.action is RiskAction.APPROVED

    def test_exactly_at_limit_rejected(self, gate, proposal, account):
        state = RiskState(equity=10_000.0, daily_loss=200.0)
        d = gate.evaluate(proposal, state, account)
        assert d.action is RiskAction.REJECTED
        assert "daily_loss_limit" in d.failed_checks

    def test_above_limit_rejected(self, gate, proposal, account):
        state = RiskState(equity=10_000.0, daily_loss=250.0)
        d = gate.evaluate(proposal, state, account)
        assert d.action is RiskAction.REJECTED
        assert "daily_loss_limit" in d.failed_checks

    def test_explicit_monetary_limit_wins_over_pct(self, gate, proposal, account):
        state = RiskState(equity=10_000.0, daily_loss=50.0, daily_loss_limit=60.0)
        d = gate.evaluate(proposal, state, account)
        assert d.action is RiskAction.APPROVED
        state = RiskState(equity=10_000.0, daily_loss=60.0, daily_loss_limit=60.0)
        d = gate.evaluate(proposal, state, account)
        assert d.action is RiskAction.REJECTED

    def test_configured_monetary_limit_used_when_state_has_none(self, proposal, account):
        gate = make_gate(daily_loss_limit=100.0)
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, daily_loss=99.0), account)
        assert d.action is RiskAction.APPROVED
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, daily_loss=100.0), account)
        assert d.action is RiskAction.REJECTED

    def test_units_are_absolute_monetary(self, gate, proposal, account):
        """daily_loss is monetary, never a percentage — the check's observed
        value is the monetary loss compared with a monetary limit."""
        state = RiskState(equity=10_000.0, daily_loss=150.0)
        d = gate.evaluate(proposal, state, account)
        check = next(c for c in d.checks if c.name == "daily_loss_limit")
        assert check.observed_value == pytest.approx(150.0)  # dollars, not %
        assert check.limit == pytest.approx(200.0)


class TestConsecutiveLosses:
    def test_below_limit_passes(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, consecutive_losses=2), account)
        assert d.action is RiskAction.APPROVED

    def test_at_limit_rejected(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, consecutive_losses=3), account)
        assert d.action is RiskAction.REJECTED
        assert "consecutive_loss_protection" in d.failed_checks

    def test_above_limit_rejected(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, consecutive_losses=7), account)
        assert d.action is RiskAction.REJECTED

    def test_losses_never_increase_risk(self, gate, proposal, account):
        """§13: no martingale/recovery — losing streaks can only block,
        never change the computed risk of a proposal that still passes."""
        clean = gate.evaluate(proposal, RiskState(equity=10_000.0), account)
        losing = gate.evaluate(
            proposal, RiskState(equity=10_000.0, consecutive_losses=2), account
        )
        assert clean.risk_amount == losing.risk_amount  # identical math
        assert clean.action is losing.action is RiskAction.APPROVED


class TestPositionLimits:
    def test_open_positions_below_limit(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, open_positions=0), account)
        assert d.action is RiskAction.APPROVED

    def test_open_positions_at_limit(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, open_positions=1), account)
        assert d.action is RiskAction.REJECTED
        assert "max_open_positions" in d.failed_checks

    def test_open_positions_above_limit(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, open_positions=2), account)
        assert d.action is RiskAction.REJECTED

    def test_pending_orders_below_limit(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, pending_orders=1), account)
        assert d.action is RiskAction.APPROVED

    def test_pending_orders_at_limit(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, pending_orders=2), account)
        assert d.action is RiskAction.REJECTED
        assert "max_pending_orders" in d.failed_checks

    def test_pending_orders_above_limit(self, gate, proposal, account):
        d = gate.evaluate(proposal, RiskState(equity=10_000.0, pending_orders=3), account)
        assert d.action is RiskAction.REJECTED

    def test_inconsistent_count_evidence_fails_closed(self, gate, proposal, risk_state):
        """RiskState says 0 open positions, AccountState says 3 — the gate
        does not pick the convenient number."""
        account = make_account(open_positions=3)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "inconsistent open_positions" in d.reasons[0]


class TestSpread:
    def test_acceptable_spread_passes(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)  # 20 of 50
        assert d.action is RiskAction.APPROVED

    def test_exactly_at_limit_passes(self, gate, proposal, risk_state):
        account = make_account(spread_points=50.0)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_above_limit_rejected(self, gate, proposal, risk_state):
        account = make_account(spread_points=51.0)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "max_spread" in d.failed_checks
        assert "exceeds limit" in d.reasons[0]

    def test_missing_spread_evidence_rejected(self, gate, proposal, risk_state):
        """§15: missing spread is NEVER substituted with zero."""
        account = make_account(spread_points=None)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "spread evidence missing" in d.reasons[0]

    def test_unconfigured_limit_rejected(self, proposal, risk_state, account):
        gate = make_gate(max_spread_points=None)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "spread limit not configured" in d.reasons[0]


class TestVolume:
    def test_minimum_volume_passes(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(suggested_volume=0.01), risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_maximum_volume_at_limit(self, gate, risk_state, account):
        """Volume 100 (max) is step-valid; risk will still be checked."""
        d = gate.evaluate(make_proposal(suggested_volume=100.0), risk_state, account)
        assert "volume_limits" not in d.failed_checks  # volume itself is fine
        assert d.action is RiskAction.REJECTED  # but the risk is insane
        assert "max_risk_per_trade" in d.failed_checks

    def test_above_maximum_rejected(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(suggested_volume=100.01), risk_state, account)
        assert "volume_limits" in d.failed_checks

    def test_below_minimum_rejected(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(suggested_volume=0.005), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "volume_limits" in d.failed_checks

    def test_off_step_rejected(self, gate, risk_state, account):
        d = gate.evaluate(make_proposal(suggested_volume=0.085), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "not aligned to step" in d.reasons[0]

    @pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
    def test_invalid_volumes_rejected(self, bad, gate, risk_state, account):
        """Smuggled past the proposal model's own validation on purpose."""
        d = gate.evaluate(smuggle("suggested_volume", bad), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "volume_limits" in d.failed_checks

    def test_gate_never_rounds_volume_into_validity(self, gate, risk_state, account):
        bad = make_proposal(suggested_volume=0.085)
        before = bad.model_dump_json()
        d = gate.evaluate(bad, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert bad.model_dump_json() == before
        assert bad.suggested_volume == 0.085  # not repaired


class TestAccountEvidence:
    def test_valid_account_passes(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.APPROVED

    def test_insufficient_equity_evidence_fails_closed(self, proposal, account):
        gate = make_gate()
        d = gate.evaluate(proposal, RiskState(), account.model_copy(update={"equity": None}))
        assert d.action is RiskAction.REJECTED

    def test_inconsistent_equity_evidence_fails_closed(self, proposal, account):
        gate = make_gate()
        state = RiskState(equity=10_000.0)
        conflicting = account.model_copy(update={"equity": 10_500.0})
        d = gate.evaluate(proposal, state, conflicting)
        assert d.action is RiskAction.REJECTED
        assert "inconsistent equity evidence" in d.reasons[0]

    def test_equity_from_account_state_when_risk_state_lacks_it(self, proposal, account):
        gate = make_gate()
        d = gate.evaluate(proposal, RiskState(), account)  # equity only on account
        assert d.action is RiskAction.APPROVED
        assert d.risk_amount == pytest.approx(40.0)


class TestMarginSafety:
    def test_policy_off_is_not_evaluated(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        check = next(c for c in d.checks if c.name == "margin_safety")
        assert check.status is CheckStatus.NOT_EVALUATED

    def test_policy_on_with_healthy_margin(self, proposal, risk_state, account):
        gate = make_gate(min_free_margin_percent=200.0)
        d = gate.evaluate(proposal, risk_state, account)  # margin_level 10k %
        assert d.action is RiskAction.APPROVED

    def test_policy_on_with_low_margin(self, proposal, risk_state):
        gate = make_gate(min_free_margin_percent=200.0)
        account = make_account(margin_level=150.0)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "margin_safety" in d.failed_checks

    def test_policy_on_with_missing_evidence(self, proposal, risk_state):
        gate = make_gate(min_free_margin_percent=200.0)
        account = make_account(margin_level=None)
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "margin level evidence missing" in d.reasons[0]


class TestPrecedence:
    """§20: emergency > kill switch > trading disabled > account > safety > approved."""

    def test_full_precedence_order(self, proposal):
        state = RiskState(equity=10_000.0, daily_loss=999.0)
        # 1. emergency stop wins over everything
        d = make_gate().evaluate(
            proposal, state,
            make_account(emergency_stop_active=True, kill_switch_active=True, trade_allowed=False),
        )
        assert (d.action, d.failed_checks[0]) == (RiskAction.EMERGENCY_STOP, "emergency_stop")
        # 2. kill switch wins over trading-disabled and safety failures
        d = make_gate().evaluate(
            proposal, state, make_account(kill_switch_active=True, trade_allowed=False)
        )
        assert (d.action, d.failed_checks[0]) == (RiskAction.EMERGENCY_STOP, "kill_switch")
        # 3. trading disabled wins over account/safety
        d = HardRiskGate(symbol_specs={"XAUUSD": gold_symbol_spec()}).evaluate(
            proposal, state, make_account(trade_allowed=False)
        )
        assert (d.action, d.failed_checks[0]) == (RiskAction.REJECTED, "trading_enabled")
        # 4. account rejection wins over safety failures
        d = make_gate().evaluate(proposal, state, make_account(trade_allowed=False))
        assert (d.action, d.failed_checks[0]) == (RiskAction.REJECTED, "account_safety")
        assert "daily_loss_limit" in d.not_evaluated_checks  # safety tier not reached
        # 5. safety failures all listed together
        d = make_gate().evaluate(proposal, state, make_account())
        assert d.action is RiskAction.REJECTED
        assert set(d.failed_checks) >= {"daily_loss_limit"}

    def test_halted_evaluation_records_not_evaluated(self, proposal):
        state = RiskState(equity=10_000.0)
        d = make_gate().evaluate(
            proposal, state, make_account(emergency_stop_active=True)
        )
        assert len(d.not_evaluated_checks) == len(d.checks) - 1
        assert all(
            "evaluation stopped" in c.reason for c in d.checks if c.status is CheckStatus.NOT_EVALUATED
        )


class TestAdversarialCombinations:
    """§27: valid BUY combined with each violation — the gate stays closed."""

    def test_valid_buy_plus_kill_switch(self, proposal, risk_state):
        d = make_gate().evaluate(proposal, risk_state, make_account(kill_switch_active=True))
        assert d.action is RiskAction.EMERGENCY_STOP

    def test_valid_buy_plus_emergency_stop(self, proposal, risk_state):
        d = make_gate().evaluate(proposal, risk_state, make_account(emergency_stop_active=True))
        assert d.action is RiskAction.EMERGENCY_STOP

    def test_valid_buy_plus_excessive_risk(self, proposal, risk_state, account):
        d = make_gate().evaluate(make_proposal(suggested_volume=0.5), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "max_risk_per_trade" in d.failed_checks

    def test_valid_buy_plus_excessive_exposure(self, proposal, risk_state):
        d = make_gate().evaluate(
            proposal, risk_state, make_account(open_positions_notional=1_000_000.0)
        )
        assert d.action is RiskAction.REJECTED
        assert "max_total_exposure" in d.failed_checks

    def test_valid_buy_plus_daily_loss_breached(self, proposal, account):
        d = make_gate().evaluate(
            proposal, RiskState(equity=10_000.0, daily_loss=250.0), account
        )
        assert d.action is RiskAction.REJECTED
        assert "daily_loss_limit" in d.failed_checks

    def test_valid_buy_plus_position_limit(self, proposal, account):
        d = make_gate().evaluate(
            proposal, RiskState(equity=10_000.0, open_positions=1), account
        )
        assert d.action is RiskAction.REJECTED
        assert "max_open_positions" in d.failed_checks

    def test_valid_buy_plus_invalid_symbol(self, risk_state, account):
        d = make_gate().evaluate(make_proposal(symbol="EURUSD"), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "symbol_restriction" in d.failed_checks

    def test_valid_buy_plus_invalid_sl(self, risk_state, account):
        d = make_gate().evaluate(make_proposal(stop_loss=2651.0), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "sl_presence" in d.failed_checks

    def test_valid_buy_plus_invalid_tp(self, risk_state, account):
        d = make_gate().evaluate(make_proposal(take_profit=2645.0), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "tp_validity" in d.failed_checks

    def test_valid_buy_plus_missing_spread(self, proposal, risk_state):
        d = make_gate().evaluate(proposal, risk_state, make_account(spread_points=None))
        assert d.action is RiskAction.REJECTED
        assert "max_spread" in d.failed_checks

    def test_valid_buy_plus_trade_disallowed(self, proposal, risk_state):
        d = make_gate().evaluate(proposal, risk_state, make_account(trade_allowed=False))
        assert d.action is RiskAction.REJECTED

    def test_everything_broken_at_once(self, risk_state):
        """Kitchen sink: every violation simultaneously — still closed, and
        the safety tier lists every failure (nothing hidden)."""
        broken = make_proposal(symbol="EURUSD", suggested_volume=500.0, stop_loss=2655.0)
        account = make_account(spread_points=999.0)
        state = RiskState(equity=10_000.0, daily_loss=999.0, consecutive_losses=9,
                          open_positions=9, pending_orders=9)
        d = make_gate().evaluate(broken, state, account)
        assert d.action is RiskAction.REJECTED
        for check in ("symbol_restriction", "sl_presence", "volume_limits",
                      "max_risk_per_trade", "daily_loss_limit",
                      "consecutive_loss_protection", "max_open_positions",
                      "max_pending_orders", "max_spread"):
            assert check in d.failed_checks, f"{check} missing from failures"


class TestNoProposalMutation:
    """§22: mandatory — the gate may reject, never repair or alter."""

    @pytest.mark.parametrize("scenario", ["approve", "reject", "emergency"])
    def test_proposal_byte_equivalent_after_evaluation(
        self, scenario, proposal, risk_state, account
    ):
        gate = make_gate()
        before = proposal.model_dump_json()
        if scenario == "approve":
            gate.evaluate(proposal, risk_state, account)
        elif scenario == "reject":
            gate.evaluate(proposal, RiskState(equity=10_000.0, daily_loss=999.0), account)
        else:
            gate.evaluate(proposal, risk_state, make_account(kill_switch_active=True))
        assert proposal.model_dump_json() == before

    def test_no_field_is_ever_changed(self, gate, proposal, risk_state, account):
        before = proposal.model_dump()
        gate.evaluate(make_proposal(suggested_volume=0.5), risk_state, account)
        gate.evaluate(proposal, risk_state, account)
        assert proposal.model_dump() == before


class TestAdversarialMath:
    def test_nan_entry_does_not_crash_risk_calc(self, gate, risk_state, account):
        d = gate.evaluate(smuggle("entry_price", float("nan")), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert d.risk_amount is None  # never a NaN sneaking into the decision

    def test_inf_sl_does_not_crash_risk_calc(self, gate, risk_state, account):
        d = gate.evaluate(smuggle("stop_loss", float("inf")), risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert d.risk_amount is None

    def test_zero_tick_value_spec_fails_closed(self, proposal, risk_state, account):
        broken_spec = gold_symbol_spec().model_copy(update={"tick_value": 0.0})
        gate = HardRiskGate(
            RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0),
            symbol_specs={"XAUUSD": broken_spec},
        )
        d = gate.evaluate(proposal, risk_state, account)
        assert d.action is RiskAction.REJECTED
        assert "max_risk_per_trade" in d.failed_checks


class TestDeterminism:
    def test_repeated_evaluation_identical(self, gate, proposal, risk_state, account):
        first = gate.evaluate(proposal, risk_state, account)
        for _ in range(3):
            again = gate.evaluate(proposal, risk_state, account)
            assert again.model_dump() == first.model_dump()

    def test_rebuilt_gate_identical(self, proposal, risk_state, account):
        first = make_gate().evaluate(proposal, risk_state, account)
        rebuilt = make_gate().evaluate(proposal, risk_state, account)
        assert rebuilt.model_dump() == first.model_dump()

    def test_gate_decision_id_is_derived_not_generated(self, gate, proposal, risk_state, account):
        d1 = gate.evaluate(proposal, risk_state, account)
        d2 = gate.evaluate(proposal, risk_state, account)
        assert d1.gate_decision_id == d2.gate_decision_id
        assert len(d1.gate_decision_id) == 16  # sha256[:16], deterministic

    def test_different_failures_different_ids(self, proposal, risk_state, account):
        d_ok = make_gate().evaluate(proposal, risk_state, account)
        d_risk = make_gate().evaluate(
            make_proposal(suggested_volume=0.5), risk_state, account
        )
        assert d_ok.gate_decision_id != d_risk.gate_decision_id

    def test_config_snapshot_recorded(self, gate, proposal, risk_state, account):
        d = gate.evaluate(proposal, risk_state, account)
        assert d.config_snapshot["trading_enabled"] is True
        assert d.config_snapshot["max_total_exposure"] == 1_000_000.0

    def test_halt_states_recorded(self, proposal, risk_state):
        d = make_gate().evaluate(proposal, risk_state, make_account(kill_switch_active=True))
        assert d.kill_switch_active is True and d.emergency_stop_active is False
        d = make_gate().evaluate(proposal, risk_state, make_account(emergency_stop_active=True))
        assert d.emergency_stop_active is True and d.kill_switch_active is False


class TestEvents:
    def _gate_with_sink(self, **config):
        events: list[RiskEvent] = []
        gate = HardRiskGate(
            RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0, **config),
            symbol_specs={"XAUUSD": gold_symbol_spec()},
            event_sink=events.append,
        )
        return gate, events

    def test_approved_event(self, proposal, risk_state, account):
        gate, events = self._gate_with_sink()
        gate.evaluate(proposal, risk_state, account)
        assert len(events) == 1
        assert events[0].type is RiskEventType.RISK_APPROVED
        assert events[0].proposal_id == proposal.decision_id

    def test_rejected_event(self, proposal, account):
        gate, events = self._gate_with_sink()
        gate.evaluate(proposal, RiskState(equity=10_000.0, daily_loss=999.0), account)
        assert events[0].type is RiskEventType.RISK_REJECTED

    def test_emergency_stop_event(self, proposal, risk_state, account):
        gate, events = self._gate_with_sink()
        gate.evaluate(proposal, risk_state, make_account(emergency_stop_active=True))
        assert events[0].type is RiskEventType.EMERGENCY_STOP

    def test_kill_switch_event(self, proposal, risk_state, account):
        gate, events = self._gate_with_sink()
        gate.evaluate(proposal, risk_state, make_account(kill_switch_active=True))
        assert events[0].type is RiskEventType.KILL_SWITCH_ACTIVE

    def test_no_sink_is_fine(self, proposal, risk_state, account):
        gate = make_gate()
        d = gate.evaluate(proposal, risk_state, account)  # must not raise
        assert d.action is RiskAction.APPROVED


class TestFromAppConfig:
    def test_maps_app_fields(self):
        from app.core.config import AppConfig

        app = AppConfig(
            trading_enabled=True, risk_per_trade_pct=0.25,
            max_daily_loss_pct=1.5, max_open_positions=2,
            max_pending_orders=3, max_spread_points=40.0,
        )
        config = RiskGateConfig.from_app_config(app)
        assert config.trading_enabled is True
        assert config.max_risk_per_trade_pct == 0.25
        assert config.daily_loss_limit_pct == 1.5
        assert config.max_open_positions == 2
        assert config.max_pending_orders == 3
        assert config.max_spread_points == 40.0

    def test_exposure_limit_never_defaulted(self):
        from app.core.config import AppConfig

        config = RiskGateConfig.from_app_config(AppConfig())
        assert config.max_total_exposure is None  # must be set explicitly

    def test_safe_defaults_from_bare_app_config(self):
        from app.core.config import AppConfig

        config = RiskGateConfig.from_app_config(AppConfig())
        assert config.trading_enabled is False  # fail closed
        assert config.dry_run is True
