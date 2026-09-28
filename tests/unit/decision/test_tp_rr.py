"""TP/RR hardening tests (hardening §1).

Proves the ``minimum_rr`` gate is GENUINELY meaningful for structure-derived
take-profits (TP_BY_STRUCTURE derives the target independently of the risk
distance, so the actual RR is a market outcome, not a configured constant),
and documents the self-referential nature of the gate for TP_BY_RR.
"""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, DecisionAction
from app.decision import DecisionEngine, DecisionEngineConfig
from app.decision.config import TPMethod
from app.decision.levels import compute_levels
from app.risk import RiskState
from tests.unit.agents.scenarios import (
    REF_TIME,
    gold_symbol_spec,
    linear_trend_closes,
    make_snapshot,
    standard_triple,
)
from tests.unit.decision.conftest import FixedAgent

STATE = RiskState(equity=10_000.0)
SYMBOL = gold_symbol_spec()


def buy_snapshot():
    up = linear_trend_closes(300, slope=2.0, seed=11)
    return make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)


def stub_engine(swing_high: float | None, **config) -> DecisionEngine:
    """Engine with stubbed agents but the REAL trend-up snapshot, so regime,
    ATR, alignment and entry are genuine — only the structure level is
    controlled, which is exactly what the RR gate must react to."""
    agents = [
        FixedAgent("trend", AgentDirection.BUY, 0.9),
        FixedAgent(
            "structure", AgentDirection.BUY, 0.9,
            features={"last_swing_high": {"price": swing_high}} if swing_high else {},
        ),
        FixedAgent("liquidity", AgentDirection.BUY, 0.5),
    ]
    return DecisionEngine(DecisionEngineConfig(**config), agents=agents)


class TestTPMethodConfig:
    def test_both_methods_exist_with_spec_names(self):
        assert TPMethod.TP_BY_RR.value == "rr"
        assert TPMethod.TP_BY_STRUCTURE.value == "structure"

    def test_default_is_rr_and_was_not_silently_changed(self):
        assert DecisionEngineConfig().tp_method is TPMethod.TP_BY_RR

    def test_string_config_still_accepted(self):
        assert DecisionEngineConfig(tp_method="structure").tp_method is TPMethod.TP_BY_STRUCTURE
        assert DecisionEngineConfig(tp_method="rr").tp_method is TPMethod.TP_BY_RR

    def test_unknown_method_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DecisionEngineConfig(tp_method="tea_leaves")


class TestTPByRR:
    def test_rr_target_geometry(self):
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=None, opposing_structure_level=2662.0,
            symbol=SYMBOL, config=DecisionEngineConfig(),
        )
        assert plan.tp_source == "rr_target"  # opposing level ignored for rr
        assert plan.reward_risk == pytest.approx(2.0, abs=0.01)

    def test_rr_gate_note_is_carried_on_proposal(self):
        """Every TP_BY_RR proposal carries the explicit note that its RR gate
        is self-referential — it validates geometry, not market quality."""
        d = stub_engine(swing_high=2662.0).evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        assert any("self-referential" in r for r in d.proposal.reasons)

    def test_rr_gate_still_enforces_against_config_contradiction(self):
        """minimum_rr above target_rr HOLDs even for TP_BY_RR (cross-config
        sanity + tick-snapping drift), so the gate is not fully dead code."""
        engine = stub_engine(swing_high=2662.0, minimum_rr=2.5, target_rr=2.0)
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["insufficient_rr"]


class TestTPByStructureRRIsGenuine:
    """The hardening centerpiece: structure-derived TP -> real RR -> gate."""

    def test_valid_structure_target_trades_with_market_rr(self):
        engine = stub_engine(swing_high=2662.0, tp_method="structure")
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        p = d.proposal
        assert p.tp_source == "structure_level"
        assert p.take_profit == pytest.approx(2662.0)
        # RR is a market outcome (2.505), NOT the configured target_rr (2.0)
        assert p.risk_reward == pytest.approx(2.505, abs=0.01)
        assert p.risk_reward != pytest.approx(2.0, abs=0.01)

    def test_structure_target_below_minimum_rr_holds(self):
        """Swing high only 1.2 above entry vs a ~4.7 ATR stop -> RR ~0.25.
        The gate HOLDs: this is a REAL rejection of a poor payoff, not the
        target checking itself."""
        engine = stub_engine(swing_high=2651.2, tp_method="structure")
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["insufficient_rr"]
        assert d.proposal is None
        assert any("below minimum" in r for r in d.reasons)

    def test_rr_threshold_is_the_binding_constraint(self):
        """The same far structure target passes at minimum_rr 1.5 and fails
        at 3.0 — the gate genuinely grades the market's offer."""
        far = 2662.0  # RR ~2.5
        ok = stub_engine(swing_high=far, tp_method="structure", minimum_rr=1.5).evaluate(
            buy_snapshot(), risk_state=STATE, now=REF_TIME,
        )
        blocked = stub_engine(swing_high=far, tp_method="structure", minimum_rr=3.0).evaluate(
            buy_snapshot(), risk_state=STATE, now=REF_TIME,
        )
        assert ok.decision is DecisionAction.BUY
        assert ok.proposal.risk_reward == pytest.approx(2.505, abs=0.01)
        assert blocked.decision is DecisionAction.HOLD
        assert blocked.rejection_reasons == ["insufficient_rr"]

    def test_rr_varies_with_the_market_not_the_config(self):
        """Two different structure levels -> two different RRs under the
        same config (impossible under TP_BY_RR, where RR ≈ target always)."""
        near = stub_engine(swing_high=2660.0, tp_method="structure").evaluate(
            buy_snapshot(), risk_state=STATE, now=REF_TIME,
        )
        far = stub_engine(swing_high=2670.0, tp_method="structure").evaluate(
            buy_snapshot(), risk_state=STATE, now=REF_TIME,
        )
        assert near.decision is DecisionAction.BUY and far.decision is DecisionAction.BUY
        assert near.proposal.risk_reward != far.proposal.risk_reward
        assert far.proposal.risk_reward > near.proposal.risk_reward


class TestInvalidStructureLevels:
    def test_wrong_side_level_holds_invalid_geometry(self):
        """A swing high BELOW the BUY entry is not a target — used verbatim,
        flagged, rejected by the geometry gate. Never silently clamped."""
        engine = stub_engine(swing_high=2640.0, tp_method="structure")
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons == ["invalid_geometry"]
        assert any("TP must be above entry" in r for r in d.reasons)

    def test_wrong_side_level_is_never_clamped(self):
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=None, opposing_structure_level=2640.0,
            symbol=SYMBOL, config=DecisionEngineConfig(tp_method="structure"),
        )
        assert plan.take_profit == pytest.approx(2640.0)  # verbatim, not entry+min
        assert any("INVALID structure target" in n for n in plan.notes)

    def test_too_close_level_fails_rr_or_geometry(self):
        """0.05 above entry (< broker minimum 0.20): the honest level is
        unusable; the engine must HOLD (whichever gate catches it first)."""
        engine = stub_engine(swing_high=2650.25, tp_method="structure")
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.HOLD
        assert d.rejection_reasons[0] in ("insufficient_rr", "invalid_geometry")

    def test_missing_level_documented_fallback(self):
        """TP_BY_STRUCTURE with no opposing level: documented fallback to the
        RR target (never a silent substitution of methodology)."""
        engine = stub_engine(swing_high=None, tp_method="structure")
        d = engine.evaluate(buy_snapshot(), risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.BUY
        assert d.proposal.tp_source == "rr_target"
        assert any("falls back to RR target" in r for r in d.proposal.reasons)


class TestRRMethodSellMirror:
    def test_sell_structure_tp_uses_last_swing_low(self):
        from app.core.enums import TimeFrame
        from tests.unit.agents.scenarios import make_series

        down = linear_trend_closes(300, slope=-2.0, seed=12)
        series = {
            TimeFrame.M15: make_series(down, TimeFrame.M15, seed=13),
            TimeFrame.H1: make_series(down, TimeFrame.H1, seed=12),
            TimeFrame.H4: make_series(down, TimeFrame.H4, seed=12),
        }
        snap = make_snapshot(series, created_at=REF_TIME)
        agents = [
            FixedAgent("trend", AgentDirection.SELL, 0.9),
            FixedAgent("structure", AgentDirection.SELL, 0.9,
                       features={"last_swing_low": {"price": 2638.0}}),
            FixedAgent("liquidity", AgentDirection.SELL, 0.5),
        ]
        engine = DecisionEngine(DecisionEngineConfig(tp_method="structure"), agents=agents)
        d = engine.evaluate(snap, risk_state=STATE, now=REF_TIME)
        assert d.decision is DecisionAction.SELL
        assert d.proposal.tp_source == "structure_level"
        assert d.proposal.take_profit == pytest.approx(2638.0)
        assert d.proposal.take_profit < d.proposal.entry_price
