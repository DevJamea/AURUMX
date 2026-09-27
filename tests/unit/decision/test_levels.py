"""SL/TP level tests (Phase-3 §11/§12/§13): hierarchy, geometry, validation."""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection
from app.decision import DecisionEngineConfig
from app.decision.levels import (
    compute_levels,
    min_stop_distance,
    validate_levels,
)
from tests.unit.agents.scenarios import gold_symbol_spec

SYMBOL = gold_symbol_spec()  # tick .01, digits 2, stops 20 pts, freeze 10 pts
CONFIG = DecisionEngineConfig()  # atr_stop_multiple=2, target_rr=2, min via engine


def test_min_stop_distance_uses_stops_and_freeze():
    assert min_stop_distance(SYMBOL) == pytest.approx(0.20)  # 20 points * 0.01


class TestStopLossHierarchy:
    def test_structure_level_used_when_valid_and_tighter(self):
        # entry 2650, structure HL at 2648.5 (1.5 below) vs ATR stop 2x1.0=2.0:
        # the structure level is closer -> structure wins (with buffer note)
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=2648.5, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.sl_source == "structure"
        assert plan.stop_loss == pytest.approx(2648.5, abs=0.01)
        assert plan.sl_structure_level == 2648.5

    def test_atr_stop_used_when_structure_is_farther(self):
        # structure HL at 2640 (10 below) vs ATR stop 2.0 -> ATR (tighter)
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=2640.0, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.sl_source == "atr"
        assert plan.stop_loss == pytest.approx(2648.0, abs=0.01)

    def test_structure_too_close_falls_back_to_atr(self):
        # HL at 2649.9 (0.1 below entry) < broker minimum 0.20 -> ATR stop
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=2649.9, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.sl_source == "atr"
        assert any("too close" in n for n in plan.notes)

    def test_no_structure_no_atr_uses_broker_minimum(self):
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=None,
            structure_invalidation=None, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.sl_source == "broker_minimum"
        assert plan.stop_loss == pytest.approx(2649.80, abs=0.001)

    def test_atr_below_broker_minimum_is_raised(self):
        # ATR 0.05 -> 2x ATR = 0.10 < 0.20 minimum -> broker minimum wins
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=0.05,
            structure_invalidation=None, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert abs(2650.0 - plan.stop_loss) >= 0.20 - 1e-9

    def test_sell_stop_is_above_entry(self):
        plan = compute_levels(
            direction=AgentDirection.SELL, entry=2650.0, atr=1.0,
            structure_invalidation=2660.0, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.stop_loss == pytest.approx(2652.0, abs=0.01)
        assert plan.stop_loss > 2650.0


class TestTakeProfit:
    def test_rr_target_geometry_buy(self):
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=None, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.tp_source == "rr_target"
        assert plan.take_profit > 2650.0
        assert plan.stop_loss < 2650.0
        assert plan.reward_risk == pytest.approx(2.0, abs=0.01)

    def test_rr_target_geometry_sell(self):
        plan = compute_levels(
            direction=AgentDirection.SELL, entry=2650.0, atr=1.0,
            structure_invalidation=None, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        assert plan.take_profit < 2650.0
        assert plan.stop_loss > 2650.0
        assert plan.reward_risk == pytest.approx(2.0, abs=0.01)

    def test_structure_target_uses_opposing_level(self):
        config = DecisionEngineConfig(tp_method="structure")
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=None, opposing_structure_level=2680.0,
            symbol=SYMBOL, config=config,
        )
        assert plan.tp_source == "structure_level"
        assert plan.take_profit == pytest.approx(2680.0)

    def test_structure_target_rr_varies(self):
        config = DecisionEngineConfig(tp_method="structure")
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=5.0,
            structure_invalidation=None, opposing_structure_level=2655.0,
            symbol=SYMBOL, config=config,
        )
        # risk 10 (2x ATR), reward 5 -> RR 0.5 (engine gate must HOLD this)
        assert plan.reward_risk == pytest.approx(0.5, abs=0.05)

    def test_structure_target_falls_back_when_level_missing(self):
        config = DecisionEngineConfig(tp_method="structure")
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.0, atr=1.0,
            structure_invalidation=None, opposing_structure_level=None,
            symbol=SYMBOL, config=config,
        )
        assert plan.tp_source == "rr_target"  # documented fallback

    def test_levels_align_to_tick_grid(self):
        plan = compute_levels(
            direction=AgentDirection.BUY, entry=2650.03, atr=1.37,
            structure_invalidation=None, opposing_structure_level=None,
            symbol=SYMBOL, config=CONFIG,
        )
        for price in (plan.stop_loss, plan.take_profit):
            assert abs(price / 0.01 - round(price / 0.01)) < 1e-9


class TestValidation:
    def test_valid_buy_geometry_has_no_issues(self):
        assert validate_levels(
            direction=AgentDirection.BUY, entry=2650.0,
            stop_loss=2648.0, take_profit=2654.0, symbol=SYMBOL,
        ) == []

    def test_wrong_direction_sl_is_rejected(self):
        issues = validate_levels(
            direction=AgentDirection.BUY, entry=2650.0,
            stop_loss=2652.0, take_profit=2654.0, symbol=SYMBOL,
        )
        assert any("SL must be below entry" in i for i in issues)

    def test_wrong_direction_tp_is_rejected(self):
        issues = validate_levels(
            direction=AgentDirection.BUY, entry=2650.0,
            stop_loss=2648.0, take_profit=2649.0, symbol=SYMBOL,
        )
        assert any("TP must be above entry" in i for i in issues)

    def test_sell_geometry_mirrored(self):
        assert validate_levels(
            direction=AgentDirection.SELL, entry=2650.0,
            stop_loss=2652.0, take_profit=2646.0, symbol=SYMBOL,
        ) == []
        issues = validate_levels(
            direction=AgentDirection.SELL, entry=2650.0,
            stop_loss=2648.0, take_profit=2646.0, symbol=SYMBOL,
        )
        assert any("SL must be above entry" in i for i in issues)

    def test_below_broker_minimum_distance_rejected(self):
        issues = validate_levels(
            direction=AgentDirection.BUY, entry=2650.0,
            stop_loss=2649.9, take_profit=2654.0, symbol=SYMBOL,
        )
        assert any("below broker minimum" in i for i in issues)

    def test_off_grid_price_rejected(self):
        issues = validate_levels(
            direction=AgentDirection.BUY, entry=2650.0,
            stop_loss=2648.005, take_profit=2654.0, symbol=SYMBOL,
        )
        assert any("not aligned to tick size" in i for i in issues)
