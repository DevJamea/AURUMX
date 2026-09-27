"""Position-sizing calculator tests (Phase-3 §14).

Pinned formula: ``loss_per_lot = risk_distance / tick_size * tick_value`` and
``raw_volume = risk_amount / loss_per_lot``, floored to the broker step
(never rounded up — the risk budget is a hard cap).
"""

from __future__ import annotations

import pytest

from app.risk.sizing import calculate_position_size
from tests.unit.agents.scenarios import gold_symbol_spec

SYMBOL = gold_symbol_spec()  # tick .01 / value 1.0 / step .01 / min .01 / max 100


class TestSizingMath:
    def test_known_values(self):
        sizing = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2645.0, symbol=SYMBOL,
        )
        # risk $50; loss/lot = 5.0/0.01*1.0 = $500 -> raw 0.10 lots
        assert sizing.risk_amount == pytest.approx(50.0)
        assert sizing.loss_per_lot == pytest.approx(500.0)
        assert sizing.raw_volume == pytest.approx(0.10)
        assert sizing.suggested_volume == pytest.approx(0.10)
        assert sizing.monetary_risk == pytest.approx(50.0)
        assert sizing.percentage_risk == pytest.approx(0.5)
        assert sizing.feasible

    def test_direction_agnostic(self):
        long_side = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2645.0, symbol=SYMBOL,
        )
        short_side = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2655.0, symbol=SYMBOL,
        )
        assert long_side.suggested_volume == short_side.suggested_volume

    def test_floor_to_step_never_rounds_up(self):
        # raw 0.1066... -> floors to 0.10, monetary risk stays <= budget
        sizing = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2647.0, symbol=SYMBOL,  # distance 3.0
        )
        assert sizing.raw_volume == pytest.approx(50.0 / 300.0)
        assert sizing.suggested_volume == pytest.approx(0.16)  # floor(0.1666/.01)*.01
        assert sizing.monetary_risk <= sizing.risk_amount + 1e-9

    def test_wider_stop_reduces_volume(self):
        tight = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2648.0, symbol=SYMBOL,
        )
        wide = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2640.0, symbol=SYMBOL,
        )
        assert wide.suggested_volume < tight.suggested_volume


class TestLimits:
    def test_below_minimum_volume_is_infeasible_not_forced(self):
        # tiny equity: raw volume 0.0005 -> below volume_min 0.01
        sizing = calculate_position_size(
            equity=50.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2645.0, symbol=SYMBOL,
        )
        assert not sizing.feasible
        assert sizing.infeasible_reason == "volume_below_minimum"
        assert sizing.suggested_volume == 0.0  # never force the minimum

    def test_clamped_to_maximum(self):
        # huge equity: raw 5.0 lots -> clamped to volume_max 100? -> not here;
        # make raw exceed max: equity 100_000_000, risk 10%
        sizing = calculate_position_size(
            equity=100_000_000.0, risk_per_trade_pct=10.0,
            entry=2650.0, stop_loss=2649.0, symbol=SYMBOL,
        )
        assert sizing.clamped_to_max
        assert sizing.suggested_volume == SYMBOL.volume_max
        assert sizing.feasible

    def test_zero_risk_distance_is_infeasible(self):
        sizing = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2650.0, symbol=SYMBOL,
        )
        assert not sizing.feasible
        assert sizing.infeasible_reason == "zero_risk_distance"

    def test_invalid_symbol_metadata_is_infeasible(self):
        broken = SYMBOL.model_copy(update={"tick_size": 0.0})
        sizing = calculate_position_size(
            equity=10_000.0, risk_per_trade_pct=0.5,
            entry=2650.0, stop_loss=2645.0, symbol=broken,
        )
        assert not sizing.feasible
        assert sizing.infeasible_reason == "invalid_symbol_metadata"


class TestNoMartingaleByConstruction:
    def test_no_loss_history_is_an_input(self):
        """The calculator's signature has no loss/streak/history parameter —
        martingale, loss-recovery multipliers, DCA and averaging down are
        structurally impossible, not merely discouraged."""
        import inspect

        params = set(inspect.signature(calculate_position_size).parameters)
        # the complete, closed API surface: nothing else can influence size
        assert params == {"equity", "risk_per_trade_pct", "entry", "stop_loss", "symbol"}

    def test_risk_state_is_not_consumed_by_sizing(self):
        """RiskState (which carries loss info) is an engine input, never a
        sizing input."""
        import inspect

        assert "risk_state" not in inspect.signature(calculate_position_size).parameters

    def test_same_inputs_same_output(self):
        kwargs = dict(equity=10_000.0, risk_per_trade_pct=0.5,
                      entry=2650.0, stop_loss=2645.0, symbol=SYMBOL)
        assert calculate_position_size(**kwargs) == calculate_position_size(**kwargs)

    def test_conservative_default(self):
        """0.5% default risk on $10k with a 5.0 stop risks $50 — the default
        is deliberately conservative (spec §54)."""
        from app.decision import DecisionEngineConfig

        assert DecisionEngineConfig().risk_per_trade_pct == 0.5
