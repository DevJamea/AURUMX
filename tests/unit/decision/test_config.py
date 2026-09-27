"""DecisionEngineConfig tests (Phase-3 §4): thresholds are configurable,
named, validated — no scattered magic numbers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import AppConfig
from app.decision import DecisionEngineConfig
from app.decision.config import TimeframeWeights


class TestSpecKeyNames:
    """The spec's decision.* keys must exist under their exact names."""

    def test_threshold_keys_exist(self):
        config = DecisionEngineConfig()
        for key in (
            "buy_threshold", "sell_threshold", "minimum_signal_strength",
            "minimum_timeframe_alignment", "max_conflict", "minimum_rr",
            "target_rr", "atr_stop_multiple", "max_spread_points",
            "risk_per_trade_pct", "max_open_positions", "max_pending_orders",
            "max_consecutive_losses", "daily_loss_limit_pct",
            "proposal_ttl_minutes", "tp_method",
        ):
            assert hasattr(config, key), f"missing decision config key: {key}"

    def test_documented_defaults(self):
        """Initial engineering parameters (NOT statistically optimal — see
        docs/DECISIONS.md; they require validation on historical data)."""
        config = DecisionEngineConfig()
        assert config.buy_threshold == pytest.approx(1.00)
        assert config.sell_threshold == pytest.approx(1.00)
        assert config.net_threshold == pytest.approx(0.80)
        assert config.minimum_signal_strength == pytest.approx(0.45)
        assert config.minimum_timeframe_alignment == pytest.approx(0.80)
        assert config.max_conflict == pytest.approx(0.35)
        assert config.target_rr == pytest.approx(2.0)
        assert config.atr_stop_multiple == pytest.approx(2.0)
        assert config.risk_per_trade_pct == pytest.approx(0.5)

    def test_thresholds_are_validated(self):
        with pytest.raises(ValidationError):
            DecisionEngineConfig(buy_threshold=0.0)
        with pytest.raises(ValidationError):
            DecisionEngineConfig(minimum_signal_strength=1.5)
        with pytest.raises(ValidationError):
            DecisionEngineConfig(risk_per_trade_pct=-1.0)

    def test_tp_method_is_limited(self):
        DecisionEngineConfig(tp_method="rr")
        DecisionEngineConfig(tp_method="structure")
        with pytest.raises(ValidationError):
            DecisionEngineConfig(tp_method="tea_leaves")

    def test_asymmetric_thresholds_allowed(self):
        config = DecisionEngineConfig(buy_threshold=1.5, sell_threshold=1.0)
        assert config.buy_threshold != config.sell_threshold


class TestTimeframeWeights:
    def test_default_weights(self):
        w = TimeframeWeights()
        assert w.h4 == pytest.approx(0.30)
        assert w.h1 == pytest.approx(0.45)
        assert w.m15 == pytest.approx(0.25)
        assert w.total == pytest.approx(1.0)

    def test_weights_need_not_sum_to_one(self):
        """Scores renormalize over the timeframes actually present, so the
        weights are free-form positives."""
        w = TimeframeWeights(h4=0.5, h1=0.5, m15=0.5)
        assert w.total == pytest.approx(1.5)


class TestTimeframePolicy:
    """Hardening §2: H4 context is required unless explicitly relaxed."""

    def test_strict_defaults(self):
        from app.decision.config import TimeframePolicy

        policy = TimeframePolicy()
        assert policy.require_h4 is True
        assert policy.allow_missing_h4_renormalization is False

    def test_renormalization_is_explicit_opt_in(self):
        from app.decision.config import TimeframePolicy

        policy = TimeframePolicy(allow_missing_h4_renormalization=True)
        assert policy.require_h4 is True  # H4 still "required" as policy;
        assert policy.allow_missing_h4_renormalization is True  # renorm is the escape hatch

    def test_no_extra_fields(self):
        import pytest as _pytest
        from pydantic import ValidationError

        from app.decision.config import TimeframePolicy

        with _pytest.raises(ValidationError):
            TimeframePolicy(sneaky_field=1)

    def test_policy_reachable_as_decision_timeframe(self):
        config = DecisionEngineConfig()
        assert config.timeframe.require_h4 is True
        assert config.timeframe.allow_missing_h4_renormalization is False

    def test_policy_is_journaled_in_config_snapshot(self):
        dumped = DecisionEngineConfig().model_dump(mode="json")
        assert dumped["timeframe"] == {
            "require_h4": True, "allow_missing_h4_renormalization": False,
        }


class TestFromAppConfig:
    def test_risk_settings_flow_from_app_config(self):
        app = AppConfig(
            max_spread_points=40.0,
            risk_per_trade_pct=0.25,
            max_open_positions=2,
            max_pending_orders=3,
            max_daily_loss_pct=1.5,
            min_reward_risk=1.8,
        )
        config = DecisionEngineConfig.from_app_config(app)
        assert config.max_spread_points == 40.0
        assert config.risk_per_trade_pct == 0.25
        assert config.max_open_positions == 2
        assert config.max_pending_orders == 3
        assert config.daily_loss_limit_pct == 1.5
        assert config.minimum_rr == 1.8

    def test_consecutive_losses_falls_back(self):
        config = DecisionEngineConfig.from_app_config(AppConfig())
        assert config.max_consecutive_losses == 3

    def test_engine_config_snapshot_is_json_safe(self):
        dumped = DecisionEngineConfig().model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert dumped["buy_threshold"] == 1.00
