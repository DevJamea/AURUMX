"""Configuration tests — especially the safety defaults (spec §54)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import REQUIRED_REAL_TRADING_PHRASE, AppConfig
from app.core.enums import TimeFrame, TradingMode
from app.core.exceptions import UnsafeConfigurationError


class TestSafeDefaults:
    def test_fresh_install_is_read_only(self):
        config = AppConfig(_env_file=None)
        assert config.trading_enabled is False
        assert config.dry_run is True
        assert config.trading_mode is TradingMode.READ_ONLY

    def test_default_symbol_is_auto(self):
        assert AppConfig(_env_file=None).symbol == "AUTO"

    def test_default_timeframes(self):
        assert AppConfig(_env_file=None).timeframes == [
            TimeFrame.M15,
            TimeFrame.H1,
            TimeFrame.H4,
        ]

    def test_conservative_risk_defaults(self):
        config = AppConfig(_env_file=None)
        assert config.risk_per_trade_pct == 0.5
        assert config.max_open_positions == 1
        assert config.max_pending_orders == 2
        assert config.enable_trailing is False
        assert config.enable_break_even is True
        assert config.enable_partial_close is True


class TestEnvironmentParsing:
    def test_flat_env_names_from_spec(self, monkeypatch):
        monkeypatch.setenv("TRADING_ENABLED", "true")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("RISK_PER_TRADE", "1.25")
        monkeypatch.setenv("MAX_OPEN_POSITIONS", "3")
        monkeypatch.setenv("TIMEFRAMES", "M5,M15,H1,H4")
        monkeypatch.setenv("SYMBOL", "gold")

        config = AppConfig(_env_file=None)
        assert config.trading_enabled is True
        assert config.dry_run is True
        assert config.trading_mode is TradingMode.DRY_RUN
        assert config.risk_per_trade_pct == 1.25
        assert config.max_open_positions == 3
        assert config.timeframes == [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]
        assert config.symbol == "GOLD"  # normalized

    def test_empty_env_values_mean_unset(self, monkeypatch):
        monkeypatch.setenv("MT5_LOGIN", "")
        monkeypatch.setenv("MT5_PASSWORD", "")
        monkeypatch.setenv("MT5_SERVER", "")
        monkeypatch.setenv("MAX_SPREAD_POINTS", "")

        config = AppConfig(_env_file=None)
        assert config.mt5_login is None
        assert config.mt5_password is None
        assert config.mt5_server is None
        assert config.max_spread_points is None

    def test_mt5_login_without_password_is_valid_config(self, monkeypatch):
        # connection-time rejects login-without-password; config itself allows it
        monkeypatch.setenv("MT5_LOGIN", "12345")
        config = AppConfig(_env_file=None)
        assert config.mt5_login == 12345

    def test_invalid_timeframe_rejected(self, monkeypatch):
        monkeypatch.setenv("TIMEFRAMES", "M15,ZZZ")
        with pytest.raises(ValidationError):
            AppConfig(_env_file=None)

    def test_invalid_risk_rejected(self, monkeypatch):
        monkeypatch.setenv("RISK_PER_TRADE", "50")
        with pytest.raises(ValidationError):
            AppConfig(_env_file=None)


class TestRealTradingGuard:
    def test_live_trading_without_confirmation_is_refused(self):
        with pytest.raises(UnsafeConfigurationError):
            AppConfig(_env_file=None, trading_enabled=True, dry_run=False)

    def test_live_trading_with_confirmation_is_allowed(self):
        config = AppConfig(
            _env_file=None,
            trading_enabled=True,
            dry_run=False,
            real_trading_confirmed=REQUIRED_REAL_TRADING_PHRASE,
        )
        assert config.trading_mode is TradingMode.MT5_REAL

    def test_mutation_cannot_bypass_the_guard(self):
        config = AppConfig(_env_file=None, trading_enabled=True, dry_run=True)
        with pytest.raises(UnsafeConfigurationError):
            config.dry_run = False

    def test_dry_run_with_trading_enabled_is_fine(self):
        config = AppConfig(_env_file=None, trading_enabled=True, dry_run=True)
        assert config.trading_mode is TradingMode.DRY_RUN


class TestSecretHygiene:
    def test_password_never_appears_in_repr_or_summary(self):
        config = AppConfig(_env_file=None, mt5_password="sup3r-s3cret-value")
        assert "sup3r-s3cret-value" not in repr(config)
        assert "sup3r-s3cret-value" not in repr(config.model_dump())
        summary = str(config.safe_summary())
        assert "sup3r-s3cret-value" not in summary
        assert config.safe_summary()["mt5_password"] == "***set***"

    def test_safe_summary_contains_core_fields(self):
        summary = AppConfig(_env_file=None).safe_summary()
        assert summary["trading_mode"] == "READ_ONLY"
        assert summary["symbol"] == "AUTO"
