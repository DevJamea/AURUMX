"""Diagnostics script tests (read-only, used by the future launcher)."""

from __future__ import annotations

from pathlib import Path

from app.brokers.mt5 import MT5Broker
from app.core.config import AppConfig
from scripts.diagnostics import FAIL, PASS, WARN, run_diagnostics
from tests.conftest import REF_TIME, make_fake_mt5


class TestDiagnostics:
    def test_healthy_setup_all_pass(self, fake_mt5, tmp_path: Path):
        config = AppConfig(_env_file=None, data_dir=tmp_path / "data")
        broker = MT5Broker(mt5_module=fake_mt5)

        report = run_diagnostics(config=config, broker=broker, clock=lambda: REF_TIME)

        statuses = {c.name: c.status for c in report.checks}
        assert statuses["python"] == PASS
        assert statuses["configuration"] == PASS
        assert statuses["mt5_connection"] == PASS
        assert statuses["account"] == PASS
        assert statuses["gold_symbol"] == PASS
        assert statuses["market_data"] == PASS
        assert statuses["candles_M15"] == PASS
        assert statuses["candles_H1"] == PASS
        assert statuses["candles_H4"] == PASS
        assert statuses["data_directory"] == PASS
        assert not report.has_failures
        assert report.as_dict()["summary"] == PASS

    def test_no_gold_symbol_fails(self, tmp_path: Path):
        from tests.fakes.mt5_fake import FakeMT5, default_symbol

        fake = FakeMT5(symbols=[default_symbol("EURUSD", point=0.00001, digits=5)])
        config = AppConfig(_env_file=None, data_dir=tmp_path / "data")
        broker = MT5Broker(mt5_module=fake)

        report = run_diagnostics(config=config, broker=broker, clock=lambda: REF_TIME)

        statuses = {c.name: c.status for c in report.checks}
        assert statuses["gold_symbol"] == FAIL
        assert report.has_failures

    def test_stale_market_warns_when_closed(self, tmp_path: Path):
        from datetime import timedelta

        fake = make_fake_mt5()
        config = AppConfig(_env_file=None, data_dir=tmp_path / "data")
        broker = MT5Broker(mt5_module=fake)
        late_clock = lambda: REF_TIME + timedelta(days=2)  # noqa: E731

        report = run_diagnostics(config=config, broker=broker, clock=late_clock)

        statuses = {c.name: c.status for c in report.checks}
        assert statuses["market_data"] == WARN  # session CLOSED, not a system failure
        assert not report.has_failures

    def test_trading_enabled_is_flagged_as_warning(self, fake_mt5, tmp_path: Path):
        config = AppConfig(
            _env_file=None, data_dir=tmp_path / "data", trading_enabled=True, dry_run=True
        )
        broker = MT5Broker(mt5_module=fake_mt5)

        report = run_diagnostics(config=config, broker=broker, clock=lambda: REF_TIME)

        statuses = {c.name: c.status for c in report.checks}
        assert statuses["configuration"] == WARN
        assert not report.has_failures
