"""Fixtures for the control-plane tests: a runtime over the FakeMT5."""

from __future__ import annotations

import pytest

from app.control import EngineRuntime
from app.core.config import AppConfig
from tests.conftest import REF_TIME as CONTROL_REF


def make_runtime_config(**overrides) -> AppConfig:
    base = dict(
        _env_file=None,
        trading_enabled=True,
        dry_run=True,
        max_total_exposure_usd=1_000_000.0,
    )
    base.update(overrides)
    return AppConfig(**base)


@pytest.fixture()
def control_config() -> AppConfig:
    return make_runtime_config()


@pytest.fixture()
def runtime(fake_mt5, control_config):
    """A connected runtime in DRY_RUN mode with a live market scenario."""
    from app.brokers.mt5 import MT5Broker

    broker = MT5Broker(mt5_module=fake_mt5)
    runtime = EngineRuntime(
        control_config,
        broker,
        clock=lambda: CONTROL_REF,
    )
    runtime.connect()
    return runtime
