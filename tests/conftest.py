"""Shared fixtures.  All market tests run against a deterministic FakeMT5."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.config import AppConfig
from tests.fakes.mt5_fake import (
    TIMEFRAME_H1,
    TIMEFRAME_H4,
    TIMEFRAME_M15,
    FakeMT5,
    default_symbol,
)

#: A Monday 12:00 UTC — gold market open, everything deterministic.
REF_TIME = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)

GOLD_TIMEFRAMES = [TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4]


def make_fake_mt5(*, now: datetime = REF_TIME, with_market: bool = True) -> FakeMT5:
    """A broker serving several gold spellings plus decoys (silver, FX)."""
    fake = FakeMT5(
        symbols=[
            default_symbol("XAUUSD"),
            default_symbol("XAUUSDm"),
            default_symbol("GOLD"),
            default_symbol("XAUUSD.bad", point=0.0),  # gold name, broken metadata
            default_symbol("XAGUSD", point=0.001, digits=3),  # silver — never gold
            default_symbol("XAUEUR", point=0.01, digits=2),
            default_symbol("EURUSD", point=0.00001, digits=5),
        ]
    )
    if with_market:
        fake.serve_market("XAUUSD", now=now, timeframes=GOLD_TIMEFRAMES)
    return fake


@pytest.fixture()
def fixed_now() -> datetime:
    return REF_TIME


@pytest.fixture()
def fake_mt5() -> FakeMT5:
    return make_fake_mt5()


@pytest.fixture()
def broker(fake_mt5: FakeMT5) -> MT5Broker:
    """A connected read-only MT5Broker backed by the fake terminal."""
    broker = MT5Broker(mt5_module=fake_mt5)
    broker.connect()
    return broker


@pytest.fixture()
def config() -> AppConfig:
    """Safe default config (read-only mode), isolated from any .env file."""
    return AppConfig(_env_file=None)
