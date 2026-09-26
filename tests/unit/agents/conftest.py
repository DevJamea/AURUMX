"""Shared fixtures for the agent test-suite.

All scenarios come from ``scenarios.py`` (seeded, clock-anchored, deterministic)
— never from wall-clock time or randomness inside a test.
"""

from __future__ import annotations

import pytest

from app.agents import default_agents
from app.agents.base import BaseAgent
from app.core.enums import MarketRegime, VolatilityLevel
from app.decision.regime import RegimeAssessment
from tests.unit.agents.scenarios import REF_TIME


@pytest.fixture()
def agents() -> dict[str, BaseAgent]:
    return {agent.name: agent for agent in default_agents()}


def _assessment(regime: MarketRegime, volatility: VolatilityLevel) -> RegimeAssessment:
    return RegimeAssessment(
        regime=regime,
        volatility=volatility,
        as_of=REF_TIME,
        data_quality="OK",
        evidence=[f"test fixture: {regime.value}"],
        conflicts=[],
    )


@pytest.fixture()
def range_regime() -> RegimeAssessment:
    return _assessment(MarketRegime.RANGE, VolatilityLevel.NORMAL)


@pytest.fixture()
def trend_up_regime() -> RegimeAssessment:
    return _assessment(MarketRegime.TREND_UP, VolatilityLevel.NORMAL)


@pytest.fixture()
def trend_down_regime() -> RegimeAssessment:
    return _assessment(MarketRegime.TREND_DOWN, VolatilityLevel.NORMAL)


@pytest.fixture()
def high_vol_regime() -> RegimeAssessment:
    return _assessment(MarketRegime.HIGH_VOLATILITY, VolatilityLevel.HIGH)
