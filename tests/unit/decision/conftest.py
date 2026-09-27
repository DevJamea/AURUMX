"""Shared fixtures for Phase-3 decision-engine tests."""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.base import AgentResult, BaseAgent
from app.core.enums import AgentDirection, DataQuality, TimeFrame
from app.decision import DecisionEngine, DecisionEngineConfig
from app.risk import RiskState
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_series,
    make_snapshot,
    sideways_closes,
    standard_triple,
)


class FixedAgent(BaseAgent):
    """Deterministic stub agent — exact control over direction/strength for
    gate-level tests (the engine accepts any agent roster)."""

    def __init__(
        self,
        name: str,
        direction: AgentDirection,
        strength: float,
        *,
        features: dict[str, Any] | None = None,
        quality: DataQuality = DataQuality.OK,
    ) -> None:
        super().__init__()
        self._name = name
        self._direction = direction
        self._strength = strength
        self._features = features or {}
        self._quality = quality

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    def analyze(self, context) -> AgentResult:
        return self.make_result(
            context,
            direction=self._direction,
            signal_strength=self._strength,
            reasons=[f"fixed {self._direction.value} {self._strength}"],
            features=dict(self._features),
            data_quality=self._quality,
        )


@pytest.fixture()
def engine() -> DecisionEngine:
    return DecisionEngine(DecisionEngineConfig())


@pytest.fixture()
def risk_state() -> RiskState:
    return RiskState(equity=10_000.0)


@pytest.fixture()
def buy_snapshot():
    """Scenario A: H4/H1/M15 all bullish, normal spread, open market."""
    up = linear_trend_closes(300, slope=2.0, seed=11)
    return make_snapshot(standard_triple(h1_closes=up, h1_seed=11), created_at=REF_TIME)


@pytest.fixture()
def sell_snapshot():
    """Scenario B: everything bearish."""
    down = linear_trend_closes(300, slope=-2.0, seed=12)
    return make_snapshot(standard_triple(h1_closes=down, h1_seed=12), created_at=REF_TIME)


@pytest.fixture()
def range_snapshot():
    rng = sideways_closes(300, period=16, amplitude=1.2, seed=71)
    return make_snapshot(standard_triple(h1_closes=rng, h1_seed=71), created_at=REF_TIME)


@pytest.fixture()
def conflict_snapshot():
    """Scenario C: H4/H1 bullish, M15 strongly bearish."""
    up = linear_trend_closes(300, slope=2.0, seed=11)
    down_m15 = linear_trend_closes(300, slope=-0.8, seed=55)
    series = {
        TimeFrame.M15: make_series(down_m15, TimeFrame.M15, seed=12),
        TimeFrame.H1: make_series(up, TimeFrame.H1, seed=11),
        TimeFrame.H4: make_series(up, TimeFrame.H4, seed=11),
    }
    return make_snapshot(series, created_at=REF_TIME)
