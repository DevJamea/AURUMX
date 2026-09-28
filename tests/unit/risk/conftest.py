"""Shared fixtures for the Phase-4 RiskGate tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.enums import AgentDirection, TimeFrame
from app.decision.proposal import TradeProposal
from app.risk import AccountState, HardRiskGate, RiskGateConfig, RiskState
from app.risk.sizing import RiskSizing
from tests.unit.agents.scenarios import gold_symbol_spec

REF = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def make_proposal(**overrides) -> TradeProposal:
    """A clean, valid BUY proposal (risk $40 = 0.08 lots x $500/lot)."""
    sizing = RiskSizing(
        equity=10_000.0, risk_per_trade_pct=0.5, risk_amount=40.0,
        risk_distance=5.0, loss_per_lot=500.0, raw_volume=0.08,
        normalized_volume=0.08, suggested_volume=0.08,
        monetary_risk=40.0, percentage_risk=0.4, feasible=True,
    )
    base = dict(
        symbol="XAUUSD",
        direction=AgentDirection.BUY,
        entry_price=2650.2,
        stop_loss=2645.2,
        take_profit=2660.2,
        risk_distance=5.0,
        reward_distance=10.0,
        risk_reward=2.0,
        suggested_volume=0.08,
        max_allowed_volume=100.0,
        timeframe=TimeFrame.H1,
        regime="TREND_UP",
        decision_id="dec0000111",
        setup_type="trend",
        fingerprint="fp0000000000001",
        reasons=["trend up"],
        invalidation_conditions=["structure invalidated"],
        created_at=REF,
        expires_at=REF + timedelta(minutes=15),
        sl_source="atr",
        tp_source="rr_target",
        sizing=sizing,
    )
    base.update(overrides)
    return TradeProposal(**base)


def make_gate(**config) -> HardRiskGate:
    """A gate with trading enabled, exposure configured, spec registered."""
    defaults = dict(trading_enabled=True, max_total_exposure=1_000_000.0)
    defaults.update(config)
    return HardRiskGate(
        RiskGateConfig(**defaults), symbol_specs={"XAUUSD": gold_symbol_spec()}
    )


def make_account(**overrides) -> AccountState:
    base = dict(
        equity=10_000.0,
        balance=10_000.0,
        margin=100.0,
        margin_free=9_900.0,
        margin_level=10_000.0,
        trade_allowed=True,
        open_positions_notional=0.0,
        spread_points=20.0,
    )
    base.update(overrides)
    return AccountState(**base)


@pytest.fixture()
def proposal() -> TradeProposal:
    return make_proposal()


@pytest.fixture()
def gate() -> HardRiskGate:
    return make_gate()


@pytest.fixture()
def account() -> AccountState:
    return make_account()


@pytest.fixture()
def risk_state() -> RiskState:
    return RiskState(equity=10_000.0)
