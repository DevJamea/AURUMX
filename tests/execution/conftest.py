"""Shared fixtures for the Phase-5 execution tests.

Builds the full evidence chain the execution layer requires:
proposal -> HardRiskGate APPROVED decision -> ExecutionRequest, plus a
FakeMT5 with execution explicitly enabled (the read-only default stays
untouched — see tests/unit/brokers for that safety net).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.enums import TradingMode
from app.execution import (
    ExecutionRequest,
    ExecutionService,
    ExecutionServiceConfig,
    InMemoryExecutionJournal,
)
from app.risk import HardRiskGate, RiskGateConfig, RiskState
from tests.conftest import REF_TIME
from tests.fakes.mt5_fake import FakeMT5, default_symbol
from tests.unit.agents.scenarios import gold_symbol_spec
from tests.unit.risk.conftest import make_account, make_proposal

#: deterministic clock for every execution test
EXEC_REF_TIME = datetime(2026, 1, 5, 12, 30, tzinfo=UTC)


def make_exec_fake(**kwargs) -> FakeMT5:
    """A fake terminal with execution explicitly opted in (DEMO account)."""
    kwargs.setdefault("execution_enabled", True)
    fake = FakeMT5(symbols=[default_symbol("XAUUSD")], **kwargs)
    fake.serve_market("XAUUSD", now=REF_TIME)
    return fake


def make_exec_broker(fake: FakeMT5) -> MT5Broker:
    broker = MT5Broker(mt5_module=fake)
    broker.connect()
    return broker


def make_approved_decision(proposal, *, gate: HardRiskGate | None = None):
    """Run the proposal through the REAL gate and return its decision
    (the execution layer must never see a fabricated approval)."""
    gate = gate or HardRiskGate(
        RiskGateConfig(trading_enabled=True, max_total_exposure=1_000_000.0),
        symbol_specs={"XAUUSD": gold_symbol_spec()},
    )
    return gate.evaluate(proposal, RiskState(equity=10_000.0), make_account())


def make_request(proposal, decision) -> ExecutionRequest:
    return ExecutionRequest.from_proposal(
        proposal, risk_decision_id=decision.gate_decision_id
    )


def make_service(broker, *, mode: TradingMode = TradingMode.DRY_RUN, **overrides):
    """DRY_RUN by default; MT5_DEMO needs trading_enabled + dry_run=False."""
    config = dict(trading_enabled=True, dry_run=True)
    if mode is TradingMode.MT5_DEMO:
        config = dict(trading_enabled=True, dry_run=False)
    elif mode is TradingMode.READ_ONLY:
        config = dict(trading_enabled=False, dry_run=True)
    config.update(overrides)
    return ExecutionService(
        broker,
        config=ExecutionServiceConfig(**config),
        journal=InMemoryExecutionJournal(),
        clock=lambda: EXEC_REF_TIME,
    )


@pytest.fixture()
def exec_fake() -> FakeMT5:
    return make_exec_fake()


@pytest.fixture()
def exec_broker(exec_fake: FakeMT5) -> MT5Broker:
    return make_exec_broker(exec_fake)


@pytest.fixture()
def approved_pair():
    """(proposal, APPROVED RiskDecision) from the real gate."""
    proposal = make_proposal()
    decision = make_approved_decision(proposal)
    assert decision.approved, decision.reasons
    return proposal, decision


@pytest.fixture()
def dry_run_service(exec_broker) -> ExecutionService:
    return make_service(exec_broker, mode=TradingMode.DRY_RUN)


@pytest.fixture()
def demo_service(exec_broker) -> ExecutionService:
    return make_service(exec_broker, mode=TradingMode.MT5_DEMO)
