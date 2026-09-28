"""End-to-end execution pipeline (Phase 5 §1/§54):

FakeMT5 -> MarketDataService -> DecisionEngine -> TradeProposal ->
HardRiskGate(APPROVED) -> ExecutionService -> MT5 adapter (check -> send)
-> verification -> FILLED -> Reconciliation MATCHED.

Two variants: DRY_RUN (provably zero order_send) and MT5_DEMO (one checked,
sent, verified order).  Everything runs offline against the deterministic
fake terminal.
"""

from __future__ import annotations

import pytest

from app.brokers.mt5 import MT5Broker
from app.control import EngineRuntime
from app.core.config import AppConfig
from app.core.enums import TimeFrame, TradingMode
from app.core.events import EventBus
from tests.conftest import REF_TIME
from tests.integration.test_agent_pipeline import inject_scenario
from tests.unit.agents.scenarios import linear_trend_closes


def make_config(**overrides) -> AppConfig:
    base = dict(
        _env_file=None,
        trading_enabled=True,
        dry_run=True,
        max_total_exposure_usd=1_000_000.0,
    )
    base.update(overrides)
    return AppConfig(**base)


def make_runtime(fake_mt5, config) -> EngineRuntime:
    broker = MT5Broker(mt5_module=fake_mt5)
    runtime = EngineRuntime(
        config,
        broker,
        clock=lambda: REF_TIME,
        event_bus=EventBus(),
    )
    runtime.connect()
    return runtime


def serve_uptrend(fake_mt5) -> None:
    up = linear_trend_closes(300, slope=2.0, seed=11)
    inject_scenario(fake_mt5, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up})
    fake_mt5.set_tick(
        "XAUUSD", epoch=REF_TIME.timestamp() - 2, bid=2650.0, ask=2650.20
    )


class TestDryRunPipeline:
    def test_full_dry_run_cycle(self, fake_mt5):
        serve_uptrend(fake_mt5)
        runtime = make_runtime(fake_mt5, make_config(dry_run=True))
        runtime.control.start()

        outcome = runtime.evaluate_cycle()
        assert outcome["decision"]["decision"] == "BUY"
        assert outcome["risk"]["action"] == "APPROVED"

        result = runtime.execute_approved()
        assert result.status.value == "DRY_RUN"
        assert result.mode is TradingMode.DRY_RUN

        # THE DRY_RUN INVARIANT (§14): order_send/order_check never called
        assert fake_mt5.order_sends == []
        assert fake_mt5.order_checks == []

        # journal honesty: dry run, no fabricated tickets
        record = runtime.execution_journal.recent(1)[0]
        assert record.dry_run is True
        assert record.order_ticket is None
        assert record.deal_ticket is None
        assert record.position_ticket is None

        # approved proposals are intentionally single-use and cleared after
        # the execution attempt; the journal retains the execution evidence.
        assert runtime._last_proposal is None
        assert runtime._last_risk_decision is None

        # reconciliation: dry-run records are skipped -> clean empty report
        report = runtime.reconcile()
        assert report.clean
        assert report.skipped and "dry-run" in report.skipped[0]


class TestDemoPipeline:
    def test_full_demo_cycle_checked_sent_verified_reconciled(self, fake_mt5):
        fake_mt5.execution_enabled = True
        config = make_config(
            dry_run=False, real_trading_confirmed="I ACCEPT REAL TRADING RISK"
        )
        runtime = make_runtime(fake_mt5, config)
        serve_uptrend(fake_mt5)
        runtime.control.start()

        outcome = runtime.evaluate_cycle()
        assert outcome["risk"]["action"] == "APPROVED"

        result = runtime.execute_approved()
        assert result.status.value == "FILLED"
        assert result.stage.value == "VERIFIED"

        # §7A.3/§7A.4: exactly one check, then exactly one send
        assert len(fake_mt5.order_checks) == 1
        assert len(fake_mt5.order_sends) == 1
        assert fake_mt5.order_checks[0] == fake_mt5.order_sends[0]

        # the order carried the AURUMX identity
        sent = fake_mt5.order_sends[0]
        assert sent["magic"] == 0x41555258
        assert sent["comment"].startswith("AURUMX ")

        # §19: the MT5 state actually contains the position
        positions = fake_mt5.positions_get("XAUUSD")
        assert len(positions) == 1
        assert positions[0].ticket == result.position_ticket
        assert positions[0].magic == 0x41555258

        # §22: reconciliation matches
        report = runtime.reconcile()
        assert report.clean
        assert report.counts == {"MATCHED": 1}

        # the correlation chain is complete end to end
        record = runtime.execution_journal.recent(1)[0]
        assert record.proposal_id == outcome["decision"]["decision_id"]
        assert record.risk_decision_id == outcome["risk"]["gate_decision_id"]
        assert record.order_ticket == result.order_ticket
        assert record.position_ticket == result.position_ticket

    def test_demo_blocked_for_real_accounts(self, fake_mt5):
        from tests.fakes.mt5_fake import ACCOUNT_TRADE_MODE_REAL, default_account

        fake_mt5.execution_enabled = True
        fake_mt5.account = default_account(trade_mode=ACCOUNT_TRADE_MODE_REAL)
        config = make_config(
            dry_run=False, real_trading_confirmed="I ACCEPT REAL TRADING RISK"
        )
        runtime = make_runtime(fake_mt5, config)
        serve_uptrend(fake_mt5)
        runtime.control.start()
        runtime.evaluate_cycle()

        result = runtime.execute_approved()
        assert result.status.value == "NOT_ATTEMPTED"
        assert "real_account_blocked" in result.message
        assert fake_mt5.order_sends == []
        assert fake_mt5.order_checks == []


class TestSafeDefaults:
    def test_fresh_config_is_read_only_and_dry(self):
        config = AppConfig(_env_file=None)
        assert config.trading_enabled is False
        assert config.dry_run is True

    def test_read_only_runtime_never_reaches_the_broker(self, fake_mt5):
        fake_mt5.execution_enabled = True
        runtime = make_runtime(fake_mt5, make_config(trading_enabled=False))
        serve_uptrend(fake_mt5)
        runtime.control.start()
        outcome = runtime.evaluate_cycle()
        # gate rejects with trading disabled -> nothing executable
        risk = outcome["risk"]
        if risk.get("risk_decision") is not None:
            assert risk["action"] == "REJECTED"
        from app.control.errors import ExecutionRefused

        with pytest.raises(ExecutionRefused):
            runtime.execute_approved()
        assert fake_mt5.order_sends == []

    def test_live_config_requires_the_phrase(self):
        from app.core.exceptions import UnsafeConfigurationError

        with pytest.raises(UnsafeConfigurationError):
            make_config(dry_run=False)  # no REAL_TRADING_CONFIRMED
