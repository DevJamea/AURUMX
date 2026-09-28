"""REAL MT5 demo execution — explicit opt-in only (spec §18/§36).

Run (Windows, MT5 terminal running, DEMO account logged in):

    set AURUMX_LIVE_MT5=1
    set TRADING_ENABLED=true
    set DRY_RUN=false
    set REAL_TRADING_CONFIRMED=I ACCEPT REAL TRADING RISK
    set MAX_TOTAL_EXPOSURE_USD=1000000
    pytest tests/live_mt5 -m live_mt5 -s

Safety rules baked into this harness (§18):

* exactly ONE evaluation cycle;
* at most ONE order — only for a proposal the REAL HardRiskGate APPROVED;
* the order is checked (order_check) before it is sent (order_send);
* the retcode is verified and the resulting MT5 state is confirmed;
* the journal record and the reconciliation report are printed for review;
* real accounts are refused before any order is attempted.

If the market offers no setup, the test reports SKIP (no order is forced —
a fallback synthetic-but-gate-checked proposal can be enabled with
AURUMX_LIVE_ALLOW_SYNTHETIC=1, see EXECUTION.md).
"""

from __future__ import annotations

import os

import pytest

from app.core.config import AppConfig
from app.core.enums import RiskAction

pytestmark = [pytest.mark.live_mt5, pytest.mark.integration]


@pytest.fixture(scope="module")
def live_config() -> AppConfig:
    config = AppConfig.load(env_file=None)
    if not (config.trading_enabled and not config.dry_run):
        pytest.skip(
            "real demo execution requires TRADING_ENABLED=true, DRY_RUN=false "
            "(+ REAL_TRADING_CONFIRMED phrase) in the environment"
        )
    if config.max_total_exposure_usd is None:
        pytest.skip("real demo execution requires MAX_TOTAL_EXPOSURE_USD to be set")
    return config


@pytest.fixture(scope="module")
def live_runtime(live_config):
    pytest.importorskip("MetaTrader5", reason="MetaTrader5 package (Windows only)")
    from app.brokers.mt5 import MT5Broker
    from app.control import EngineRuntime

    broker = MT5Broker.from_config(live_config)
    runtime = EngineRuntime(live_config, broker)
    runtime.connect()
    status = runtime.status()
    print(f"\n[AURUMX LIVE] account_type={status['account_type']} symbol={status['symbol']}")
    if status["account_type"] != "DEMO":
        pytest.fail(
            f"account is {status['account_type']} — AURUMX refuses non-DEMO execution"
        )
    runtime.control.start()
    yield runtime
    broker.disconnect()


def test_one_demo_order_verified_and_reconciled(live_runtime):
    runtime = live_runtime

    # ---- ONE evaluation cycle ------------------------------------------
    outcome = runtime.evaluate_cycle()
    decision, risk = outcome["decision"], outcome["risk"]
    print(f"[AURUMX LIVE] decision={decision['decision']} risk={risk.get('action')}")

    proposal = runtime._last_proposal
    gate_decision = runtime._last_risk_decision

    if gate_decision is None or not gate_decision.approved:
        if os.environ.get("AURUMX_LIVE_ALLOW_SYNTHETIC") == "1" and proposal is None:
            _use_synthetic_proposal(runtime)
        else:
            pytest.skip(
                "no gate-APPROVED proposal this cycle — no order was sent "
                "(the market decides; re-run during an active session)"
            )

    # ---- ONE order (only reachable with a real APPROVED decision) -------
    result = runtime.execute_approved()
    print(
        f"[AURUMX LIVE] status={result.status.value} retcode={result.retcode} "
        f"({result.retcode_description}) order={result.order_ticket} "
        f"deal={result.deal_ticket} position={result.position_ticket} "
        f"fill={result.fill_price} volume={result.filled_volume}"
    )

    assert result.status.value in ("FILLED", "PARTIALLY_FILLED", "UNKNOWN"), (
        f"execution failed: {result.message}"
    )
    if result.status.value == "UNKNOWN":
        pytest.fail(
            "order sent but state could not be verified — MANUAL REVIEW REQUIRED "
            f"(check the terminal): {result.message}"
        )

    # ---- VERIFY against actual MT5 state (§19) ---------------------------
    positions = runtime.broker.get_positions(result.symbol)
    ours = [p for p in positions if p.magic == result.order_ticket or p.ticket == result.position_ticket]
    assert ours, "no matching position found in MT5 after a reported fill"
    position = ours[0]
    print(
        f"[AURUMX LIVE] verified position #{position.ticket} "
        f"{position.direction.value} {position.volume} @ {position.price_open} "
        f"sl={position.price_sl} tp={position.price_tp} magic={position.magic}"
    )
    assert position.magic != 0, "position does not carry the AURUMX magic number"

    # ---- RECONCILE (§22) -------------------------------------------------
    report = runtime.reconcile()
    print(f"[AURUMX LIVE] reconciliation: clean={report.clean} counts={report.counts}")
    assert report.clean, f"reconciliation mismatch: {report.summary()}"

    # ---- JOURNAL ACCURACY (§20) ------------------------------------------
    record = runtime.execution_journal.recent(1)[0]
    assert record.claims_real_fill
    assert record.order_ticket == result.order_ticket
    assert record.deal_ticket == result.deal_ticket
    assert record.position_ticket == result.position_ticket

    print("[AURUMX LIVE] DEMO EXECUTION VERIFIED — one order, checked, sent, "
          "state-confirmed and reconciled.")
    print("[AURUMX LIVE] NOTE: this does NOT authorize real-money trading.")


def _use_synthetic_proposal(runtime) -> None:
    """Fallback (explicitly opted in): build a MINIMAL valid proposal from
    the live tick and pass it through the REAL gate — the gate still
    independently approves/rejects; nothing is bypassed."""
    from datetime import UTC, datetime, timedelta

    from app.core.enums import AgentDirection, MarketRegime, TimeFrame
    from app.decision.proposal import TradeProposal
    from app.risk import RiskSizing

    symbol = runtime._symbol
    spec = runtime.broker.get_symbol(symbol)
    tick = runtime.broker.get_tick(symbol)
    now = datetime.now(UTC)
    entry = tick.ask
    distance = max(
        (spec.stops_level_points + 5) * spec.point,  # broker minimum + buffer
        spec.point * 50,
    )
    risk_amount = 20.0
    loss_per_lot = distance / spec.tick_size * spec.tick_value
    volume = max(spec.volume_min, min(0.01, risk_amount / loss_per_lot))
    volume = spec.normalize_volume(volume)
    proposal = TradeProposal(
        decision_id=f"live-demo-{int(now.timestamp())}",
        fingerprint=f"live-demo-{entry:.2f}",
        setup_type="demo_verification",
        symbol=symbol,
        direction=AgentDirection.BUY,
        entry_price=round(entry, spec.digits),
        stop_loss=round(entry - distance, spec.digits),
        take_profit=round(entry + 2 * distance, spec.digits),
        risk_reward=2.0,
        risk_distance=distance,
        reward_distance=2 * distance,
        suggested_volume=volume,
        max_allowed_volume=spec.volume_max,
        sizing=RiskSizing(
            equity=runtime._read_account().equity,
            risk_per_trade_pct=0.1,
            risk_amount=volume * loss_per_lot,
            risk_distance=distance,
            loss_per_lot=loss_per_lot,
            raw_volume=volume,
            normalized_volume=volume,
            suggested_volume=volume,
            monetary_risk=volume * loss_per_lot,
            percentage_risk=0.1,
            feasible=True,
        ),
        sl_source="demo_verification",
        tp_source="demo_verification",
        timeframe=TimeFrame.M15,
        regime=MarketRegime.UNKNOWN,
        reasons=["live demo verification harness"],
        invalidation_conditions=[],
        created_at=now,
        expires_at=now + timedelta(minutes=5),
    )
    risk_state, account_state = runtime._risk_inputs()
    decision = runtime.gate.evaluate(proposal, risk_state, account_state)
    if decision.action is not RiskAction.APPROVED:
        pytest.skip(f"synthetic demo proposal was gate-REJECTED (correctly): {decision.reasons}")
    runtime._last_proposal = proposal
    runtime._last_risk_decision = decision
    print(
        f"[AURUMX LIVE] synthetic proposal gate-APPROVED "
        f"(risk ${decision.risk_amount:.2f}) — proceeding with ONE order"
    )
