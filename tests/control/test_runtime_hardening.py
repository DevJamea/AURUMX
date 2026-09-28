"""Runtime hardening: single-use approvals, journaled expiry, serialization."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest

from app.brokers.mt5 import MT5Broker
from app.control import EngineRuntime
from app.control.errors import EngineNotStartedError, ExecutionRefused
from app.core.enums import TradingMode
from app.execution import ExecutionStatus
from tests.control.conftest import CONTROL_REF, make_runtime_config
from tests.control.test_runtime import inject_trend


def _demo_runtime(fake_mt5, clock=None) -> EngineRuntime:
    fake_mt5.execution_enabled = True
    config = make_runtime_config(
        dry_run=False, real_trading_confirmed="I ACCEPT REAL TRADING RISK"
    )
    runtime = EngineRuntime(
        config, MT5Broker(mt5_module=fake_mt5), clock=clock or (lambda: CONTROL_REF)
    )
    runtime.connect()
    inject_trend(fake_mt5)
    runtime.control.start()
    return runtime


class TestSingleUseApproval:
    def test_approval_cleared_after_successful_execution(self, fake_mt5):
        runtime = _demo_runtime(fake_mt5)
        runtime.evaluate_cycle()
        result = runtime.execute_approved()
        assert result.status is ExecutionStatus.FILLED
        assert runtime._last_proposal is None and runtime._last_risk_decision is None
        with pytest.raises(ExecutionRefused, match="no proposal"):
            runtime.execute_approved()
        assert len(fake_mt5.order_sends) == 1

    def test_approval_consumed_even_when_refused_because_engine_stopped(self, fake_mt5):
        runtime = _demo_runtime(fake_mt5)
        runtime.evaluate_cycle()
        runtime.control.stop()
        with pytest.raises(EngineNotStartedError):
            runtime.execute_approved()
        runtime.control.start()
        with pytest.raises(ExecutionRefused, match="no proposal"):
            runtime.execute_approved()
        assert fake_mt5.order_sends == []


class TestExpiredProposal:
    def test_expired_proposal_is_refused_journaled_and_never_sent(self, fake_mt5):
        now = [CONTROL_REF]
        runtime = _demo_runtime(fake_mt5, clock=lambda: now[0])
        runtime.evaluate_cycle()
        assert runtime._last_proposal is not None
        now[0] = CONTROL_REF + timedelta(days=1)

        with pytest.raises(ExecutionRefused, match="proposal_expired"):
            runtime.execute_approved()

        assert fake_mt5.order_sends == [] and fake_mt5.order_checks == []
        record = runtime.execution_journal.recent(1)[0]
        assert record.status is ExecutionStatus.NOT_ATTEMPTED
        assert "proposal_expired" in record.reasons
        assert record.mode in (TradingMode.MT5_DEMO, TradingMode.DRY_RUN)
        # and the approval is gone
        with pytest.raises(ExecutionRefused, match="no proposal"):
            runtime.execute_approved()


class TestConcurrency:
    def test_concurrent_execute_approved_sends_at_most_one_order(self, fake_mt5):
        runtime = _demo_runtime(fake_mt5)
        runtime.evaluate_cycle()

        barrier = threading.Barrier(4)
        results: list[object] = []
        lock = threading.Lock()

        def attempt() -> None:
            barrier.wait()
            try:
                outcome: object = runtime.execute_approved()
            except ExecutionRefused as exc:
                outcome = exc
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        executed = [r for r in results if not isinstance(r, Exception)]
        refused = [r for r in results if isinstance(r, ExecutionRefused)]
        assert len(results) == 4
        assert len(executed) == 1 and len(refused) == 3
        assert len(fake_mt5.order_sends) == 1

    def test_evaluate_cycle_waits_while_an_execution_is_in_flight(self, fake_mt5):
        """The runtime's critical section is real: a concurrent control
        request must not interleave with an execution in progress."""
        runtime = _demo_runtime(fake_mt5)
        runtime.evaluate_cycle()

        entered, release = threading.Event(), threading.Event()
        original = runtime.execution_service.execute

        def slow_execute(request, decision):
            entered.set()
            assert release.wait(timeout=5)
            return original(request, decision)

        runtime.execution_service.execute = slow_execute  # type: ignore[method-assign]

        executor = threading.Thread(target=runtime.execute_approved)
        executor.start()
        assert entered.wait(timeout=5)

        evaluator = threading.Thread(target=runtime.evaluate_cycle)
        evaluator.start()
        evaluator.join(timeout=0.3)
        assert evaluator.is_alive(), "evaluate_cycle ran concurrently with an execution"

        release.set()
        executor.join(timeout=5)
        evaluator.join(timeout=5)
        assert not executor.is_alive() and not evaluator.is_alive()
        assert len(fake_mt5.order_sends) == 1
