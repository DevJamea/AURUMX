"""EngineControl tests: safe defaults, halts, reset semantics, audit trail."""

from __future__ import annotations

from app.control import EngineControl
from app.core.events import EventBus
from app.risk import HaltStatus, InMemoryKillSwitchStore


class TestSafeDefaults:
    def test_fresh_control_is_stopped(self):
        control = EngineControl()
        assert control.started is False

    def test_fresh_control_has_no_halts(self):
        control = EngineControl()
        assert control.halt_flags() == {
            "kill_switch_active": False, "emergency_stop_active": False,
        }

    def test_halt_states_are_independent(self):
        control = EngineControl()
        control.engage_emergency_stop("test")
        flags = control.halt_flags()
        assert flags["emergency_stop_active"] is True
        assert flags["kill_switch_active"] is False  # distinct halts (§5)

        control2 = EngineControl()
        control2.engage_kill_switch("test")
        flags = control2.halt_flags()
        assert flags["kill_switch_active"] is True
        assert flags["emergency_stop_active"] is False


class TestHalts:
    def test_emergency_stop_engages(self):
        control = EngineControl()
        control.engage_emergency_stop("manual")
        stop = control.emergency_stop()
        assert stop.status is HaltStatus.ACTIVE
        assert stop.reason == "manual"
        assert stop.engaged

    def test_reset_clears_both_halts(self):
        control = EngineControl()
        control.engage_emergency_stop()
        control.engage_kill_switch()
        result = control.reset_kill_switch()
        assert result["kill_switch_active"] is False
        assert result["emergency_stop_active"] is False

    def test_reset_is_the_only_way_out(self):
        """Engaged halts persist across control-plane restarts (the store
        is injected — persistence is the store's job, per Phase-4 §24)."""
        store = InMemoryKillSwitchStore()
        control = EngineControl(emergency_stop_store=store)
        control.engage_emergency_stop()
        # "restart": a brand-new control over the same persisted state
        control2 = EngineControl(emergency_stop_store=store)
        assert control2.halt_flags()["emergency_stop_active"] is True
        control2.reset_kill_switch()
        assert control2.halt_flags()["emergency_stop_active"] is False


class TestAuditTrail:
    def test_every_operation_emits(self):
        bus = EventBus()
        control = EngineControl(event_bus=bus)
        control.start()
        control.engage_emergency_stop("why")
        control.reset_kill_switch()
        control.stop()
        types = [e.type for e in bus.history()]
        assert types == [
            "CONTROL_STARTED",
            "CONTROL_EMERGENCY_STOP",
            "CONTROL_KILL_SWITCH_RESET",
            "CONTROL_STOPPED",
        ]

    def test_summary_shape(self):
        control = EngineControl()
        summary = control.summary()
        assert summary["started"] is False
        assert "kill_switch" in summary and "emergency_stop" in summary
