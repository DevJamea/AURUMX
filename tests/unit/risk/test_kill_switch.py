"""Kill switch + emergency stop state tests (Phase 4 §4/§5/§24)."""

from __future__ import annotations

import json

from app.core.enums import RiskAction
from app.risk import (
    AccountState,
    EmergencyStopState,
    HaltStatus,
    InMemoryKillSwitchStore,
    KillSwitchState,
    halt_flags,
)


class TestHaltStatus:
    def test_all_representable_states(self):
        assert [s.value for s in HaltStatus] == [
            "STANDBY", "ACTIVE", "TRIGGERED", "RESET_REQUIRED",
        ]

    def test_standby_is_not_engaged(self):
        state = KillSwitchState()
        assert state.status is HaltStatus.STANDBY
        assert state.engaged is False
        assert state.requires_reset is False
        assert state.gate_action() is None

    def test_active_engages(self):
        state = KillSwitchState(status=HaltStatus.ACTIVE, reason="operator halt")
        assert state.engaged
        assert state.gate_action() is RiskAction.EMERGENCY_STOP

    def test_triggered_requires_reset(self):
        state = EmergencyStopState(status=HaltStatus.TRIGGERED, source="auto:circuit_breaker")
        assert state.engaged and state.requires_reset

    def test_reset_required_latches(self):
        state = KillSwitchState(status=HaltStatus.RESET_REQUIRED)
        assert state.engaged and state.requires_reset
        assert state.gate_action() is RiskAction.EMERGENCY_STOP


class TestPersistenceInterface:
    def test_store_roundtrip(self):
        store = InMemoryKillSwitchStore()
        assert store.load().status is HaltStatus.STANDBY  # fresh default
        store.save(KillSwitchState(status=HaltStatus.ACTIVE, reason="halt"))
        loaded = store.load()
        assert loaded.status is HaltStatus.ACTIVE
        assert loaded.reason == "halt"

    def test_emergency_stop_store_roundtrip(self):
        store = InMemoryKillSwitchStore()
        store.save(EmergencyStopState(status=HaltStatus.TRIGGERED))
        assert store.load().requires_reset

    def test_state_is_json_safe(self):
        blob = json.dumps(KillSwitchState(status=HaltStatus.ACTIVE).model_dump(mode="json"))
        assert "ACTIVE" in blob


class TestBridgeToGateInputs:
    """§24: the documented mapping from (future) control-layer state to the
    AccountState flags the gate consumes."""

    def test_halt_flags_standby(self):
        flags = halt_flags(KillSwitchState(), EmergencyStopState())
        assert flags == {"kill_switch_active": False, "emergency_stop_active": False}

    def test_halt_flags_engaged(self):
        flags = halt_flags(
            KillSwitchState(status=HaltStatus.ACTIVE),
            EmergencyStopState(status=HaltStatus.TRIGGERED),
        )
        assert flags == {"kill_switch_active": True, "emergency_stop_active": True}

    def test_flags_flow_into_account_state(self):
        account = AccountState(**halt_flags(
            kill_switch=KillSwitchState(status=HaltStatus.ACTIVE),
        ))
        assert account.kill_switch_active is True
        assert account.emergency_stop_active is False

    def test_kill_switch_and_emergency_stop_are_independent(self):
        """§5: two distinct halts — one can be engaged without the other."""
        flags = halt_flags(kill_switch=KillSwitchState(status=HaltStatus.ACTIVE))
        assert flags["kill_switch_active"] and not flags["emergency_stop_active"]
        flags = halt_flags(emergency_stop=EmergencyStopState(status=HaltStatus.ACTIVE))
        assert flags["emergency_stop_active"] and not flags["kill_switch_active"]


class TestNoAutomaticEmergencies:
    def test_gate_never_invents_halt_conditions(self):
        """§5: only caller-supplied flags produce EMERGENCY_STOP — the state
        models default to STANDBY and nothing auto-engages them."""
        assert KillSwitchState().engaged is False
        assert EmergencyStopState().engaged is False
