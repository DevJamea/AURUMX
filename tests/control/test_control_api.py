"""LocalControlAPI tests (5E §29/§33): routes, safe defaults, control ops,
and the hard guarantees — no execution endpoint, GUI served, localhost."""

from __future__ import annotations

import pytest

from app.control.api import LocalControlAPI
from app.core.enums import TradingMode


@pytest.fixture()
def api(runtime) -> LocalControlAPI:
    with LocalControlAPI(runtime, host="127.0.0.1", port=0) as api:
        yield api


@pytest.fixture()
def client():
    import httpx

    return httpx.Client(timeout=5.0)


class TestReadRoutes:
    def test_status_shape(self, api, client):
        response = client.get(f"{api.url}/status")
        assert response.status_code == 200
        data = response.json()
        for key in (
            "system_state", "started", "mt5_connected", "account_type",
            "symbol", "mode", "trading_enabled", "dry_run",
            "kill_switch_active", "emergency_stop_active", "reconciliation",
            "bid", "ask", "spread",
        ):
            assert key in data, key

    def test_account_type_comes_from_the_broker(self, api, client):
        """§31: never infer DEMO from configuration — the engine verifies."""
        data = client.get(f"{api.url}/status").json()
        assert data["account_type"] == "DEMO"  # the FAKE terminal's actual account

    def test_market_route(self, api, client):
        data = client.get(f"{api.url}/market").json()
        assert data["tick"]["bid"] > 0
        assert data["trading_data_ok"] is True

    def test_empty_views_before_any_cycle(self, api, client):
        assert client.get(f"{api.url}/agents").json() == {"agents": []}
        assert client.get(f"{api.url}/decision").json() == {"decision": None}
        assert client.get(f"{api.url}/risk").json() == {"risk_decision": None}
        assert client.get(f"{api.url}/execution").json() == {"executions": []}

    def test_positions_route(self, api, client):
        data = client.get(f"{api.url}/positions").json()
        assert "positions" in data

    def test_reconciliation_route_before_any_run(self, api, client):
        data = client.get(f"{api.url}/reconciliation").json()
        assert data["status"] == "NOT_RUN"
        assert data["execution_allowed"] is True

    def test_events_route(self, api, client):
        data = client.get(f"{api.url}/events").json()
        assert isinstance(data["events"], list)

    def test_unknown_route_404(self, api, client):
        assert client.get(f"{api.url}/nope").status_code == 404

    def test_gui_is_served(self, api, client):
        response = client.get(f"{api.url}/")
        assert response.status_code == 200
        assert "AURUMX" in response.text
        assert "text/html" in response.headers["Content-Type"]

    def test_no_secrets_in_any_response(self, api, client):
        for route in ("status", "market", "agents", "decision", "risk",
                      "execution", "positions", "reconciliation", "events"):
            body = client.get(f"{api.url}/{route}").text.lower()
            assert "password" not in body
            assert "secret" not in body
            assert "authorization" not in body


class TestControlOperations:
    def test_start_stop(self, api, client):
        assert client.post(f"{api.url}/control/start").json()["ok"]
        assert client.get(f"{api.url}/status").json()["started"] is True
        assert client.post(f"{api.url}/control/stop").json()["ok"]
        assert client.get(f"{api.url}/status").json()["started"] is False

    def test_emergency_stop_engages_and_resets(self, api, client):
        client.post(f"{api.url}/control/start")
        assert client.post(f"{api.url}/control/emergency_stop").json()["ok"]
        status = client.get(f"{api.url}/status").json()
        assert status["emergency_stop_active"] is True
        # the banner halt blocks evaluation until reset
        assert client.get(f"{api.url}/decision").status_code == 200  # read still fine
        assert client.post(f"{api.url}/control/reset_kill_switch").json()[
            "emergency_stop_active"
        ] is False

    def test_dry_run_ratchet(self, runtime, fake_mt5, control_config):
        """The control plane can force DRY_RUN but never un-force it."""
        from app.brokers.mt5 import MT5Broker
        from app.control import EngineRuntime
        from tests.control.conftest import make_runtime_config

        # non-dry-run requires the config-level confirmation phrase
        # (AppConfig safety) — demo execution is demo+phrase+runtime check
        demo_config = make_runtime_config(
            dry_run=False,
            real_trading_confirmed="I ACCEPT REAL TRADING RISK",
        )
        broker = MT5Broker(mt5_module=fake_mt5)
        demo_runtime = EngineRuntime(demo_config, broker, clock=runtime.clock)
        demo_runtime.connect()
        assert demo_runtime.execution_service.mode is TradingMode.MT5_DEMO

        with LocalControlAPI(demo_runtime, host="127.0.0.1", port=0) as api:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                assert client.post(f"{api.url}/control/dry_run").json()["ok"]
                status = client.get(f"{api.url}/status").json()
                assert status["mode"] == "DRY_RUN"
                # there is no op to go back to demo mode
                response = client.post(f"{api.url}/control/demo")
                assert response.status_code == 404

    def test_unknown_control_op_404(self, api, client):
        assert client.post(f"{api.url}/control/execute").status_code == 404
        assert client.post(f"{api.url}/control/buy").status_code == 404

    def test_evaluate_requires_start(self, api, client, runtime):
        """The engine refuses cycles while stopped (409, not an error page)."""
        response = client.post(f"{api.url}/control/reconcile")  # allowed anytime
        assert response.status_code == 200
        # evaluate is not exposed as an endpoint at all — cycles are driven
        # by the future worker; the runtime method still enforces started:
        from app.control.errors import EngineNotStartedError

        with pytest.raises(EngineNotStartedError):
            runtime.evaluate_cycle()


class TestNoExecutionEndpoint:
    """The API is an interface to the engine, not an execution path (§29)."""

    def test_there_is_no_way_to_send_an_order_over_http(self, api, client):
        for path in ("/control/execute", "/control/order", "/control/trade",
                     "/execute", "/order", "/trade", "/control/buy",
                     "/control/sell"):
            assert client.post(f"{api.url}{path}").status_code == 404, path
            assert client.get(f"{api.url}{path}").status_code == 404, path

    def test_control_ops_cannot_touch_the_risk_gate(self, api, client, runtime):
        """No op bypasses the gate: even with a started engine, execution
        only happens through the runtime's explicit execute_approved path,
        which requires a gate APPROVED decision."""
        client.post(f"{api.url}/control/start")
        # no cycle has run -> nothing approved -> execution refused
        from app.control.errors import ExecutionRefused

        with pytest.raises(ExecutionRefused):
            runtime.execute_approved()


class TestServerBehavior:
    def test_binds_localhost_by_default(self):
        import inspect

        from app.control.api import LocalControlAPI as API

        signature = inspect.signature(API.__init__)
        assert signature.parameters["host"].default == "127.0.0.1"
        assert signature.parameters["port"].default == 8757

    def test_health(self, api, client):
        assert client.get(f"{api.url}/health").json() == {"ok": True}

    def test_reconcile_via_api(self, api, client, runtime):
        data = client.post(f"{api.url}/control/reconcile").json()
        assert "clean" in data and "counts" in data
        assert client.get(f"{api.url}/reconciliation").json()["status"] == "CLEAN"
