"""Local Control API (Phase 5E) — stdlib HTTP server, zero new deps.

Serves:

* ``GET /``                      the GUI (static page served by this process)
* ``GET /status``                connection/account/mode/halts/tick overview
* ``GET /market``                latest validated market snapshot summary
* ``GET /agents``                last cycle's agent results
* ``GET /decision``              last decision record summary
* ``GET /risk``                  last risk-gate decision summary
* ``GET /execution``             recent execution journal records
* ``GET /positions``             open positions (broker read via the engine)
* ``GET /reconciliation``        guard status + last report
* ``GET /events``                recent bus events (journal / recent events)
* ``POST /control/start|stop|dry_run|emergency_stop|reset_kill_switch|
        reconcile|acknowledge_reconciliation``

Safety (spec §5/§28/§29):

* there is NO endpoint that sends an order — the API is an interface to
  the engine, not an execution path;
* the handler layer never touches the broker or MT5 directly; it only
  calls ``EngineRuntime`` methods;
* binds to 127.0.0.1 by default (local Windows control plane);
* responses are JSON, credential-free by construction;
* Phase 5 is localhost-only without authentication — remote exposure
  requires the Phase-7 API with auth (documented limitation).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from app.control.errors import ControlError
from app.control.runtime import EngineRuntime
from app.core.logging import get_logger

log = get_logger("control.api")

_STATIC_DIR = Path(__file__).parent / "static"

_GET_ROUTES = (
    "status", "market", "agents", "decision", "risk",
    "execution", "positions", "reconciliation", "events",
)

_POST_ROUTES = (
    "start", "stop", "dry_run", "emergency_stop",
    "reset_kill_switch", "reconcile", "acknowledge_reconciliation",
)


class _Handler(BaseHTTPRequestHandler):
    """Closed-over the runtime; one instance per request (http.server)."""

    runtime: EngineRuntime  # injected via the server factory

    # ---- plumbing -------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("control api request", event="CONTROL_HTTP", detail=format % args)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- GET --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/":
                page = (_STATIC_DIR / "index.html").read_bytes()
                self._send_html(page)
                return
            if path == "/health":
                self._send_json({"ok": True})
                return
            if path.lstrip("/") in _GET_ROUTES:
                self._send_json(self._route_get(path.lstrip("/")))
                return
            self._send_json({"error": f"unknown route {path}"}, status=404)
        except ControlError as exc:
            self._send_json({"error": str(exc)}, status=409)
        except Exception as exc:  # noqa: BLE001 - handler must never crash the server
            log.error("control api error", event="CONTROL_HTTP_ERROR", error=str(exc))
            self._send_json({"error": str(exc)}, status=500)

    def _route_get(self, route: str) -> dict:
        runtime = self.runtime
        if route == "status":
            return runtime.status()
        if route == "market":
            return runtime.market()
        if route == "agents":
            return runtime.agents()
        if route == "decision":
            return runtime.decision_summary()
        if route == "risk":
            return runtime.risk_summary()
        if route == "execution":
            return runtime.execution_summary()
        if route == "positions":
            return runtime.positions()
        if route == "reconciliation":
            return runtime.reconciliation_summary()
        return runtime.events()  # route == "events"

    # ---- POST -------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0].rstrip("/")
        route = path.lstrip("/").removeprefix("control/")
        try:
            if route not in _POST_ROUTES:
                self._send_json({"error": f"unknown control op {path}"}, status=404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            payload: dict = {}
            if length:
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send_json({"error": "invalid JSON body"}, status=400)
                    return
            self._send_json(self._route_post(route, payload))
        except ControlError as exc:
            self._send_json({"error": str(exc)}, status=409)
        except Exception as exc:  # noqa: BLE001 - handler must never crash the server
            log.error("control api error", event="CONTROL_HTTP_ERROR", error=str(exc))
            self._send_json({"error": str(exc)}, status=500)

    def _route_post(self, route: str, payload: dict) -> dict:
        runtime = self.runtime
        if route == "start":
            runtime.control.start()
        elif route == "stop":
            runtime.control.stop()
        elif route == "dry_run":
            runtime.force_dry_run()
        elif route == "emergency_stop":
            runtime.control.engage_emergency_stop(
                reason=str(payload.get("reason") or "operator")
            )
        elif route == "reset_kill_switch":
            return runtime.control.reset_kill_switch()
        elif route == "reconcile":
            return runtime.reconcile().summary()
        elif route == "acknowledge_reconciliation":
            ok = runtime.acknowledge_reconciliation(
                reason=str(payload.get("reason") or "operator")
            )
            if not ok:
                raise ControlError(
                    "nothing to acknowledge (no mismatched report, or already clean)"
                )
        return {"ok": True, "op": route}


class LocalControlAPI:
    """The local control-plane server (GUI + JSON API in one process)."""

    def __init__(
        self,
        runtime: EngineRuntime,
        *,
        host: str = "127.0.0.1",
        port: int = 8757,
    ) -> None:
        self._runtime = runtime
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        if self._server is not None:
            return self._server.server_address[1]
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return

        handler = type(
            "BoundHandler",
            (_Handler,),
            {"runtime": self._runtime, "protocol_version": "HTTP/1.1"},
        )
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._server.daemon_threads = True
        import threading

        thread = threading.Thread(
            target=self._server.serve_forever,
            name="aurumx-control-api",
            daemon=True,
        )
        thread.start()
        log.info(
            "control plane listening",
            event="CONTROL_API_STARTED",
            url=self.url,
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            log.info("control plane stopped", event="CONTROL_API_STOPPED")

    def __enter__(self) -> LocalControlAPI:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


__all__ = ["LocalControlAPI"]
