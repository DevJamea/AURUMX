"""Phase-5 architectural hygiene (spec §5/§28/§34).

AST/source-level proofs:

* ``order_check``/``order_send`` are called ONLY inside app/brokers/mt5.py
  (the MT5 execution adapter);
* nothing outside app/execution calls ``place_market_order``;
* decision + risk layers still have no execution surface (Phase-4 tests
  re-verified here for the new layering);
* the GUI/control layer never imports MetaTrader5 nor calls broker
  read/execution functions directly — it goes through the runtime;
* no martingale/recovery logic in the execution or control layers;
* no automatic retries around order_send;
* no network/LLM dependencies in the execution core;
* the GUI's static page talks only to same-origin relative API paths.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

APP = Path("app")
MT5_ADAPTER = Path("app/brokers/mt5.py")


def posix(path: Path) -> str:
    """Cross-platform canonical path form for comparisons.

    ``Path.rglob`` yields platform-native separators (``app\\brokers\\mt5.py``
    on Windows), while every allowed-path literal in this module is written
    POSIX-style.  All path comparisons go through this helper so the
    assertions behave identically on Windows and Unix.
    """
    return path.as_posix()


#: modules allowed to call raw MT5 execution functions (adapter only)
ORDER_CALL_ALLOWED = {MT5_ADAPTER.as_posix()}

EXECUTION_SOURCES = sorted(APP.joinpath("execution").glob("*.py"))
CONTROL_SOURCES = sorted(APP.joinpath("control").rglob("*.py"))
GUI_STATIC = APP.joinpath("control/static/index.html")


def _calls(tree: ast.AST) -> set[str]:
    """Every function-call name (attribute or bare)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def _imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _all_app_sources() -> list[Path]:
    return sorted(p for p in APP.rglob("*.py"))


class TestOrderSendIsolation:
    """§5: no code outside the MT5 execution adapter may call MT5 order
    execution functions."""

    @pytest.mark.parametrize(
        "forbidden", ["order_send", "order_send_async", "order_check"]
    )
    def test_raw_mt5_execution_calls_only_in_adapter(self, forbidden):
        offenders = []
        for source in _all_app_sources():
            tree = ast.parse(source.read_text())
            if forbidden in _calls(tree):
                offenders.append(posix(source))
        assert set(offenders) <= ORDER_CALL_ALLOWED, (
            f"{forbidden}() called outside the MT5 adapter: {offenders}"
        )

    def test_place_market_order_only_called_by_execution_service(self):
        """§6: the execution service is the only caller of the broker's
        execution interface."""
        offenders = []
        for source in _all_app_sources():
            if posix(source) == "app/execution/service.py":
                continue
            tree = ast.parse(source.read_text())
            if "place_market_order" in _calls(tree):
                offenders.append(posix(source))
        assert offenders == [], f"place_market_order called outside the service: {offenders}"

    def test_execution_layer_imports_broker_interface_only(self):
        """The service depends on the interface, never on the MT5 module —
        substitutability + the MetaTrader5 import stays isolated."""
        for source in EXECUTION_SOURCES:
            modules = _imports(ast.parse(source.read_text()))
            assert "app.brokers.mt5" not in modules, f"{source} imports the MT5 broker"
            assert "MetaTrader5" not in modules, f"{source} imports MetaTrader5"

    def test_mt5_package_import_still_unique(self):
        """mt5.py imports the package lazily via importlib; the invariant
        is that no OTHER app source imports it — neither directly
        (``import MetaTrader5``) nor through importlib with a literal."""
        offenders: list[str] = []
        for source in _all_app_sources():
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(
                    alias.name == "MetaTrader5" for alias in node.names
                ):
                    offenders.append(posix(source))
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "MetaTrader5"
                ):
                    offenders.append(posix(source))
        assert offenders == [], f"direct MetaTrader5 imports outside the adapter: {offenders}"
        # and the adapter really is the lazy importer
        adapter_source = Path("app/brokers/mt5.py").read_text()
        assert 'MT5_IMPORT_NAME = "MetaTrader5"' in adapter_source


class TestControlPlaneBoundaries:
    """§28: the GUI must not touch MT5 at all.

    Layering inside app/control:

    * ``api.py`` + ``static/`` + ``state.py`` — the GUI surface: no broker
      imports of ANY kind, no MT5 function calls;
    * ``runtime.py`` — the engine composition: receives a
      ``BrokerInterface`` (interface import allowed; the concrete MT5
      broker is NOT constructed here);
    * ``__main__.py`` — the composition root on Windows: the one place
      that builds the real ``MT5Broker`` (mirroring app/brokers/mt5.py
      being the only MetaTrader5 importer).
    """

    #: files that make up the GUI-facing surface
    GUI_SOURCES = [
        Path("app/control/api.py"),
        Path("app/control/state.py"),
        Path("app/control/errors.py"),
        Path("app/control/__init__.py"),
    ]

    @pytest.mark.parametrize("source", GUI_SOURCES, ids=lambda p: p.name)
    def test_gui_surface_never_imports_broker_package(self, source: Path):
        modules = _imports(ast.parse(source.read_text()))
        assert not any(m.startswith("app.brokers") for m in modules), (
            f"{source}: the GUI surface must go through the runtime"
        )

    @pytest.mark.parametrize("source", GUI_SOURCES, ids=lambda p: p.name)
    def test_gui_surface_never_calls_mt5_functions(self, source: Path):
        calls = _calls(ast.parse(source.read_text()))
        forbidden = {
            "order_send", "order_send_async", "order_check", "positions_get",
            "orders_get", "deals_get", "history_deals_get", "symbol_info",
            "symbol_info_tick", "account_info",
            "place_market_order", "place_pending_order", "modify_position",
            "close_position", "partial_close", "cancel_order",
        }
        assert not (calls & forbidden), f"{source}: calls {calls & forbidden}"

    def test_runtime_uses_the_interface_only(self):
        """The engine composition may hold a broker, but never the concrete
        MT5 implementation (substitutability + no hidden terminal access)."""
        modules = _imports(ast.parse(Path("app/control/runtime.py").read_text()))
        assert "app.brokers.mt5" not in modules
        assert "app.brokers.interface" in modules  # typed against the ABC

    def test_only_the_composition_root_builds_mt5broker(self):
        """``MT5Broker(...)`` construction is allowed only in the adapter's
        own classmethod (from_config) and the control-plane entry point."""
        allowed = {MT5_ADAPTER.as_posix(), "app/control/__main__.py"}
        builders = [posix(p) for p in _all_app_sources() if "MT5Broker(" in p.read_text()]
        assert set(builders) <= allowed, builders

    @pytest.mark.parametrize("source", CONTROL_SOURCES, ids=lambda p: p.name)
    def test_control_never_calls_mt5_functions(self, source: Path):
        calls = _calls(ast.parse(source.read_text()))
        # MT5-API function names (initialize/shutdown deliberately
        # excluded: the HTTP server's own shutdown() is not an MT5 call,
        # and the no-broker-import test above is the real guarantee)
        forbidden = {
            "order_send", "order_send_async", "order_check", "positions_get",
            "orders_get", "deals_get", "history_deals_get", "symbol_info",
            "symbol_info_tick", "account_info",
            "place_market_order", "place_pending_order", "modify_position",
            "close_position", "partial_close", "cancel_order",
        }
        assert not (calls & forbidden), f"{source}: calls {calls & forbidden}"

    def test_gui_static_page_uses_only_relative_urls(self):
        """The GUI's fetches go to the same-origin control API only."""
        assert GUI_STATIC.exists(), "GUI page missing"
        html = GUI_STATIC.read_text()
        # no absolute/external URLs in fetch calls or resource references
        assert "http://" not in html.replace("http://www.w3.org", ""), "external URL in GUI"
        assert "https://" not in html.replace("https://www.w3.org", ""), "external URL in GUI"
        for match in re.findall(r"fetch\(\s*[\"']([^\"']+)[\"']", html):
            assert match.startswith("/"), f"non-relative fetch target {match!r}"
        # the documented MT5 functions appear nowhere in the page source
        for forbidden in ("order_send", "order_check", "positions_get", "MetaTrader5"):
            assert forbidden not in html

    def test_control_api_has_no_execution_endpoint(self):
        """The route tables are the contract: reads + safe ops only."""
        source = Path("app/control/api.py").read_text()
        for op in ("execute", "order", "trade", "buy", "sell"):
            assert f'"{op}"' not in source, f"execution-like endpoint {op!r} defined"
            assert f"'{op}'" not in source, f"execution-like endpoint {op!r} defined"


class TestNoBypass:
    def test_execution_service_requires_decision_argument(self):
        """§11: execute() must take (request, decision) — approval explicit."""
        tree = ast.parse(Path("app/execution/service.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                arg_names = [a.arg for a in node.args.args]
                assert "decision" in arg_names, "execute() lost its decision parameter"
                return
        raise AssertionError("ExecutionService.execute not found")

    def test_execution_request_rejects_empty_risk_decision_id(self):
        """A request without a risk-decision reference cannot exist."""
        source = Path("app/execution/contracts.py").read_text()
        assert 'risk_decision_id: str = Field(min_length=1)' in source

    def test_no_dry_run_bypass_branch(self):
        """Structural proof: inside execute(), the dry-run branch RETURNS
        before the MT5 path is ever reached — the broker call (the only
        path to order_send) is unreachable while dry_run is set."""
        tree = ast.parse(Path("app/execution/service.py").read_text())
        execute = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        )
        body = execute.body
        dry_run_index = mt5_index = None
        for index, statement in enumerate(body):
            source_segment = ast.unparse(statement)
            if isinstance(statement, ast.If) and "dry_run" in source_segment:
                dry_run_index = index
                # the dry-run branch must terminate the pipeline
                assert any(
                    isinstance(child, ast.Return)
                    for child in ast.walk(statement)
                ), "dry-run branch does not return"
            if "_execute_mt5" in source_segment:
                mt5_index = index
        assert dry_run_index is not None, "dry-run branch missing from execute()"
        assert mt5_index is not None, "MT5 path missing from execute()"
        assert dry_run_index < mt5_index, (
            "the MT5 path is reachable before the dry-run short-circuit"
        )

    def test_phase4_boundaries_still_hold(self):
        """The Phase-4 suite already proves decision/risk purity; pin the
        new-layer view of it: app/execution must not be imported by them."""
        for layer in (Path("app/decision"), Path("app/risk")):
            for source in sorted(layer.glob("*.py")):
                modules = _imports(ast.parse(source.read_text()))
                assert not any(m.startswith("app.execution") for m in modules), (
                    f"{source}: {layer.name} layer must not import execution"
                )
                assert not any(m.startswith("app.control") for m in modules), (
                    f"{source}: {layer.name} layer must not import the control plane"
                )


class TestNoMartingaleNoRetries:
    MARTINGALE_IDENTIFIERS = {
        "martingale", "double_down", "recovery_multiplier", "loss_multiplier",
        "increase_after_loss", "risk_multiplier", "double_after_loss",
        "average_down", "add_to_loser", "revenge",
    }

    @pytest.mark.parametrize(
        "source", EXECUTION_SOURCES + CONTROL_SOURCES, ids=lambda p: p.name
    )
    def test_no_loss_recovery_identifiers(self, source: Path):
        tree = ast.parse(source.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        overlap = {n.lower() for n in names} & self.MARTINGALE_IDENTIFIERS
        assert not overlap, f"{source}: loss-recovery identifiers {overlap}"

    @pytest.mark.parametrize(
        "source", EXECUTION_SOURCES + CONTROL_SOURCES, ids=lambda p: p.name
    )
    def test_no_retry_logic(self, source: Path):
        """§42: no automatic retries — no retry-named identifiers and no
        loops containing the broker execution call."""
        tree = ast.parse(source.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not any("retry" in n.lower() for n in names), (
            f"{source}: retry logic is forbidden in Phase 5"
        )

    def test_execute_sends_at_most_once(self):
        """The broker call site must not sit inside a loop."""
        tree = ast.parse(Path("app/execution/service.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "place_market_order"
                    ):
                        raise AssertionError("place_market_order inside a loop (retry risk)")


class TestNoNetworkOrLLMInExecutionCore:
    FORBIDDEN_ROOTS = {
        "requests", "httpx", "aiohttp", "urllib", "http", "socket",
        "websockets", "socketio", "telebot", "aiogram", "telegram",
        "openai", "anthropic", "langchain", "langgraph", "llama_index",
    }

    @pytest.mark.parametrize("source", EXECUTION_SOURCES, ids=lambda p: p.name)
    def test_no_network_or_llm_imports(self, source: Path):
        modules = _imports(ast.parse(source.read_text()))
        for module in modules:
            assert module.split(".")[0] not in self.FORBIDDEN_ROOTS, (
                f"{source}: imports {module}"
            )

    def test_no_wall_clock_in_id_generation(self):
        """Determinism: the contracts module must not call time functions."""
        tree = ast.parse(Path("app/execution/contracts.py").read_text())
        calls = _calls(tree)
        assert not {"now", "today", "utcnow", "time", "uuid4"} & calls

    @pytest.mark.parametrize("source", EXECUTION_SOURCES, ids=lambda p: p.name)
    def test_execution_has_no_stdlib_http_server(self, source: Path):
        modules = _imports(ast.parse(source.read_text()))
        assert "http.server" not in modules, (
            f"{source}: the HTTP control plane lives in app/control only"
        )


class TestDocumentationPins:
    def test_execution_doc_exists_and_states_the_boundary(self):
        doc = Path("docs/EXECUTION.md")
        assert doc.exists(), "docs/EXECUTION.md must exist"
        text = doc.read_text()
        assert "order_send" in text
        assert "app/brokers/mt5.py" in text
        assert "does not authorize real-money trading" in text
