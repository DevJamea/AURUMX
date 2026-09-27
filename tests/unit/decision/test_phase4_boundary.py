"""Phase-3 / Phase-4 boundary tests (hardening §4/§5/§6).

Phase 3 = decision correctness (may PROPOSE).
Phase 4 = independent safety barrier (may BLOCK).
These tests pin the seam: the decision layer must not be able to reach the
RiskGate implementation, an order, a broker, a network socket or the wall
clock — even by accident.
"""

from __future__ import annotations

import ast
from pathlib import Path

DECISION_SOURCES = sorted(Path("app/decision").glob("*.py")) + [Path("app/decision/__init__.py")]
RISK_SOURCES = sorted(Path("app/risk").glob("*.py"))


def _module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestPhase3CannotReachTheRiskGate:
    def test_decision_layer_never_imports_the_gate(self):
        for source in DECISION_SOURCES:
            modules = _module_names(ast.parse(source.read_text()))
            assert not any("risk.gate" in m for m in modules), (
                f"{source}: decision layer must not import the Phase-4 gate"
            )

    def test_decision_layer_never_names_the_gate(self):
        for source in DECISION_SOURCES:
            text = source.read_text()
            assert "RiskGate" not in text, f"{source}: RiskGate referenced in decision layer"
            assert "RiskDecision" not in text, f"{source}: RiskDecision referenced in decision layer"
            assert "RiskAction" not in text, f"{source}: RiskAction referenced in decision layer"

    def test_decision_layer_may_use_pure_risk_math_only(self):
        """app.decision imports from app.risk are limited to the pure
        sizing/state modules — never the gate contract."""
        allowed = {"app.risk.sizing", "app.risk.state"}
        for source in DECISION_SOURCES:
            for module in _module_names(ast.parse(source.read_text())):
                if module.startswith("app.risk"):
                    assert module in allowed, f"{source}: unexpected app.risk import {module}"


class TestPhase3CannotExecute:
    EXECUTION_CALLS = {
        "order_send", "order_send_async", "order_check", "positions_add",
        "orders_add", "modify_position", "close_position", "cancel_order",
        "place_order", "send_order", "execute", "submit_order",
    }

    def test_no_execution_calls_in_decision_or_risk_layers(self):
        for source in DECISION_SOURCES + RISK_SOURCES:
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert node.name not in self.EXECUTION_CALLS, (
                        f"{source}: defines execution method {node.name}()"
                    )
                if isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "attr", None) or getattr(func, "id", None)
                    assert name not in self.EXECUTION_CALLS, (
                        f"{source}: calls {name}()"
                    )

    def test_trade_proposal_has_no_execution_surface(self):
        from app.decision.proposal import TradeProposal

        for forbidden in ("send", "execute", "place", "submit", "cancel", "modify"):
            assert not hasattr(TradeProposal, forbidden), f"TradeProposal.{forbidden} exists"

    def test_broker_package_not_imported_by_decision_or_risk(self):
        for source in DECISION_SOURCES + RISK_SOURCES:
            modules = _module_names(ast.parse(source.read_text()))
            assert not any(m.startswith("app.brokers") for m in modules), (
                f"{source}: imports the broker package"
            )


class TestNoMt5NoNetworkNoWallClock:
    FORBIDDEN_ROOTS = {
        "MetaTrader5", "mt5", "requests", "urllib", "http", "socket",
        "httpx", "websockets", "socketio", "telebot", "aiogram",
    }
    WALL_CLOCK_CALLS = {"datetime.now", "datetime.today", "datetime.utcnow",
                        "time.time", "time.monotonic", "time.localtime"}

    def test_no_broker_network_or_asyncio_imports(self):
        for source in DECISION_SOURCES + RISK_SOURCES:
            for module in _module_names(ast.parse(source.read_text())):
                root = module.split(".")[0]
                assert root not in self.FORBIDDEN_ROOTS, f"{source}: imports {module}"

    def test_no_wall_clock_calls(self):
        for source in DECISION_SOURCES + RISK_SOURCES:
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    dotted = ast.unparse(node.func)
                    assert dotted not in self.WALL_CLOCK_CALLS, f"{source}: calls {dotted}()"

    def test_gate_contract_is_pure(self):
        """The Phase-4 contract itself must stay backtest-safe (it will be
        evaluated inside backtests too)."""
        source = Path("app/risk/gate.py")
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert ast.unparse(node.func) not in self.WALL_CLOCK_CALLS
        assert not any(
            m.startswith(("MetaTrader5", "requests", "httpx", "socket"))
            for m in _module_names(tree)
        )


class TestBoundaryDocumentation:
    def test_pipeline_order_is_documented(self):
        """The DecisionEngine docstring pins the Phase-4 hand-off."""
        text = Path("app/decision/engine.py").read_text()
        assert "RiskGate" in text or "risk gate" in text.lower()

    def test_gate_contract_documents_required_checks(self):
        text = Path("app/risk/gate.py").read_text()
        for check in ("max_risk_per_trade", "emergency_stop", "kill_switch",
                      "symbol_restriction", "volume_limits"):
            assert check in text
