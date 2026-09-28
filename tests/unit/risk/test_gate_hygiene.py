"""RiskGate source hygiene (Phase 4 §13/§26/§28).

The safety barrier must be provably free of execution, network, wall-clock,
randomness — and free of any martingale/loss-recovery logic.  AST-level
checks (immune to docstring mentions).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RISK_SOURCES = sorted(Path("app/risk").glob("*.py"))

FORBIDDEN_IMPORT_ROOTS = {
    "MetaTrader5", "mt5", "requests", "httpx", "aiohttp", "urllib", "http",
    "socket", "websockets", "telebot", "aiogram", "telegram",
    "random", "secrets", "uuid",
}

FORBIDDEN_CALLS = {
    "order_send", "order_send_async", "order_check", "positions_add",
    "modify_position", "close_position", "cancel_order", "place_order",
    "send_order", "submit_order",
    "datetime.now", "datetime.today", "datetime.utcnow",
    "time.time", "time.monotonic", "time.sleep",
    "random.random", "random.randint", "random.uniform",
    "uuid.uuid4",
    "open", "eval", "exec",
}

MARTINGALE_IDENTIFIERS = {
    "martingale", "double_down", "recovery_multiplier", "loss_multiplier",
    "increase_after_loss", "risk_multiplier", "double_after_loss",
    "average_down", "add_to_loser", "revenge",
}


class TestNoExecutionInRiskLayer:
    @pytest.mark.parametrize("source", RISK_SOURCES, ids=lambda p: p.name)
    def test_no_mt5_network_or_randomness_imports(self, source: Path):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS, (
                        f"{source.name}: imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS, (
                    f"{source.name}: imports from {node.module}"
                )

    @pytest.mark.parametrize("source", RISK_SOURCES, ids=lambda p: p.name)
    def test_no_execution_or_wallclock_calls(self, source: Path):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                dotted = ast.unparse(node.func)
                assert dotted not in FORBIDDEN_CALLS, f"{source.name}: calls {dotted}()"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, (
                    f"{source.name}: calls {node.func.id}()"
                )

    @pytest.mark.parametrize("source", RISK_SOURCES, ids=lambda p: p.name)
    def test_no_execution_package_import(self, source: Path):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                assert not module.startswith("app.execution"), (
                    f"{source.name}: must not import execution ({module})"
                )
                assert not module.startswith("app.brokers"), (
                    f"{source.name}: must not import the broker layer ({module})"
                )


class TestNoMartingaleAnywhere:
    @pytest.mark.parametrize("source", RISK_SOURCES, ids=lambda p: p.name)
    def test_no_loss_recovery_identifiers(self, source: Path):
        """§13: no martingale / recovery / risk-increase-after-loss logic —
        the identifiers must not even exist in the risk layer."""
        tree = ast.parse(source.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for identifier in names:
            assert identifier.lower() not in MARTINGALE_IDENTIFIERS, (
                f"{source.name}: loss-recovery identifier '{identifier}'"
            )

    def test_gate_has_no_loss_history_risk_modifier(self):
        """The gate's risk math inputs are geometry + volume + spec — the
        consecutive-loss state can only BLOCK, never scale risk."""
        source = Path("app/risk/engine.py").read_text()
        risk_fn = source[source.index("def _check_max_risk_per_trade"):]
        risk_fn = risk_fn[: risk_fn.index("def _check_max_total_exposure")]
        assert "consecutive_losses" not in risk_fn
        assert "daily_loss" not in risk_fn

    def test_gate_config_has_no_loss_modifiers(self):
        from app.risk import RiskGateConfig

        for field_name in RiskGateConfig.model_fields:
            assert field_name.lower() not in MARTINGALE_IDENTIFIERS


class TestGateCannotMutateProposals:
    """§22 at the source level: the gate never assigns to proposal attributes."""

    @pytest.mark.parametrize("source", RISK_SOURCES, ids=lambda p: p.name)
    def test_no_attribute_assignment_on_proposal(self, source: Path):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        # 'proposal.<attr> = ...' or common parameter names
                        if target.value.id in ("proposal", "p", "trade"):
                            raise AssertionError(
                                f"{source.name}: assigns {ast.unparse(target)} — the gate "
                                "must never modify a proposal"
                            )


class TestPurityByReimport:
    def test_gate_module_imports_cleanly_without_side_effects(self):
        """Importing the risk layer twice must be side-effect free (no
        registration, no connections, no global mutation beyond caches)."""
        import importlib

        import app.risk as risk_package

        importlib.reload(risk_package)
        importlib.reload(risk_package)
        assert risk_package.HardRiskGate is not None
