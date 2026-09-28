"""Agent registry — deterministic orchestration of all agents.

Runs every registered agent over a ``MarketContext`` in a stable order.
Agent failures are **isolated**: a raising agent becomes a NEUTRAL/INVALID
result with the error recorded — one broken agent never kills the analysis
cycle (and never becomes a reason to trade).
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import MarketContext
from app.agents.liquidity import LiquidityAgent
from app.agents.macro import MacroAgent
from app.agents.mean_reversion import MeanReversionAgent
from app.agents.momentum import MomentumAgent
from app.agents.structure import StructureAgent
from app.agents.trend import TrendAgent
from app.agents.volatility import VolatilityAgent
from app.core.enums import AgentDirection, DataQuality
from app.core.events import get_event_bus
from app.core.logging import get_logger

log = get_logger("agents.registry")


def default_agents() -> list[BaseAgent]:
    """The Phase-2 agent roster in deterministic order."""
    return [
        TrendAgent(),
        MomentumAgent(),
        StructureAgent(),
        LiquidityAgent(),
        VolatilityAgent(),
        MeanReversionAgent(),
        MacroAgent(),
    ]


class AgentRegistry:
    """Ordered, failure-isolated agent runner."""

    def __init__(self, agents: list[BaseAgent] | None = None) -> None:
        self._agents: list[BaseAgent] = list(agents) if agents is not None else default_agents()
        self._disabled: set[str] = set()

    # ------------------------------------------------------------------
    @property
    def agents(self) -> list[BaseAgent]:
        return list(self._agents)

    def names(self) -> list[str]:
        return [agent.name for agent in self._agents]

    def enable(self, name: str, enabled: bool = True) -> None:
        if enabled:
            self._disabled.discard(name)
        else:
            self._disabled.add(name)

    def is_enabled(self, name: str) -> bool:
        return name not in self._disabled

    # ------------------------------------------------------------------
    def run_all(self, context: MarketContext) -> list[AgentResult]:
        """Run all enabled agents.  Deterministic order, isolated failures."""
        results: list[AgentResult] = []
        for agent in self._agents:
            if agent.name in self._disabled:
                continue
            try:
                result = agent.analyze(context)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                log.error(
                    "agent raised during analysis",
                    event="AGENT_ERROR",
                    agent=agent.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                get_event_bus().emit(
                    "AGENT_ERROR",
                    component="agents.registry",
                    agent=agent.name,
                    error=str(exc),
                )
                result = AgentResult(
                    agent=agent.name,
                    direction=AgentDirection.NEUTRAL,
                    signal_strength=0.0,
                    primary_timeframe=getattr(agent, "primary_timeframe", None),
                    reasons=[],
                    warnings=[f"agent failed: {type(exc).__name__}: {exc}"],
                    data_quality=DataQuality.INVALID,
                    source_time=context.created_at,
                    snapshot_time=context.created_at,
                )
            results.append(result)
            get_event_bus().emit(
                "AGENT_SIGNAL",
                component="agents.registry",
                agent=result.agent,
                direction=result.direction.value,
                signal_strength=round(result.signal_strength, 4),
                data_quality=result.data_quality.value,
            )
        return results

    def run(self, name: str, context: MarketContext) -> AgentResult:
        """Run a single agent by name (raises KeyError when unknown)."""
        for agent in self._agents:
            if agent.name == name:
                return agent.analyze(context)
        raise KeyError(f"unknown agent: {name}")
