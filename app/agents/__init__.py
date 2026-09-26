"""Specialized analysis agents (Phase 2 — implemented).

Seven agents share one contract (``BaseAgent`` → ``AgentResult``) and consume
only a validated ``MarketContext`` built from a Phase-1 ``MarketSnapshot``:

* ``TrendAgent``        — EMA/ADX/slope trend classification (H1 + H4 macro)
* ``MomentumAgent``     — regime-aware RSI/MACD/ROC momentum (H1 + M15)
* ``StructureAgent``    — confirmed swings, HH/HL/LH/LL, BOS/CHOCH (H1 + H4)
* ``LiquidityAgent``    — wicks/sweeps/breakouts at levels (M15 + H1) — NOT order-book
* ``VolatilityAgent``   — ATR/BB classification (contextual, never directional)
* ``MeanReversionAgent``— range-gated statistical reversion (no martingale/DCA)
* ``MacroAgent``        — provider-based macro/news interface (NEUTRAL/NO_DATA by default)

Invariants (enforced by tests):

* agents never touch a broker, credentials, orders, the network or a wall clock;
* ``signal_strength`` is a documented deterministic evidence score — never a
  probability (spec §57);
* same context in ⇒ same result out (backtest-compatible);
* missing/invalid data ⇒ NEUTRAL with an explicit data-quality status.
"""

from app.agents.base import AgentResult, BaseAgent
from app.agents.context import (
    DEFAULT_ROLES,
    MarketContext,
    build_market_context,
)
from app.agents.features import (
    StructureEvent,
    SwingPoint,
    TimeframeFeatures,
    classify_volatility,
    compute_features,
)
from app.agents.liquidity import LiquidityAgent
from app.agents.macro import BLACKOUT_WINDOW, MacroAgent, MacroDataProvider, MacroEvent
from app.agents.mean_reversion import MeanReversionAgent
from app.agents.momentum import MomentumAgent
from app.agents.registry import AgentRegistry, default_agents
from app.agents.structure import StructureAgent
from app.agents.trend import TrendAgent
from app.agents.volatility import VolatilityAgent

__all__ = [
    "AgentRegistry",
    "AgentResult",
    "BaseAgent",
    "BLACKOUT_WINDOW",
    "DEFAULT_ROLES",
    "LiquidityAgent",
    "MacroAgent",
    "MacroDataProvider",
    "MacroEvent",
    "MarketContext",
    "MeanReversionAgent",
    "MomentumAgent",
    "StructureAgent",
    "StructureEvent",
    "SwingPoint",
    "TimeframeFeatures",
    "TrendAgent",
    "VolatilityAgent",
    "build_market_context",
    "classify_volatility",
    "compute_features",
    "default_agents",
]
