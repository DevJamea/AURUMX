"""Decision engine (Phase 2: regime + synthesis contract; Phase 3: full engine).

Implemented so far:

* ``regime``    — deterministic regime detection (TREND_UP/DOWN, RANGE,
  HIGH/LOW_VOLATILITY, UNCERTAIN) + the per-agent relevance matrix that
  decides which agents may influence decisions under each regime.
* ``synthesis`` — the input/output contract aggregating agent evidence while
  **preserving disagreement** (supporting/opposing stances, conflicts).  The
  reference aggregation is deterministic.
* Phase 3 additions: ``config`` (all thresholds in one place), ``alignment``
  (H4/H1/M15 agreement), ``levels`` (deterministic SL/TP hierarchy),
  ``proposal`` (pure TradeProposal + entry-price providers), ``engine``
  (the full BUY/SELL/HOLD/ABORT gate pipeline — analysis only, NEVER
  execution), ``journal`` (answerability: every evaluation recordable).
"""

from app.decision.regime import (
    AGENT_RELEVANCE_WEIGHTS,
    REGIME_RELEVANCE,
    RegimeAssessment,
    RegimeDetector,
    relevance_for,
    relevance_of,
    weighted_strength,
)
from app.decision.synthesis import (
    ACTION_NET_THRESHOLD,
    AgentStance,
    SynthesisInput,
    SynthesisOutput,
    synthesize,
)

__all__ = [
    "ACTION_NET_THRESHOLD",
    "AGENT_RELEVANCE_WEIGHTS",
    "AgentStance",
    "REGIME_RELEVANCE",
    "RegimeAssessment",
    "RegimeDetector",
    "SynthesisInput",
    "SynthesisOutput",
    "relevance_for",
    "relevance_of",
    "synthesize",
    "weighted_strength",
]


# ---- Phase 3: decision engine + proposal + journal ------------------------
from app.decision.config import DecisionEngineConfig, TimeframeWeights
from app.decision.engine import Decision, DecisionEngine, GateResult
from app.decision.journal import DecisionRecord, InMemoryDecisionJournal
from app.decision.levels import LevelPlan, compute_levels, validate_levels
from app.decision.proposal import (
    TickEntryProvider,
    TradeProposal,
)

__all__ = [
    "Decision",
    "DecisionEngine",
    "DecisionEngineConfig",
    "DecisionRecord",
    "GateResult",
    "InMemoryDecisionJournal",
    "LevelPlan",
    "TickEntryProvider",
    "TimeframeWeights",
    "TradeProposal",
    "compute_levels",
    "validate_levels",
]
